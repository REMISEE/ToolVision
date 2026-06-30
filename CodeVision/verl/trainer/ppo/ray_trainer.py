# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import gc
import os
import random
import re
import subprocess
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.tracking import MultimodalGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        # self.validation_generations_logger = ValidationGenerationsLogger(
        #     project_name=self.config.trainer.project_name,
        #     experiment_name=self.config.trainer.experiment_name,
        # )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.training_generations_logger = MultimodalGenerationsLogger(training=True)
        self.validation_generations_logger = MultimodalGenerationsLogger(training=False)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): RayPPOTrainer._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RayPPOTrainer._to_jsonable(v) for v in value]
        return value

    @staticmethod
    def _get_indexed(source, key, idx, default=None):
        values = source.get(key, None)
        if values is None or idx >= len(values):
            return default
        value = values[idx]
        return RayPPOTrainer._to_jsonable(value)

    @staticmethod
    def _git_rev(path):
        try:
            result = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        extra_dump_infos_dict=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        if extra_dump_infos_dict:
            for k, v in extra_dump_infos_dict.items():
                if len(v) == n and k not in base_data:
                    base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: self._to_jsonable(v[i]) for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _append_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        extra_dump_infos_dict=None,
        start_index=0,
        batch_index=None,
    ):
        """Append one validation batch to generations JSONL without retaining all previous batches."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
            "sample_index": list(range(start_index, start_index + n)),
        }
        if batch_index is not None:
            base_data["batch_index"] = [batch_index] * n

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        if extra_dump_infos_dict:
            for k, v in extra_dump_infos_dict.items():
                if len(v) == n and k not in base_data:
                    base_data[k] = v

        with open(filename, "a", encoding="utf-8") as handle:
            for i in range(n):
                entry = {k: self._to_jsonable(v[i]) for k, v in base_data.items()}
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return filename

    def _build_validation_diagnostic_records(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        extra_dump_infos_dict,
        raw_generation_path,
        sampled_path,
        start_index=0,
    ):
        records = []
        bucket_counts = defaultdict(int)
        for idx in range(len(outputs)):
            global_idx = start_index + idx
            data_source = str(self._get_indexed(extra_dump_infos_dict, "data_source", idx, "unknown"))
            uid = str(self._get_indexed(extra_dump_infos_dict, "uid", idx, global_idx))
            sample_id = f"{uid}::response_{global_idx}"
            extra_info = self._get_indexed(extra_dump_infos_dict, "extra_info", idx, {}) or {}
            benchmark = str(extra_info.get("source_benchmark") or data_source) if isinstance(extra_info, dict) else data_source
            score = float(scores[idx])
            used_tool = self._get_indexed(reward_extra_infos_dict, "used_tool", idx, None)
            if used_tool is None:
                used_tool = self._get_indexed(extra_dump_infos_dict, "used_any_tool", idx, None)
            if used_tool is None:
                used_tool = 1.0 if "<tool_call>" in str(outputs[idx]) else 0.0
            used_tool_bool = bool(float(used_tool))
            invalid_tool_call = bool(float(self._get_indexed(reward_extra_infos_dict, "invalid_tool_call", idx, 0.0) or 0.0))
            final_answer = str(self._get_indexed(reward_extra_infos_dict, "final_answer", idx, ""))
            tool_names = str(self._get_indexed(reward_extra_infos_dict, "tool_names", idx, ""))
            ocrbench_v2_payload = self._get_indexed(reward_extra_infos_dict, "ocrbench_v2_payload", idx, None)
            ocrbench_v2_question_type = self._get_indexed(reward_extra_infos_dict, "ocrbench_v2_question_type", idx, None)
            ocrbench_v2_language = self._get_indexed(reward_extra_infos_dict, "ocrbench_v2_language", idx, None)
            if not tool_names:
                tool_names = ",".join(sorted(set(re.findall(r'"name"\s*:\s*"([^"]+)"', str(outputs[idx])))))
            is_correct = score >= 1.0

            bucket_tags = []
            if not is_correct:
                bucket_tags.append("wrong_any")
            if invalid_tool_call:
                bucket_tags.append("invalid_tool_call")
            if used_tool_bool and not is_correct:
                bucket_tags.append("tool_used_wrong")
            if used_tool_bool and is_correct:
                bucket_tags.append("tool_used_correct")
            if (not used_tool_bool) and is_correct:
                bucket_tags.append("correct_no_tool")
            for tag in bucket_tags:
                bucket_counts[f"{benchmark}/{tag}"] += 1

            records.append(
                {
                    "uid": uid,
                    "sample_id": sample_id,
                    "benchmark": benchmark,
                    "data_source": data_source,
                    "model_path": str(self.config.actor_rollout_ref.model.path),
                    "exp_name": str(self.config.trainer.experiment_name),
                    "is_correct": is_correct,
                    "used_tool": used_tool_bool,
                    "tool_names": [name for name in tool_names.split(",") if name],
                    "invalid_tool_call": invalid_tool_call,
                    "num_turns": self._get_indexed(extra_dump_infos_dict, "__num_turns__", idx, None),
                    "final_answer": final_answer,
                    "ground_truth": gts[idx],
                    "score": score,
                    "output_path": None,
                    "raw_generation_path": raw_generation_path,
                    "trace_pack_path": sampled_path,
                    "bucket_tags": bucket_tags,
                    "ocrbench_v2_payload": ocrbench_v2_payload,
                    "ocrbench_v2_question_type": ocrbench_v2_question_type,
                    "ocrbench_v2_language": ocrbench_v2_language,
                }
            )
        return records, bucket_counts

    def _write_validation_diagnostics_manifest(
        self,
        dump_path,
        state,
        raw_generation_path=None,
        completed=False,
    ):
        os.makedirs(dump_path, exist_ok=True)
        summary_path = os.path.join(dump_path, "bucket_summary.json")
        manifest_path = os.path.join(dump_path, "manifest.json")
        metadata_path = os.path.join(dump_path, "metadata.jsonl")
        sampled_path = os.path.join(dump_path, "sampled_traces.jsonl")

        summary = {
            "total": int(state.get("total", 0)),
            "quadrants": dict(state.get("quadrant_counts", {})),
            "bucket_counts": dict(state.get("bucket_counts", {})),
            "sampled_trace_count": int(state.get("sampled_trace_count", 0)),
            "max_per_bucket": int(state.get("max_per_bucket", self.config.trainer.get("diagnostic_max_per_bucket", 50))),
            "completed": bool(completed),
            "streaming": True,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(self._to_jsonable(summary), handle, ensure_ascii=False, indent=2)

        lmms_eval_path = os.environ.get("LMMS_EVAL_PATH", "/mnt/cpfs/delinmao/lmms-eval")
        manifest = {
            "global_steps": self.global_steps,
            "model_path": str(self.config.actor_rollout_ref.model.path),
            "experiment_name": str(self.config.trainer.experiment_name),
            "diagnostics_dir": dump_path,
            "metadata_path": metadata_path,
            "sampled_traces_path": sampled_path,
            "raw_generation_path": raw_generation_path,
            "lmms_eval_path": lmms_eval_path,
            "lmms_eval_commit": self._git_rev(lmms_eval_path) if os.path.isdir(lmms_eval_path) else "",
            "ocrbench_v2_scorer_path": os.path.join(lmms_eval_path, "lmms_eval/tasks/ocrbench_v2/utils.py"),
            "completed": bool(completed),
            "streaming": True,
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self._to_jsonable(manifest), handle, ensure_ascii=False, indent=2)

    def _append_validation_diagnostics(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        state,
        extra_dump_infos_dict=None,
        raw_generation_path=None,
        start_index=0,
    ):
        """Append validation diagnostics for one batch and update summary state."""
        os.makedirs(dump_path, exist_ok=True)
        metadata_path = os.path.join(dump_path, "metadata.jsonl")
        sampled_path = os.path.join(dump_path, "sampled_traces.jsonl")
        extra_dump_infos_dict = extra_dump_infos_dict or {}

        records, batch_bucket_counts = self._build_validation_diagnostic_records(
            inputs=inputs,
            outputs=outputs,
            gts=gts,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            extra_dump_infos_dict=extra_dump_infos_dict,
            raw_generation_path=raw_generation_path,
            sampled_path=sampled_path,
            start_index=start_index,
        )

        state.setdefault("bucket_counts", defaultdict(int))
        state.setdefault("quadrant_counts", defaultdict(int))
        state.setdefault("sampled_per_bucket", defaultdict(int))
        state.setdefault("sampled_trace_count", 0)
        state.setdefault("total", 0)
        state.setdefault("max_per_bucket", int(self.config.trainer.get("diagnostic_max_per_bucket", 50)))
        full_trace = bool(self.config.trainer.get("save_full_trajectory_all", False))
        max_per_bucket = int(state["max_per_bucket"])

        with open(metadata_path, "a", encoding="utf-8") as metadata_handle, open(
            sampled_path, "a", encoding="utf-8"
        ) as sampled_handle:
            for idx, record in enumerate(records):
                metadata_handle.write(json.dumps(self._to_jsonable(record), ensure_ascii=False) + "\n")
                state["total"] += 1

                for tag in record["bucket_tags"]:
                    key = f"{record['benchmark']}/{tag}"
                    state["bucket_counts"][key] += 1

                if record["is_correct"] and record["used_tool"]:
                    state["quadrant_counts"]["correct_tool"] += 1
                elif record["is_correct"]:
                    state["quadrant_counts"]["correct_no_tool"] += 1
                elif record["used_tool"]:
                    state["quadrant_counts"]["wrong_tool"] += 1
                else:
                    state["quadrant_counts"]["wrong_no_tool"] += 1
                if record["invalid_tool_call"]:
                    state["quadrant_counts"]["invalid_tool_call"] += 1

                should_write_trace = full_trace
                selected_keys = []
                if not should_write_trace:
                    for tag in record["bucket_tags"]:
                        key = f"{record['benchmark']}/{tag}"
                        if state["sampled_per_bucket"][key] < max_per_bucket:
                            should_write_trace = True
                            selected_keys.append(key)
                    if should_write_trace and not selected_keys:
                        selected_keys = [f"{record['benchmark']}/{tag}" for tag in record["bucket_tags"]]
                if should_write_trace:
                    for key in selected_keys:
                        state["sampled_per_bucket"][key] += 1
                    trace = {
                        **record,
                        "input": inputs[idx],
                        "output": outputs[idx],
                        "extra_info": self._get_indexed(extra_dump_infos_dict, "extra_info", idx, {}),
                        "tool_reward_detail": self._get_indexed(extra_dump_infos_dict, "tool_reward_detail", idx, None),
                    }
                    sampled_handle.write(json.dumps(self._to_jsonable(trace), ensure_ascii=False) + "\n")
                    state["sampled_trace_count"] += 1

        for key, value in batch_bucket_counts.items():
            # batch_bucket_counts is already reflected in state, but keeping the variable
            # makes it easier to compare streaming and non-streaming accounting in logs.
            _ = (key, value)
        return metadata_path, sampled_path

    def _dump_validation_diagnostics(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        extra_dump_infos_dict=None,
        raw_generation_path=None,
    ):
        """Dump compact validation metadata plus capped trace samples."""
        os.makedirs(dump_path, exist_ok=True)
        metadata_path = os.path.join(dump_path, "metadata.jsonl")
        sampled_path = os.path.join(dump_path, "sampled_traces.jsonl")
        summary_path = os.path.join(dump_path, "bucket_summary.json")
        manifest_path = os.path.join(dump_path, "manifest.json")

        extra_dump_infos_dict = extra_dump_infos_dict or {}
        n = len(outputs)

        def _get(source, key, idx, default=None):
            values = source.get(key, None)
            if values is None or idx >= len(values):
                return default
            value = values[idx]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return value

        def _jsonable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            return value

        def _git_rev(path):
            try:
                result = subprocess.run(
                    ["git", "-C", path, "rev-parse", "HEAD"],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                return result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                return ""

        records = []
        bucket_counts = defaultdict(int)
        for idx in range(n):
            data_source = str(_get(extra_dump_infos_dict, "data_source", idx, "unknown"))
            uid = str(_get(extra_dump_infos_dict, "uid", idx, idx))
            sample_id = f"{uid}::response_{idx}"
            extra_info = _get(extra_dump_infos_dict, "extra_info", idx, {}) or {}
            benchmark = str(extra_info.get("source_benchmark") or data_source)
            score = float(scores[idx])
            used_tool = _get(reward_extra_infos_dict, "used_tool", idx, None)
            if used_tool is None:
                used_tool = _get(extra_dump_infos_dict, "used_any_tool", idx, None)
            if used_tool is None:
                used_tool = 1.0 if "<tool_call>" in str(outputs[idx]) else 0.0
            used_tool_bool = bool(float(used_tool))
            invalid_tool_call = bool(float(_get(reward_extra_infos_dict, "invalid_tool_call", idx, 0.0) or 0.0))
            final_answer = str(_get(reward_extra_infos_dict, "final_answer", idx, ""))
            tool_names = str(_get(reward_extra_infos_dict, "tool_names", idx, ""))
            ocrbench_v2_payload = _get(reward_extra_infos_dict, "ocrbench_v2_payload", idx, None)
            ocrbench_v2_question_type = _get(reward_extra_infos_dict, "ocrbench_v2_question_type", idx, None)
            ocrbench_v2_language = _get(reward_extra_infos_dict, "ocrbench_v2_language", idx, None)
            if not tool_names:
                tool_names = ",".join(sorted(set(re.findall(r'"name"\s*:\s*"([^"]+)"', str(outputs[idx])))))
            is_correct = score >= 1.0

            bucket_tags = []
            if not is_correct:
                bucket_tags.append("wrong_any")
            if invalid_tool_call:
                bucket_tags.append("invalid_tool_call")
            if used_tool_bool and not is_correct:
                bucket_tags.append("tool_used_wrong")
            if used_tool_bool and is_correct:
                bucket_tags.append("tool_used_correct")
            if (not used_tool_bool) and is_correct:
                bucket_tags.append("correct_no_tool")
            for tag in bucket_tags:
                bucket_counts[f"{benchmark}/{tag}"] += 1

            records.append(
                {
                    "uid": uid,
                    "sample_id": sample_id,
                    "benchmark": benchmark,
                    "data_source": data_source,
                    "model_path": str(self.config.actor_rollout_ref.model.path),
                    "exp_name": str(self.config.trainer.experiment_name),
                    "is_correct": is_correct,
                    "used_tool": used_tool_bool,
                    "tool_names": [name for name in tool_names.split(",") if name],
                    "invalid_tool_call": invalid_tool_call,
                    "num_turns": _get(extra_dump_infos_dict, "__num_turns__", idx, None),
                    "final_answer": final_answer,
                    "ground_truth": gts[idx],
                    "score": score,
                    "output_path": None,
                    "raw_generation_path": raw_generation_path,
                    "trace_pack_path": sampled_path,
                    "bucket_tags": bucket_tags,
                    "ocrbench_v2_payload": ocrbench_v2_payload,
                    "ocrbench_v2_question_type": ocrbench_v2_question_type,
                    "ocrbench_v2_language": ocrbench_v2_language,
                }
            )

        with open(metadata_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

        grouped = defaultdict(list)
        for idx, record in enumerate(records):
            for tag in record["bucket_tags"]:
                grouped[(record["benchmark"], tag)].append(idx)

        rng = random.Random(int(self.config.trainer.get("diagnostic_sample_seed", 42)))
        max_per_bucket = int(self.config.trainer.get("diagnostic_max_per_bucket", 50))
        selected = {}
        if bool(self.config.trainer.get("save_full_trajectory_all", False)):
            selected = {record["sample_id"]: idx for idx, record in enumerate(records)}
        else:
            for _, indices in sorted(grouped.items()):
                chosen = indices
                if len(chosen) > max_per_bucket:
                    chosen = rng.sample(chosen, max_per_bucket)
                for idx in chosen:
                    selected[records[idx]["sample_id"]] = idx

        with open(sampled_path, "w", encoding="utf-8") as handle:
            for idx in sorted(selected.values()):
                trace = {
                    **records[idx],
                    "input": inputs[idx],
                    "output": outputs[idx],
                    "extra_info": _get(extra_dump_infos_dict, "extra_info", idx, {}),
                    "tool_reward_detail": _get(extra_dump_infos_dict, "tool_reward_detail", idx, None),
                }
                handle.write(json.dumps(_jsonable(trace), ensure_ascii=False) + "\n")

        quadrant_counts = defaultdict(int)
        for record in records:
            if record["is_correct"] and record["used_tool"]:
                quadrant_counts["correct_tool"] += 1
            elif record["is_correct"]:
                quadrant_counts["correct_no_tool"] += 1
            elif record["used_tool"]:
                quadrant_counts["wrong_tool"] += 1
            else:
                quadrant_counts["wrong_no_tool"] += 1
            if record["invalid_tool_call"]:
                quadrant_counts["invalid_tool_call"] += 1
        summary = {
            "total": len(records),
            "quadrants": dict(quadrant_counts),
            "bucket_counts": dict(bucket_counts),
            "sampled_trace_count": len(selected),
            "max_per_bucket": max_per_bucket,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(summary), handle, ensure_ascii=False, indent=2)

        lmms_eval_path = os.environ.get("LMMS_EVAL_PATH", "/mnt/cpfs/delinmao/lmms-eval")
        manifest = {
            "global_steps": self.global_steps,
            "model_path": str(self.config.actor_rollout_ref.model.path),
            "experiment_name": str(self.config.trainer.experiment_name),
            "diagnostics_dir": dump_path,
            "metadata_path": metadata_path,
            "sampled_traces_path": sampled_path,
            "lmms_eval_path": lmms_eval_path,
            "lmms_eval_commit": _git_rev(lmms_eval_path) if os.path.isdir(lmms_eval_path) else "",
            "ocrbench_v2_scorer_path": os.path.join(lmms_eval_path, "lmms_eval/tasks/ocrbench_v2/utils.py"),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(manifest), handle, ensure_ascii=False, indent=2)
        print(f"Dumped validation diagnostics to {dump_path}")

    def _maybe_log_train_generations(self, batch: DataProto):
        """Log a table of training samples to the configured logger (wandb or swanlab)"""
        import random
        from PIL import Image
        generations_to_log = self.config.trainer.log_train_generations
        log_train_freq = self.config.trainer.get("log_train_freq", 1)

        if generations_to_log == 0 or self.global_steps % log_train_freq > 0:
            return
        prompts, response = batch.batch["prompts"], batch.batch["responses"]
        # Keep special tokens so visual placeholders (<|vision_start|><|image_pad|><|vision_end|>) are present for interleaving
        prompts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)
        response = self.tokenizer.batch_decode(response, skip_special_tokens=True)

        images = batch.non_tensor_batch.get("merged_images", None)
        if images is None:
            images = [None] * len(prompts)

        # Prefer samples whose Required transforms are non-empty
        import json, re
        total_n = len(prompts)
        res_ids = list(range(total_n))
        detail_arr = batch.non_tensor_batch.get("tool_reward_detail", None)
        req_arr = batch.non_tensor_batch.get("required_transforms", None)

        def _has_required_transforms(idx: int) -> bool:
            try:
                if detail_arr is not None and idx < len(detail_arr) and detail_arr[idx] is not None:
                    s = str(detail_arr[idx])
                    if "Required transforms" in s and not re.search(r"Required\s+transforms:\s*\[\s*\]", s):
                        return True
                if req_arr is not None and idx < len(req_arr):
                    v = req_arr[idx]
                    if v is None:
                        return False
                    if isinstance(v, str):
                        try:
                            val = json.loads(v)
                        except Exception:
                            return len(v.strip()) > 0
                    else:
                        val = v
                    if isinstance(val, dict):
                        return bool(val)
                    if isinstance(val, (list, tuple)):
                        return len(val) > 0
                    # Fallback: treat any non-empty scalar as present
                    return bool(val)
            except Exception:
                return False
            return False

        candidate_ids = [i for i in res_ids if _has_required_transforms(i)]
        if len(candidate_ids) >= generations_to_log:
            sample_ids = random.sample(candidate_ids, generations_to_log)
        elif len(candidate_ids) > 0:
            sample_ids = candidate_ids
            remaining = [i for i in res_ids if i not in set(sample_ids)]
            need = max(0, generations_to_log - len(sample_ids))
            if need > 0 and len(remaining) > 0:
                sample_ids += random.sample(remaining, min(need, len(remaining)))
        else:
            sample_ids = random.sample(res_ids, generations_to_log)

        sample_inputs = []
        sample_outputs = []
        sample_images = []
        sample_scores = []
        for idx in sample_ids:
            sample_inputs.append(prompts[idx])
            sample_outputs.append(response[idx])
            sample_images.append(images[idx])
            # Build score string: tool/reward details + optional ground truth
            score_detail = ""
            detail = batch.non_tensor_batch.get("tool_reward_detail", None)
            if detail is not None and len(detail) > idx and detail[idx] is not None:
                score_detail = str(detail[idx])

            gt_val = None
            try:
                # Prefer per-item access for robustness
                item = batch[idx]
                rm = item.non_tensor_batch.get("reward_model", {}) if hasattr(item, "non_tensor_batch") else {}
                if isinstance(rm, dict):
                    gt_val = rm.get("ground_truth", None)
                if gt_val is None and hasattr(item, "non_tensor_batch"):
                    gt_val = item.non_tensor_batch.get("ground_truth", None)
                # Normalize numpy scalars/arrays
                try:
                    import numpy as _np
                    if isinstance(gt_val, _np.ndarray):
                        if gt_val.size == 1:
                            gt_val = gt_val.item()
                        else:
                            gt_val = gt_val.tolist()
                    elif isinstance(gt_val, _np.generic):
                        gt_val = gt_val.item()
                except Exception:
                    pass
            except Exception:
                gt_val = None

            if gt_val is not None and gt_val != "":
                score = (score_detail + " || " if score_detail else "") + f"GT: {gt_val}"
                sample_scores.append(score)
            else:
                sample_scores.append(score_detail)

        samples = list(zip(sample_inputs, sample_outputs, sample_images, sample_scores))

        # Log to each configured logger
        self.training_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _maybe_log_val_generations(self, inputs, outputs, scores, images=None, detail_arr=None, ground_truths=None):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np
        import random
        import json
        import re

        # Prefer samples with non-empty Required transforms; fetch images/details cached in _validate
        sample_inputs = inputs
        sample_outputs = outputs
        sample_scores = scores  # may be numeric; we will rebuild detailed string when possible
        sample_images = images
        if sample_images is None or len(sample_images) != len(sample_inputs):
            sample_images = [None] * len(sample_inputs)

        # Build detail strings if we cached them in non_tensor_batch during validation
        gt_arr = ground_truths

        detailed_scores = []
        for i in range(len(sample_inputs)):
            detail = ""
            if detail_arr is not None and i < len(detail_arr) and detail_arr[i] is not None:
                detail = str(detail_arr[i])
            gt_val = None
            if gt_arr is not None and i < len(gt_arr):
                gt_val = gt_arr[i]
            if gt_val is not None:
                detailed_scores.append((detail + " || " if detail else "") + f"GT: {gt_val}")
            else:
                detailed_scores.append(detail if detail else str(sample_scores[i]) if i < len(sample_scores) else "")

        # Filter preferred samples by non-empty Required transforms
        def _has_required_transforms_from_detail(s: str) -> bool:
            try:
                if "Used transforms" in s and not re.search(r"Used\s+transforms:\s*\[\s*\]", s):
                    return True
            except Exception:
                pass
            return False

        res_ids = list(range(len(sample_inputs)))
        candidate_ids = [i for i in res_ids if _has_required_transforms_from_detail(detailed_scores[i])]
        if len(candidate_ids) >= generations_to_log:
            sample_ids = random.sample(candidate_ids, generations_to_log)
        elif len(candidate_ids) > 0:
            sample_ids = candidate_ids
            remaining = [i for i in res_ids if i not in set(sample_ids)]
            need = max(0, generations_to_log - len(sample_ids))
            if need > 0 and len(remaining) > 0:
                sample_ids += random.sample(remaining, min(need, len(remaining)))
        else:
            # Deterministic subset for validation
            rng = np.random.RandomState(42)
            sample_ids = list(range(len(sample_inputs)))
            rng.shuffle(sample_ids)
            sample_ids = sample_ids[:generations_to_log]

        # Compose samples with images list for wandb visualization
        samples = list(
            zip(
                [sample_inputs[i] for i in sample_ids],
                [sample_outputs[i] for i in sample_ids],
                [sample_images[i] for i in sample_ids],
                [detailed_scores[i] for i in sample_ids],
            )
        )

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        diagnostics_dir = self.config.trainer.get("diagnostics_dir", None)
        stream_validation_dump = bool(
            self.config.trainer.get("stream_validation_dump", bool(val_data_dir or diagnostics_dir))
        )
        raw_generation_path = os.path.join(val_data_dir, f"{self.global_steps}.jsonl") if val_data_dir else None

        # Full validation can be huge for rollout8. In streaming mode, keep only
        # lightweight arrays needed for metrics and append heavy traces per batch.
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        val_images_agg = []
        val_details_agg = []
        val_dump_extra_infos_dict: dict[str, list] = defaultdict(list)

        log_inputs = []
        log_outputs = []
        log_scores = []
        log_images = []
        log_details = []
        log_gts = []
        max_log_samples = int(self.config.trainer.get("log_val_generations", 0) or 0)

        diag_state = None
        if stream_validation_dump:
            if val_data_dir:
                os.makedirs(val_data_dir, exist_ok=True)
                if raw_generation_path and os.path.exists(raw_generation_path):
                    os.remove(raw_generation_path)
            if diagnostics_dir:
                os.makedirs(diagnostics_dir, exist_ok=True)
                for name in ["metadata.jsonl", "sampled_traces.jsonl", "bucket_summary.json", "manifest.json"]:
                    path = os.path.join(diagnostics_dir, name)
                    if os.path.exists(path):
                        os.remove(path)
                diag_state = {
                    "bucket_counts": defaultdict(int),
                    "quadrant_counts": defaultdict(int),
                    "sampled_per_bucket": defaultdict(int),
                    "sampled_trace_count": 0,
                    "total": 0,
                    "max_per_bucket": int(self.config.trainer.get("diagnostic_max_per_bucket", 50)),
                }
                self._write_validation_diagnostics_manifest(
                    dump_path=diagnostics_dir,
                    state=diag_state,
                    raw_generation_path=raw_generation_path,
                    completed=False,
                )
            print("stream_validation_dump=True: validation generations/diagnostics will be appended per batch")

        def _as_list(values, n=None, default=None):
            if values is None:
                return [default] * int(n or 0)
            try:
                out = list(values)
            except Exception:
                out = [values]
            if n is not None and len(out) != n:
                if len(out) < n:
                    out = out + [default] * (n - len(out))
                else:
                    out = out[:n]
            return out

        def _keep_metric_value(value):
            if isinstance(value, np.ndarray):
                return value.size == 1 and np.issubdtype(value.dtype, np.number)
            if isinstance(value, np.generic):
                return np.issubdtype(type(value), np.number) or isinstance(value, np.bool_)
            return isinstance(value, (int, float, bool))

        def _extend_metric_infos(batch_reward_extra_infos_dict, n):
            reward_extra_infos_dict["reward"].extend(batch_reward_extra_infos_dict["reward"])
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            for key, values in batch_reward_extra_infos_dict.items():
                if key == "reward" or len(values) != n:
                    continue
                if key == "pred":
                    reward_extra_infos_dict[key].extend(values)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")
                    continue
                if values and _keep_metric_value(values[0]):
                    reward_extra_infos_dict[key].extend(values)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

        global_sample_index = 0
        for batch_idx, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            batch_uids = _as_list(test_batch.non_tensor_batch.get("uid", None), len(input_texts), None)
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]

            sample_uids.extend(batch_uids)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            batch_n = len(output_texts)

            ntb = test_output_gen_batch.non_tensor_batch
            dump_keys = [
                "__num_turns__",
                "turn_scores",
                "tools_used",
                "tool_count",
                "used_any_tool",
                "tool_crop_boxes",
                "required_transforms",
                "gt_bbox",
                "tool_exec_success",
                "tool_exec_success_count",
                "tool_exec_error_count",
                "code_count",
                "tool_reward_detail",
                "step_answerability_version",
                "step_answerability_v0",
                "step_answerability_scores",
                "step_answerability_valid",
                "step_answerability_records",
            ]
            batch_dump_extra_infos_dict: dict[str, list] = defaultdict(list)
            for key in dump_keys:
                if key in ntb:
                    batch_dump_extra_infos_dict[key] = _as_list(ntb.get(key, None), batch_n, None)

            imgs = _as_list(ntb.get("merged_images", None), batch_n, None)
            details = _as_list(ntb.get("tool_reward_detail", None), batch_n, None)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            for key in ["data_source", "uid", "extra_info"]:
                if key in test_batch.non_tensor_batch:
                    batch_dump_extra_infos_dict[key] = _as_list(test_batch.non_tensor_batch.get(key, None), batch_n, None)

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            batch_reward_extra_infos_dict: dict[str, list] = {"reward": scores}
            if "reward_extra_info" in result:
                for key, values in result["reward_extra_info"].items():
                    batch_reward_extra_infos_dict[key] = _as_list(values, batch_n, None)
            _extend_metric_infos(batch_reward_extra_infos_dict, batch_n)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            if stream_validation_dump:
                if val_data_dir:
                    self._append_generations(
                        inputs=input_texts,
                        outputs=output_texts,
                        gts=ground_truths,
                        scores=scores,
                        reward_extra_infos_dict=batch_reward_extra_infos_dict,
                        dump_path=val_data_dir,
                        extra_dump_infos_dict=batch_dump_extra_infos_dict,
                        start_index=global_sample_index,
                        batch_index=batch_idx,
                    )
                if diagnostics_dir:
                    self._append_validation_diagnostics(
                        inputs=input_texts,
                        outputs=output_texts,
                        gts=ground_truths,
                        scores=scores,
                        reward_extra_infos_dict=batch_reward_extra_infos_dict,
                        dump_path=diagnostics_dir,
                        state=diag_state,
                        extra_dump_infos_dict=batch_dump_extra_infos_dict,
                        raw_generation_path=raw_generation_path,
                        start_index=global_sample_index,
                    )
                    self._write_validation_diagnostics_manifest(
                        dump_path=diagnostics_dir,
                        state=diag_state,
                        raw_generation_path=raw_generation_path,
                        completed=False,
                    )

                if len(log_inputs) < max_log_samples:
                    take = min(max_log_samples - len(log_inputs), batch_n)
                    log_inputs.extend(input_texts[:take])
                    log_outputs.extend(output_texts[:take])
                    log_scores.extend(scores[:take])
                    log_images.extend(imgs[:take])
                    log_details.extend(details[:take])
                    log_gts.extend(ground_truths[:take])
            else:
                sample_inputs.extend(input_texts)
                sample_outputs.extend(output_texts)
                sample_gts.extend(ground_truths)
                val_images_agg.extend(imgs)
                val_details_agg.extend(details)
                for key, values in batch_dump_extra_infos_dict.items():
                    val_dump_extra_infos_dict[key].extend(values)

            global_sample_index += batch_n

            if stream_validation_dump:
                del (
                    test_gen_batch,
                    test_gen_batch_padded,
                    test_output_gen_batch_padded,
                    test_output_gen_batch,
                    test_batch,
                    result,
                    reward_tensor,
                    input_ids,
                    output_ids,
                )
                gc.collect()

        if stream_validation_dump:
            self._maybe_log_val_generations(
                inputs=log_inputs,
                outputs=log_outputs,
                scores=log_scores,
                images=log_images,
                detail_arr=log_details,
                ground_truths=log_gts,
            )
            if diagnostics_dir:
                self._write_validation_diagnostics_manifest(
                    dump_path=diagnostics_dir,
                    state=diag_state,
                    raw_generation_path=raw_generation_path,
                    completed=True,
                )
        else:
            # Directly pass images and details to logger without caching
            self._maybe_log_val_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                images=val_images_agg,
                detail_arr=val_details_agg,
                ground_truths=sample_gts,
            )

            if val_data_dir:
                self._dump_generations(
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    gts=sample_gts,
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=val_data_dir,
                    extra_dump_infos_dict=val_dump_extra_infos_dict,
                )

            if diagnostics_dir:
                self._dump_validation_diagnostics(
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    gts=sample_gts,
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=diagnostics_dir,
                    extra_dump_infos_dict=val_dump_extra_infos_dict,
                    raw_generation_path=raw_generation_path,
                )

        for key_info in list(reward_extra_infos_dict.keys()):
            lst = reward_extra_infos_dict[key_info]
            if len(lst) not in (0, len(sample_scores)):
                print(
                    f"drop inconsistent validation metric field '{key_info}': "
                    f"len={len(lst)}, expected={len(sample_scores)}"
                )
                del reward_extra_infos_dict[key_info]

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    metric_dict[f"val-{data_source}/{var_name}/{metric_name}"] = metric_val
                    if metric_name == f"mean@{n_max}":
                        metric_dict[f"val-{data_source}/{var_name}"] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # Only-test mode: run validation once and exit immediately
        try:
            only_test_flag = False
            if hasattr(self.config, "trainer") and self.config.trainer is not None:
                only_test_flag = bool(self.config.trainer.get("only_test", False)) or bool(
                    self.config.trainer.get("val_only", False)
                )
            # Also allow top-level override for convenience
            only_test_flag = bool(self.config.get("only_test", only_test_flag))
        except Exception:
            only_test_flag = False

        if only_test_flag:
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for only_test mode.")
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            # Save metrics to file for convenient inspection
            try:
                output_path = None
                if hasattr(self.config, "trainer") and self.config.trainer is not None:
                    output_path = self.config.trainer.get("val_metrics_output", None)
                if output_path is None:
                    base_dir = None
                    if hasattr(self.config, "trainer") and self.config.trainer is not None:
                        base_dir = self.config.trainer.get("default_local_dir", None)
                    if base_dir is None:
                        base_dir = "."
                    if not os.path.isabs(base_dir):
                        base_dir = os.path.join(os.getcwd(), base_dir)
                    os.makedirs(base_dir, exist_ok=True)
                    output_path = os.path.join(base_dir, "only_test_val_metrics.json")
                else:
                    dir_name = os.path.dirname(output_path) or "."
                    if not os.path.isabs(dir_name):
                        dir_name = os.path.join(os.getcwd(), dir_name)
                    os.makedirs(dir_name, exist_ok=True)
                
                # Convert numpy types to Python native types for JSON serialization
                def convert_to_native(obj):
                    """Recursively convert numpy types to Python native types."""
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_to_native(value) for key, value in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_native(item) for item in obj]
                    else:
                        return obj
                
                val_metrics_converted = convert_to_native(val_metrics)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"global_steps": self.global_steps, "metrics": val_metrics_converted}, f, ensure_ascii=False, indent=2)
                print(f"Saved validation metrics to {output_path}")
            except Exception as e:
                print(f"Warning: Failed to save validation metrics: {e}")
            return

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            for key in (
                                "R_acc",
                                "R_fmt",
                                "R_protocol",
                                "R_mut",
                                "MutWeight",
                                "P_regular_tool",
                                "P_turn_overuse",
                                "R_base_total",
                                "R_step_raw",
                                "R_step",
                                "StepScoredCount",
                                "StepValidCount",
                                "StepBestScore",
                                "R_total",
                                "NumTurns",
                                "step_answerability_version",
                                "step_answerability_v0",
                                "step_answerability_scores",
                                "step_answerability_valid",
                                "step_answerability_records",
                            ):
                                if key in batch.non_tensor_batch:
                                    values = batch.non_tensor_batch[key]
                                    if hasattr(values, "tolist"):
                                        values = values.tolist()
                                    reward_extra_infos_dict.setdefault(key, values)

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                
                self._maybe_log_train_generations(batch)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

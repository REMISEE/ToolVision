# Malformed Tool-Call Recovery Plan

## Current Behavior

The Hermes parser in `verl/experimental/agent_loop/tool_parser.py` decodes assistant output and extracts `<tool_call>...</tool_call>` blocks.

Current failure path:

1. Model emits a `<tool_call>` block with malformed JSON.
2. `json.loads()` fails.
3. Parser logs `Failed to decode tool call: ...`.
4. Parser returns an empty `function_calls` list.
5. `ToolAgentLoop._handle_generating_state()` sees no valid tool calls and terminates the trajectory.
6. The model never receives feedback that its tool-call JSON was invalid.

The trajectory is still used for RL. Reward-side code detects the malformed block later through `tool_usage_info()` and marks `invalid_tool_call=1`, but by then the trajectory has already died.

## Why This Matters

Without recovery, one JSON typo can convert an otherwise useful tool trajectory into:

- no tool execution,
- no second observation turn,
- usually no final `<answer>`,
- low/zero format reward,
- low/zero accuracy,
- a noisy negative RL sample.

This is especially risky for MUT-style data, where tool-call density will be higher and external tool observations are central to solving the sample.

## Recommended Design

Do not auto-repair JSON and do not use guided decoding inside RL rollout yet.

Instead, treat malformed tool calls as a tool observation:

```text
Error: your previous <tool_call> block was not valid JSON:
<short parser error>

Please continue inside <think>...</think>. Re-emit exactly one valid
<tool_call>{"name":"code_image_tool","arguments":{...}}</tool_call>,
or provide the final answer inside <answer>...</answer>.
```

Then continue the agent loop as if a tool response had been appended.

Benefits:

- Keeps sampling distribution unconstrained.
- Gives the model a chance to self-correct.
- Preserves evidence that the first call was malformed.
- Avoids terminating a trajectory purely because of one missing brace or escape.

## Implementation Shape

1. Extend parser result without using shared mutable parser state.

   Best option: add a new parser method while keeping the old API:

   ```python
   @dataclass
   class ToolParseError:
       message: str
       snippet: str

   async def extract_tool_calls_with_errors(
       self, responses_ids: list[int]
   ) -> tuple[str, list[FunctionCall], list[ToolParseError], bool]:
       ...
   ```

   The final bool is `had_tool_call_tag`.

   Reason: `ToolParser` is class-level/shared in the agent loop. Storing `last_parse_error` on the parser object would be unsafe under concurrent rollout.

2. In `ToolAgentLoop._handle_generating_state()`:

   - Use `extract_tool_calls_with_errors()` when available.
   - If valid calls exist, proceed to `PROCESSING_TOOLS`.
   - If `had_tool_call_tag=True` and parse errors exist, append an error observation and return `GENERATING`.
   - If no tool-call tag exists, keep the normal termination path.

3. Add a helper in `ToolAgentLoop`:

   ```python
   async def _append_text_tool_observation(self, agent_data, text: str) -> AgentState:
       message = {"role": "tool", "content": text}
       ...
       agent_data.prompt_ids += response_ids
       agent_data.response_mask += [0] * len(response_ids)
       agent_data.user_turns += 1
       return AgentState.GENERATING
   ```

   This should share the same chat-template/tokenization logic as `_handle_processing_tools_state()` but without image handling.

4. Track counters:

   - `agent_data.malformed_tool_call_count`
   - output extra field `malformed_tool_call_count`
   - optionally `tool_parse_recovery_count`

   Keep reward-side `invalid_tool_call` unchanged. A recovered trajectory should still carry a small invalid-call penalty if the malformed block remains in the raw output.

5. Guard against loops:

   - Existing `max_assistant_turns=12` already caps retries.
   - Add a local per-trajectory malformed retry cap, recommended `2`.
   - After the cap, terminate to avoid endless parse-error loops.

## Tests

Unit tests for the parser:

- Valid tool call returns one `FunctionCall`, zero errors.
- Missing closing brace returns zero calls and one parse error.
- Missing `</tool_call>` sets `had_tool_call_tag=True` and parse error.
- Plain final answer without tool call returns zero calls, zero errors, `had_tool_call_tag=False`.

Agent-loop smoke:

- Feed a malformed tool call and verify an error observation is appended.
- Verify a normal `<answer>...</answer>` still terminates.
- Verify retry cap terminates after repeated malformed calls.

## Deployment Recommendation

Do not mix this into the current max-num-seqs boundary probes. First establish the safe `MAX_NUM_SEQS` value with the current code.

After choosing a stable serving setting, merge this recovery patch before MUT runs. MUT has higher tool density, so the recovery path is more valuable there.


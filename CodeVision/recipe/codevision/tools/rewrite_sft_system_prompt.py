#!/usr/bin/env python3
"""Rewrite ShareGPT SFT system prompts with a CodeVision system/tool schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


TOOL_WRAPPER = """# Tool
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def build_system_prompt(system_prompt_path: Path, tool_config_path: Path) -> str:
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    config = yaml.safe_load(tool_config_path.read_text(encoding="utf-8"))
    tool_schema = config["tools"][0]["tool_schema"]
    code_schema = tool_schema["function"]["parameters"]["properties"].get("code")
    if isinstance(code_schema, dict) and isinstance(code_schema.get("description"), str):
        code_schema["description"] = code_schema["description"].replace("\n", "\\n")
    schema_text = json.dumps(tool_schema, ensure_ascii=False)
    return f"{system_prompt}\n{TOOL_WRAPPER.format(tool_schema=schema_text)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input ShareGPT JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Output ShareGPT JSON file.")
    parser.add_argument("--system-prompt", type=Path, required=True, help="Replacement system prompt text file.")
    parser.add_argument("--tool-config", type=Path, required=True, help="Replacement CodeImageTool YAML config.")
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    new_system = build_system_prompt(args.system_prompt, args.tool_config)
    updated = 0
    for record in records:
        if isinstance(record, dict):
            record["system"] = new_system
            updated += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {updated} records to {args.output}")


if __name__ == "__main__":
    main()

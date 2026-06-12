#!/usr/bin/env python3
"""Check an OpenAI-compatible chat/completions API endpoint."""

from __future__ import annotations

import os
import sys

from openai import APIError, DefaultHttpxClient, OpenAI


def getenv(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return ""


def getenv_with_name(*names: str) -> tuple[str, str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return name, value.strip()
    return "", ""


def validate_ascii_env(name: str, value: str, *, secret: bool = False) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        bad = value[exc.start : exc.end]
        print(
            f"{name} contains a non-ASCII/invalid character at position {exc.start}: "
            f"codepoint={ord(bad[0]):#x}. Re-export this variable from a clean ASCII key/value.",
            file=sys.stderr,
        )
        if not secret:
            print(f"{name}={value!r}", file=sys.stderr)
        return False
    return True


def main() -> int:
    # Prefer the names used in our submit scripts, but accept LLM_JUDGE_* as aliases.
    base_url_name, base_url = getenv_with_name("OFFLINE_SFT_QWEN_BASE_URL", "LLM_JUDGE_BASE_URL")
    model_name, model = getenv_with_name("OFFLINE_SFT_QWEN_MODEL", "LLM_JUDGE_MODEL_NAME")
    api_key_name, api_key = getenv_with_name(
        "OFFLINE_SFT_QWEN_API_KEY", "DASHSCOPE_API_KEY", "LLM_JUDGE_API_KEY", "OPENAI_API_KEY"
    )
    timeout = float(os.getenv("API_CHECK_TIMEOUT", "30"))
    trust_env = os.getenv("API_CHECK_TRUST_ENV", "0").strip().lower() in {"1", "true", "yes", "on"}

    missing = [
        name
        for name, value in {
            "OFFLINE_SFT_QWEN_BASE_URL/LLM_JUDGE_BASE_URL": base_url,
            "OFFLINE_SFT_QWEN_MODEL/LLM_JUDGE_MODEL_NAME": model,
            "OFFLINE_SFT_QWEN_API_KEY/DASHSCOPE_API_KEY/LLM_JUDGE_API_KEY/OPENAI_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        print("Missing env:", ", ".join(missing), file=sys.stderr)
        return 2
    if not validate_ascii_env(base_url_name, base_url):
        return 2
    if not validate_ascii_env(model_name, model):
        return 2
    if not validate_ascii_env(api_key_name, api_key, secret=True):
        return 2

    # DSW often sets a SOCKS proxy; trust_env=False avoids requiring httpx[socks].
    # Set API_CHECK_TRUST_ENV=1 only when the request must go through that proxy.
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=DefaultHttpxClient(trust_env=trust_env, timeout=timeout),
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with exactly: ok"},
                {"role": "user", "content": "ping"},
            ],
            temperature=0,
            max_tokens=8,
            extra_body={"enable_thinking": False},
        )
    except (APIError, UnicodeError) as exc:
        print(f"API check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    text = response.choices[0].message.content
    print(f"base_url={base_url}")
    print(f"model={model}")
    print(f"response={text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

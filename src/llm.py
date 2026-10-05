"""LLM provider configuration (any litellm provider; Groq free tier by default)."""

import os
from typing import Any

from crewai import LLM

DEFAULT_MODEL = "groq/openai/gpt-oss-120b"


class SafeLLM(LLM):
    """LLM that strips crewai's `cache_breakpoint` flag, which several providers reject."""

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(messages, list):
            messages = [
                {k: v for k, v in m.items() if k != "cache_breakpoint"}
                if isinstance(m, dict)
                else m
                for m in messages
            ]
        return super().call(messages, *args, **kwargs)


def model_name() -> str:
    """Model id in litellm format, e.g. 'groq/openai/gpt-oss-120b'."""
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def api_key_var() -> str:
    """Name of the env var holding the key for the configured provider."""
    return f"{model_name().split('/')[0].upper()}_API_KEY"


def llm_configured() -> bool:
    """True if the API key for the configured provider is set."""
    return bool(os.getenv(api_key_var()))


def get_llm() -> LLM:
    """Build the LLM used by all agents (deterministic, with retries for rate limits)."""
    return SafeLLM(model=model_name(), temperature=0, num_retries=6)

"""Shared, dependency-free config: which env vars hold provider API keys.
Its own module so both cli.py and web.py can use it without a circular import
(cli.py needs to import web.py to wire the `prompteval web` subcommand).
"""

import os

API_KEY_VARS = ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY")


def has_any_api_key() -> bool:
    return any(os.environ.get(key) for key in API_KEY_VARS)

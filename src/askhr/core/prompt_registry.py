"""
Core: loads versioned prompt templates from config/prompts.yaml.

Prompts are effectively part of this system's behavior — a wording
change can shift what the LLM answers just as much as a code change can.
Storing them in a separate, versioned YAML file (rather than as inline
Python strings) means a prompt change shows up as its own diff, each
prompt carries its own version number independent of app version, and
prompt changes can be reviewed without touching src/.

This module only loads and validates the raw config; it doesn't build
any LangChain objects — search/prompts.py does that, using get_prompt()
from here as its source of truth.
"""
from pathlib import Path
from typing import Any, Dict

import yaml

from askhr.core.config import PROJECT_ROOT

PROMPTS_FILE = PROJECT_ROOT / "config" / "prompts.yaml"

_prompts_cache: Dict[str, Any] = None


def _load_all_prompts() -> Dict[str, Any]:
    global _prompts_cache
    if _prompts_cache is None:
        if not PROMPTS_FILE.exists():
            raise FileNotFoundError(f"Prompt config file not found: {PROMPTS_FILE}")
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            _prompts_cache = yaml.safe_load(f)
    return _prompts_cache


def get_prompt(name: str) -> Dict[str, Any]:
    """
    Returns the raw config dict for one named prompt — version,
    description, template/persona text, input_variables — as defined in
    config/prompts.yaml.

    Raises KeyError with the list of available prompt names if `name`
    isn't found, rather than a bare KeyError with no context.
    """
    prompts = _load_all_prompts()
    if name not in prompts:
        raise KeyError(
            f"No prompt named {name!r} in {PROMPTS_FILE}. "
            f"Available: {sorted(prompts.keys())}"
        )
    return prompts[name]


if __name__ == "__main__":
    # Manual check: confirms the YAML file parses and each expected
    # prompt entry is present, with no model or network call needed.
    for name in ("qa_prompt", "example_prompt", "verification_prompt"):
        entry = get_prompt(name)
        print(f"{name} (v{entry.get('version')}): {entry.get('description', '').strip()}")
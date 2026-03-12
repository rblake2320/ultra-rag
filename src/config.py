"""
Config loader: reads config.yaml, resolves ${ENV_VAR} references,
falls back to .env file at D:\\rag-ingest\\.env.
"""
import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"
ENV_PATH     = PROJECT_ROOT / ".env"

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _load_dotenv(path: Path) -> None:
    """Load .env file into os.environ (does not override existing vars)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


def _resolve(obj: Any) -> Any:
    """Recursively resolve ${VAR} references in strings."""
    if isinstance(obj, str):
        def replace(m):
            var = m.group(1)
            # Check if var exists in environment at all (allows empty values)
            if var not in os.environ:
                raise ValueError(f"Environment variable '{var}' is not set. "
                                 f"Add it to {ENV_PATH} or set it in the shell.")
            return os.environ[var]

        # Only resolve if the string contains references
        if _ENV_VAR_RE.search(obj):
            return _ENV_VAR_RE.sub(replace, obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve(v) for v in obj]
    return obj


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and return the fully-resolved config dict."""
    _load_dotenv(ENV_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _resolve(raw)


# Singleton — loaded once per process
_cfg: dict | None = None


def get_config() -> dict:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg

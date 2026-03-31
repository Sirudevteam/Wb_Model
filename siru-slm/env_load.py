"""Load `.env` files so parent workspace keys are not shadowed by a bad `siru-slm/.env`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_project_env() -> Optional[str]:
    """
    Load `.env` from the folder above `siru-slm/` (workspace) and from `siru-slm/`.

    Shallow path first, then deeper with ``override=False`` so **parent keys win** if
    both files set the same variable (fixes 401 when `siru-slm/.env` has a bad key).

    We do **not** auto-load ``cwd/.env`` — a shallow unrelated `.env` could win otherwise.

    Returns the path of the shallowest file that was loaded, or None.
    """
    here = Path(__file__).resolve().parent
    paths: list[Path] = []
    for p in (here.parent / ".env", here / ".env"):
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp.is_file() and rp not in paths:
            paths.append(rp)

    paths.sort(key=lambda p: len(p.parts))

    if not paths:
        load_dotenv()
        return None

    for i, env_path in enumerate(paths):
        load_dotenv(env_path, override=(i == 0))

    return str(paths[0])


def clean_env(value: Optional[str]) -> str:
    """Strip whitespace, BOM, and surrounding quotes from env values."""
    if value is None:
        return ""
    s = value.strip().lstrip("\ufeff")
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s

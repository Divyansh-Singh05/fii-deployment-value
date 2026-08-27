"""Persisted artifact digests.

Verification is a claim about a moment: *these numbers came from these files, in
this state*. The lock file is what carries that claim forward, so a later run
can distinguish "not yet checked" from "checked, and the source has since
changed". Without it every run would report a first sighting and STALE would be
unreachable.

The file is committed. A reviewer comparing it against their own copy of the
source trees learns immediately whether they are looking at the same artifacts.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Final

DEFAULT_PATH: Final = Path("outputs/evidence.lock.json")
_SCHEMA: Final = 1


def key(tree: str, artifact: str) -> str:
    """Lock key for one artifact. Several evidence items may share it."""
    return f"{tree}:{artifact}"


def load(path: Path | None = None) -> dict[str, str]:
    """Read the digest map. A missing file is an empty map, not an error."""
    p = path or DEFAULT_PATH
    if not p.is_file():
        return {}
    payload = json.loads(p.read_text())
    if payload.get("schema") != _SCHEMA:
        raise ValueError(f"{p}: unsupported lock schema {payload.get('schema')!r}")
    digests: dict[str, str] = payload["artifacts"]
    return digests


def save(digests: dict[str, str], path: Path | None = None) -> Path:
    """Write the digest map, sorted so diffs are readable."""
    p = path or DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "schema": _SCHEMA,
                "recorded_on": date.today().isoformat(),
                "artifacts": dict(sorted(digests.items())),
            },
            indent=2,
        )
        + "\n"
    )
    return p

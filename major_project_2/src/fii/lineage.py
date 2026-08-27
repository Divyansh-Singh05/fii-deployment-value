"""
ARTIFACT LINEAGE  —  audit item 12

Every parquet/NPZ the pipeline writes gets a sidecar recording exactly what
produced it, so a stored result can never be ambiguous about which source
version, which code and which configuration it corresponds to.

WHY.  The audit found this the hard way. C1's own header carries a provenance
note admitting the original harness "ran from a scratch directory that was
cleared between sessions; only its output survived", so the vintages could not
be reproduced and had to be re-estimated — which moved two of four archetype
thresholds by more than two bootstrap standard errors (PHASE3_PREREG
Amendment 3b). A sidecar would have made that a lookup instead of a
reconstruction.

WHAT IS RECORDED
    run           timestamp, host, user, wall-clock duration
    code          git commit, dirty flag, branch, the module that wrote it
    command       the full argv that produced the artifact
    inputs        path, size, mtime and a content hash for every source read
    config        the resolved config block in force
    extras        caller-supplied facts (vintage schedule, mapping version...)

Hashing is chunked, and files above HASH_CAP_MB record size+mtime instead of a
digest so a 500 MB parquet does not add minutes to a 30-second stage.

    from fii.lineage import write_with_lineage
    write_with_lineage(df, OUT, inputs=[SRC1, SRC2],
                       extras={"n_vintages": 106})
"""
from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HASH_CAP_MB = 256
_T0 = dt.datetime.now()


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10, cwd=Path(__file__).resolve().parent
                              ).stdout.strip() or None
    except Exception:
        return None


def file_hash(p: Path) -> dict:
    p = Path(p)
    if not p.exists():
        return {"path": str(p), "exists": False}
    size = p.stat().st_size
    rec = {"path": str(p), "bytes": size,
           "mtime": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(
               timespec="seconds")}
    if size > HASH_CAP_MB * 1024 * 1024:
        rec["sha256"] = None
        rec["hash_skipped"] = f"larger than {HASH_CAP_MB} MB"
        return rec
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    rec["sha256"] = h.hexdigest()
    return rec


def lineage(inputs=None, extras=None, config=None) -> dict:
    now = dt.datetime.now()
    return {
        "artifact_lineage_version": 1,
        "run": {
            "written_at": now.isoformat(timespec="seconds"),
            "elapsed_s": round((now - _T0).total_seconds(), 2),
            "host": platform.node(),
            "user": getpass.getuser(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "code": {
            "module": getattr(sys.modules.get("__main__"), "__file__", None),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "git_describe": _git("describe", "--always", "--dirty"),
        },
        "command": {"argv": sys.argv,
                    "cwd": os.getcwd(),
                    "reconstructed": " ".join([sys.executable, *sys.argv])},
        "inputs": [file_hash(p) for p in (inputs or [])],
        "config": config,
        "extras": extras or {},
    }


def sidecar_path(out: Path) -> Path:
    out = Path(out)
    return out.with_suffix(out.suffix + ".lineage.json")


def write_lineage(out: Path, inputs=None, extras=None, config=None) -> Path:
    """Write the sidecar for an artifact that has already been saved."""
    rec = lineage(inputs, extras, config)
    rec["output"] = file_hash(out)
    sp = sidecar_path(out)
    sp.write_text(json.dumps(rec, indent=2, default=str))
    c = rec["code"]
    print(f"[lineage] {sp.name}  commit "
          f"{(c['git_commit'] or 'n/a')[:10]}"
          f"{' (dirty)' if c['git_dirty'] else ''}  "
          f"{len(rec['inputs'])} input(s)")
    return sp


def write_with_lineage(df, out: Path, inputs=None, extras=None, config=None):
    """Write a polars frame and its sidecar together."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    write_lineage(out, inputs, extras, config)
    return out

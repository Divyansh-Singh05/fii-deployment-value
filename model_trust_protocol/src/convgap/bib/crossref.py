"""Resolve references against Crossref, refusing matches that do not match.

The journal requires a DOI for every reference. A DOI transcribed from memory
or inferred from a search snippet is worse than no DOI, because it resolves to
something and the reader assumes it is right. This module queries Crossref by
title and accepts a record only when the returned title is close enough to the
one asked for; anything else is reported UNRESOLVED for a human to settle.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Final

_API: Final = "https://api.crossref.org/works"
_MAILTO: Final = "divyanshsinghpost@gmail.com"
_ACCEPT: Final = 0.82
"""Title similarity below which a candidate is refused rather than guessed."""

_PREPRINT_HOSTS: Final = ("ssrn", "nber", "arxiv", "repec", "researchgate")
"""Container names that indicate a working paper rather than a journal article.

Crossref indexes the preprint and the published article separately, and the
preprint often scores an identical title match. A reference list for a
peer-reviewed submission must cite the published version, so preprints are
rejected here rather than ranked lower.
"""


def _is_preprint(item: dict[str, Any]) -> bool:
    container = str((item.get("container-title") or [""])[0]).lower()
    publisher = str(item.get("publisher", "")).lower()
    doi = str(item.get("DOI", "")).lower()
    if item.get("type") != "journal-article":
        return True
    if not container:
        return True
    return any(host in container or host in publisher or host in doi for host in _PREPRINT_HOSTS)


def _name(raw: str) -> str:
    """Title-case a surname Crossref returns shouted.

    Several publishers deposit author names in upper case. Reproducing that in a
    reference list is wrong, and lower-casing wholesale breaks particles and
    initials, so each token is handled on its own.
    """
    out = []
    for token in raw.split():
        if token.isupper() and len(token) > 1 and not token.endswith("."):
            out.append(token.capitalize())
        else:
            out.append(token)
    return " ".join(out)


def _apa_name(family: str, given: str) -> str:
    """APA 7 author form: surname, then initials with periods and spaces."""
    surname = _name(family).strip()
    initials = " ".join(
        f"{part[0].upper()}."
        for part in re.split(r"[\s\-]+", given.strip())
        if part and part[0].isalpha()
    )
    return f"{surname}, {initials}".strip(", ") if surname else initials


def _issued_year(item: dict[str, Any]) -> int | None:
    """Year of the printed issue, not of the first online posting.

    Crossref's ``issued`` field carries the earliest recorded date, which for a
    journal that posts online-first is a year or two before the issue a reader
    would cite.
    """
    for key in ("published-print", "journal-issue", "published", "issued"):
        block = item.get(key) or {}
        if key == "journal-issue":
            block = block.get("published-print") or block.get("published-online") or {}
        parts = block.get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class Record:
    """A Crossref record, or the reason none was accepted."""

    query: str
    resolved: bool
    similarity: float = 0.0
    doi: str = ""
    title: str = ""
    journal: str = ""
    year: int | None = None
    volume: str = ""
    issue: str = ""
    pages: str = ""
    authors: tuple[str, ...] = ()
    note: str = ""

    @property
    def doi_url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""


def _fetch(params: str) -> dict[str, object]:
    out = subprocess.run(
        ["curl", "-s", "--max-time", "25", f"{_API}?{params}&mailto={_MAILTO}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"crossref request failed: rc={out.returncode}")
    payload: dict[str, object] = json.loads(out.stdout)
    return payload


def resolve(title: str, *, rows: int = 10) -> Record:
    """Look up one title. Accept only a close match."""
    from urllib.parse import quote_plus

    try:
        message = _fetch(
            f"query.title={quote_plus(title)}&rows={rows}&filter=type:journal-article"
        )["message"]
    except (RuntimeError, json.JSONDecodeError) as exc:
        return Record(title, False, note=f"lookup failed: {exc}")

    items = message.get("items", []) if isinstance(message, dict) else []
    want = _norm(title)
    best: Record | None = None
    for item in items:
        if _is_preprint(item):
            continue
        got = item.get("title") or [""]
        sim = SequenceMatcher(None, want, _norm(got[0])).ratio()
        if best is not None and sim <= best.similarity:
            continue

        best = Record(
            query=title,
            resolved=sim >= _ACCEPT,
            similarity=sim,
            doi=item.get("DOI", ""),
            title=got[0],
            journal=(item.get("container-title") or [""])[0],
            year=_issued_year(item),
            volume=item.get("volume", "") or "",
            issue=item.get("issue", "") or "",
            pages=item.get("page", "") or "",
            authors=tuple(
                _apa_name(a.get("family", ""), a.get("given", "")) for a in item.get("author", [])
            ),
        )
    if best is None:
        return Record(title, False, note="no published journal article among the candidates")
    if not best.resolved:
        return Record(
            query=title,
            resolved=False,
            similarity=best.similarity,
            title=best.title,
            doi=best.doi,
            journal=best.journal,
            year=best.year,
            note=f"closest match scored {best.similarity:.2f}, below {_ACCEPT}",
        )
    return best


def resolve_all(titles: list[str], *, pause: float = 0.6) -> list[Record]:
    out: list[Record] = []
    for t in titles:
        out.append(resolve(t))
        time.sleep(pause)
    return out


def apa7(rec: Record) -> str:
    """Format one resolved record in APA 7 with an https DOI."""
    if not rec.resolved:
        return f"[UNRESOLVED] {rec.query} — {rec.note}"
    names = rec.authors
    if not names:
        authors = ""
    elif len(names) == 1:
        authors = names[0]
    elif len(names) == 2:
        authors = f"{names[0]}, & {names[1]}"
    elif len(names) <= 20:
        authors = ", ".join(names[:-1]) + f", & {names[-1]}"
    else:
        authors = ", ".join(names[:19]) + ", . . . " + names[-1]
    year = f"({rec.year})." if rec.year else "(n.d.)."
    title = rec.title.replace("\u2010", "-").replace("\u2011", "-")
    vol = f", {rec.volume}" if rec.volume else ""
    iss = f"({rec.issue})" if rec.issue else ""
    pages = f", {rec.pages}" if rec.pages else ""
    journal = f" {rec.journal}{vol}{iss}{pages}." if rec.journal else ""
    doi = f" {rec.doi_url}" if rec.doi else ""
    stop = "" if title.rstrip().endswith(("?", "!")) else "."
    return f"{authors} {year} {title}{stop}{journal}{doi}".strip()

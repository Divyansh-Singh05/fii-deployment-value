"""Table 1 - the friction ladder.

One row per application of the dataset, ordered by the level of real-world
friction at which its value is lost. Read down the table, the ladder maps the
boundary of the dataset's usefulness: what the record is worth is exactly the
set of uses that survive to the bottom.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from convgap.application import ApplicationResult
from convgap.friction import CLEARS_ALL, FrictionLevel
from convgap.provenance import SourceTier, VerificationStatus
from convgap.role import Role

_COLUMNS = [
    "key",
    "title",
    "role",
    "instrument",
    "clears",
    "dies_at",
    "significance",
    "value",
    "conversion_ratio",
    "status",
    "tier",
    "mechanism",
]


@dataclass(frozen=True, slots=True)
class Table1:
    """Rendered exhibit plus the audit summary that must accompany it."""

    rows: tuple[ApplicationResult, ...]

    # -- audit ---------------------------------------------------------------

    @property
    def n_publishable(self) -> int:
        return sum(1 for r in self.rows if r.publishable)

    @property
    def blocking(self) -> tuple[ApplicationResult, ...]:
        return tuple(r for r in self.rows if not r.publishable)

    # -- the finding ---------------------------------------------------------

    @property
    def applications(self) -> tuple[ApplicationResult, ...]:
        """Rows that apply the dataset under study, excluding controls."""
        return tuple(r for r in self.rows if r.role is Role.APPLICATION)

    @property
    def survivors(self) -> tuple[ApplicationResult, ...]:
        return tuple(r for r in self.applications if r.survives)

    @property
    def attrition(self) -> dict[FrictionLevel, int]:
        """How many applications die at each level, ascending."""
        counts = dict.fromkeys(FrictionLevel, 0)
        for row in self.applications:
            if row.fails_at is not None:
                counts[row.fails_at] += 1
        return counts

    def _record(self, row: ApplicationResult) -> dict[str, str]:
        ratio = f"{row.detectability.conversion_ratio:.0%}" if row.detectability is not None else ""
        return {
            "key": row.key,
            "title": row.title,
            "role": str(row.role),
            "instrument": str(row.layer),
            "clears": f"{row.clears}/{int(CLEARS_ALL)}",
            "dies_at": "-" if row.fails_at is None else row.fails_at.label,
            "significance": str(row.significance),
            "value": str(row.value),
            "conversion_ratio": ratio,
            "status": str(row.status),
            "tier": str(row.tier),
            "mechanism": row.mechanism,
        }

    # -- rendering -----------------------------------------------------------

    def to_csv(self, path: Path | None = None) -> str:
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in self.rows:
            writer.writerow(self._record(row))
        text = buf.getvalue()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text

    @staticmethod
    def _escape(text: str) -> str:
        """Pipes inside a cell would terminate the column early."""
        return text.replace("|", "\\|")

    def to_markdown(self) -> str:
        lines = [
            "| # | Application | Instrument | Clears | Dies at | Significance | Value | Status |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in self.rows:
            rec = {k: self._escape(v) for k, v in self._record(row).items()}
            marker = " *(control)*" if row.role is Role.CONTROL else ""
            lines.append(
                "| {key} | {title}".format(**rec)
                + marker
                + " | "
                "{instrument} | {clears} | {dies_at} | {significance} | "
                "{value} | {status} |".format(**rec)
            )
        return "\n".join(lines)

    def ladder(self) -> str:
        """The attrition summary: how far the dataset carries."""
        lines = ["friction level              applications lost"]
        running = len(self.applications)
        for level in FrictionLevel:
            lost = self.attrition[level]
            running -= lost
            bar = "#" * lost if lost else "."
            lines.append(
                f"  {int(level)} {level.label:<22} {bar:<6} {lost} lost, {running} still standing"
            )
        n_survive = len(self.survivors)
        lines.append(
            f"\n  {n_survive} of {len(self.applications)} applications of the dataset "
            f"survive every level."
        )
        return "\n".join(lines)

    def audit_note(self) -> str:
        """The sentence that must accompany the table in any draft."""
        total = len(self.rows)
        counts = {s: sum(1 for r in self.rows if r.status is s) for s in VerificationStatus}
        secondary = sum(1 for r in self.rows if r.tier is SourceTier.SECONDARY)
        parts = [f"{self.n_publishable}/{total} rows publishable"]
        for status, label in (
            (VerificationStatus.UNVERIFIED, "untraced"),
            (VerificationStatus.STALE, "STALE (source changed since verification)"),
            (VerificationStatus.MISMATCH, "MISMATCH (artifact lacks the declared value)"),
            (VerificationStatus.MISSING, "MISSING (source not found)"),
        ):
            if counts[status]:
                parts.append(f"{counts[status]} {label}")
        if secondary:
            parts.append(
                f"{secondary} still sourced from a project document rather than "
                f"the artifact that produced the number"
            )
        return "; ".join(parts) + "."


def build(results: list[ApplicationResult]) -> Table1:
    """Order rows by the friction level at which they die, then by key.

    Survivors sort last: the exhibit is read as a descent, and what remains at
    the bottom is the answer to what the dataset is worth.
    """

    def sort_key(row: ApplicationResult) -> tuple[int, str]:
        return (row.clears, row.key)

    return Table1(rows=tuple(sorted(results, key=sort_key)))

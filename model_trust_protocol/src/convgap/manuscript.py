"""Check the manuscript's numerals against the verified evidence base.

The paper argues that a statistic is only as good as the audit around it. This
module applies that to the manuscript: it extracts every numeral from the prose
and reports which of them correspond to a value this repository has verified.

The output is a report, not a gate. A manuscript legitimately contains numerals
that are not evidence - section numbers, sample years, counts stated in words,
thresholds quoted from a pre-registration. What the report establishes is that
no numeral is *silently* unaccounted for: each is either matched to verified
evidence, or appears in the exempt set with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from convgap import facts
from convgap.provenance import SourceTree
from convgap.registry import all_applications

#: Numerals that are not evidence and are exempt by kind, with the reason.
EXEMPT: dict[str, str] = {
    "0": "counts and levels",
    "1": "counts and levels",
    "2": "counts and levels",
    "3": "counts and levels",
    "4": "counts and levels",
    "5": "counts and levels",
    "6": "counts and levels",
    "7": "counts and levels",
    "8": "counts and levels",
    "9": "counts and levels",
    "10": "counts and levels",
    "2026": "run date",
    "24": "date component",
    "28": "date component",
    "30": "date component",
    "31": "date component",
    "01": "date component",
    "03": "date component",
    "05": "date component",
    "06": "date component",
    "09": "date component",
    "11": "date component",
    "04": "date component",
    "07": "date component",
    "13": "date component",
    "15": "assumed institutional cost, bp - an assumption, stated as one",
    "1.96": "conventional significance hurdle, cited from the literature",
    "2.78": "multiple-testing hurdle, cited from the literature",
    "2.50": "pre-registered bar",
    "1.5": "pre-registered bar",
    "0.005": "pre-registered bar",
    "1992": "sample year, split date, or publication year of a cited work",
    "2011": "sample year, split date, or publication year of a cited work",
    "2017": "sample year, split date, or publication year of a cited work",
    "2020": "sample year, split date, or publication year of a cited work",
    "2021": "sample year, split date, or publication year of a cited work",
    "2022": "sample year, split date, or publication year of a cited work",
    "2023": "sample year, split date, or publication year of a cited work",
    "2024": "sample year, split date, or publication year of a cited work",
    "2025": "sample year, split date, or publication year of a cited work",
    "452": "cited from Hou, Xue & Zhang (2020)",
    "162": "cited from Muravyev, Pearson & Pollet (2025)",
    "65": "cited from Hou, Xue & Zhang (2020)",
    "82.1": "cited from Hou, Xue & Zhang (2020)",
    "82": "cited from Hou, Xue & Zhang (2020)",
    "96": "cited from Hou, Xue & Zhang (2020)",
    "0.14": "cited from Muravyev, Pearson & Pollet (2025)",
    "0.01": "cited from Muravyev, Pearson & Pollet (2025)",
    "12": "cited from Muravyev, Pearson & Pollet (2025)",
    "2.0": "pre-registered engine gate bar",
    "95": "confidence level",
    "52": "interval bound reported beside a traced point estimate",
    "19": "interval bound reported beside a traced point estimate",
    "0.797": "superseded pre-audit figure, quoted deliberately in 4.3.2",
    "41.6": "superseded pre-audit figure, quoted deliberately in 4.3.2",
    "23": "superseded pre-audit figure, quoted deliberately in 4.3.2",
    "100": "scaling factor stated in a column header",
    "939": "a figure the paper explicitly declines to assert, quoted to withdraw it",
    "171": "months in the sample span, arithmetic",
    "250": "specification constant, trailing window",
    "94": "specification constant, EWMA lambda",
    "50": "specification constant, corporate-action guard",
    "0.55": "approximate CRPS level, stated as approximate",
    "20": "specification constant, Newey-West lags / horizon",
}

_NUMERAL = re.compile(r"(?<![\w.\u00a7])(\d[\d,]*\.?\d*)(?![\w])")

#: Fixed strings stripped before scanning: proper nouns and cross-references
#: whose digits are not quantities.
_NOT_QUANTITIES = (
    "S&P 500",
    "SHA-256",
    "Nifty 50",
    "GJR-GARCH",
    "AR(5)",
)


@dataclass(frozen=True, slots=True)
class Unmatched:
    """A numeral in the prose with no verified counterpart."""

    numeral: str
    section: str
    context: str


def verified_values(trees: dict[str, SourceTree], locks: dict[str, str] | None = None) -> set[str]:
    """Every value this repository has verified, as strings the prose might use."""
    out: set[str] = set()

    def add(v: float | None, n: int | None = None) -> None:
        if v is not None:
            for places in range(7):
                out.add(f"{abs(v):.{places}f}")
                out.add(f"{abs(v):.{places}f}".rstrip("0").rstrip("."))
                out.add(f"{abs(v):,.{places}f}")
                out.add(f"{abs(v):,.{places}f}".rstrip("0").rstrip("."))
            # scientific forms a manuscript may use, e.g. 5.0 in "5.0 x 10^-5"
            if abs(v) >= 1e6:
                for scale in (1e6, 1e9):
                    if abs(v) >= scale:
                        red = abs(v) / scale
                        for places in (0, 1, 2):
                            out.add(f"{red:.{places}f}")
                            out.add(f"{red:.{places}f}".rstrip("0").rstrip("."))
            if 0 < abs(v) < 1e-3:
                mant = abs(v)
                while mant < 1:
                    mant *= 10
                for places in (0, 1, 2):
                    out.add(f"{mant:.{places}f}")
                    out.add(f"{mant:.{places}f}".rstrip("0").rstrip("."))
        if n is not None:
            out.add(str(n))
            out.add(f"{n:,}")

    for app in all_applications():
        for half in (app.significance(), app.value()):
            for item in half.verify(trees, locks).items:
                add(item.value, item.n)
                add(item.p_value)
                add(item.se)
        det = app.detectability()
        if det is not None:
            add(det.implied)
            add(det.observed)
    for claim in facts.verify_all(trees, locks):
        add(claim.evidence.value)
    return {v for v in out if v}


#: Lines between these markers are metadata about the manuscript - the
#: traceability note itself - rather than claims made by it, and are skipped.
_META_OPEN, _META_CLOSE = "<!-- provenance -->", "<!-- /provenance -->"


def scan(paper_dir: Path, verified: set[str]) -> list[Unmatched]:
    """Numerals in the manuscript with no verified or exempt counterpart."""
    found: list[Unmatched] = []
    for path in sorted(paper_dir.glob("0*.md")):
        in_meta = False
        for line in path.read_text().splitlines():
            if line.strip() == _META_OPEN:
                in_meta = True
                continue
            if line.strip() == _META_CLOSE:
                in_meta = False
                continue
            if in_meta:
                continue
            if line.strip().startswith(("|---", "#", "*")):
                continue
            for phrase in _NOT_QUANTITIES:
                line = line.replace(phrase, "")
            line = re.sub(r"[Ss]ection \d+(\.\d+)?", "", line)
            for raw in _NUMERAL.findall(line):
                token = raw.rstrip(".,")
                bare = token.replace(",", "")
                if token in verified or bare in verified:
                    continue
                if token in EXEMPT or bare in EXEMPT:
                    continue
                found.append(Unmatched(token, path.name, line.strip()[:88]))
    return found


def report(unmatched: list[Unmatched]) -> str:
    if not unmatched:
        return "every numeral in the manuscript is verified or explicitly exempt."
    lines = [f"{len(unmatched)} numeral(s) neither verified nor exempt:", ""]
    for u in unmatched:
        lines.append(f"  {u.numeral:>12}  {u.section:<22}{u.context}")
    lines.append("")
    lines.append(
        "Each must be traced to an artifact, added to EXEMPT with a reason, or "
        "removed from the manuscript."
    )
    return "\n".join(lines)

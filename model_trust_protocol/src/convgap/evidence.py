"""A number, its meaning, and its origin."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from convgap.extract import ExtractionError, Extractor
from convgap.provenance import (
    Source,
    SourceTier,
    SourceTree,
    VerificationStatus,
    check,
)


class Statistic(StrEnum):
    """The quantity an :class:`Evidence` value reports.

    Kept explicit because the paper's central objection to naive comparison is
    that a t-statistic, a Sharpe ratio and an AUC are not commensurable. Typing
    them prevents the package from quietly comparing across kinds.
    """

    T_STAT = "t"
    P_VALUE = "p"
    AUC = "AUC"
    LOG_LOSS = "log-loss"
    BRIER = "Brier"
    SHARPE = "Sharpe"
    CRPS = "CRPS"
    DM_STAT = "Diebold-Mariano t"
    IC = "information coefficient"
    BASIS_POINTS = "bp"
    BREACH_RATE = "breach rate"
    KUPIEC_P = "Kupiec p"
    COST_BPS = "breakeven cost (bp)"
    CAPITAL_PCT = "capital (%)"
    KAPPA = "Cohen kappa"
    RATIO = "ratio"


class Era(StrEnum):
    """Sample partition. The frozen split is 2021-04-30 / 2021-07-01."""

    TRAIN = "TRAIN"
    TEST = "TEST"
    FULL = "FULL"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One traced number.

    An ``Evidence`` is deliberately awkward to construct without a
    :class:`~convgap.provenance.Source`. That friction is the feature: it is
    what stops a remembered figure from reaching a table.
    """

    value: float
    statistic: Statistic
    source: Source
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    era: Era | None = None
    n: int | None = None
    se: float | None = None
    p_value: float | None = None
    note: str = ""
    extractor: Extractor | None = None
    """Reads the declared value back out of the artifact. Optional but expected:
    without one, verification establishes only that the file is unchanged."""
    rtol: float = 5e-3
    """Relative tolerance for the extracted-versus-declared comparison. The
    default admits values rounded to two significant figures in a document."""
    extraction_note: str = ""

    def verify(self, trees: dict[str, SourceTree], locks: dict[str, str] | None = None) -> Evidence:
        """Return a copy whose status reflects the artifact currently on disk.

        When an extractor is present the artifact is additionally asked for the
        value, and a disagreement is reported as ``MISMATCH`` rather than
        silently passing on a digest match.
        """
        status, stamped = check(self.source, trees, locks)
        if self.extractor is None or status in (
            VerificationStatus.MISSING,
            VerificationStatus.OUTDATED,
        ):
            return replace(self, status=status, source=stamped)
        try:
            found = self.extractor(stamped.path(trees))
        except (ExtractionError, OSError, ValueError) as exc:
            return replace(
                self,
                status=VerificationStatus.MISMATCH,
                source=stamped,
                extraction_note=str(exc),
            )
        if abs(found - self.value) > self.rtol * max(abs(self.value), 1e-12):
            return replace(
                self,
                status=VerificationStatus.MISMATCH,
                source=stamped,
                extraction_note=f"artifact holds {found:g}, channel declares {self.value:g}",
            )
        return replace(
            self,
            status=status,
            source=stamped,
            extraction_note=f"confirmed {found:g}",
        )

    @property
    def tier(self) -> SourceTier:
        return self.source.tier

    @property
    def label(self) -> str:
        era = f" [{self.era}]" if self.era else ""
        return f"{self.statistic} = {self.value:g}{era}"

    def __str__(self) -> str:
        return f"{self.label} ({self.status})"


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Several readings of the same quantity, typically one per era.

    Era-wise reporting is a standing requirement of the source programme: a
    result present in one era only is a different claim from one present in
    both, and collapsing them hides exactly that.
    """

    label: str
    items: tuple[Evidence, ...]

    def verify(
        self, trees: dict[str, SourceTree], locks: dict[str, str] | None = None
    ) -> EvidenceSet:
        return replace(self, items=tuple(e.verify(trees, locks) for e in self.items))

    @property
    def status(self) -> VerificationStatus:
        """Weakest status across members — a set is only as traced as its worst item."""
        order = [
            VerificationStatus.MISSING,
            VerificationStatus.OUTDATED,
            VerificationStatus.MISMATCH,
            VerificationStatus.STALE,
            VerificationStatus.UNVERIFIED,
            VerificationStatus.DERIVED,
            VerificationStatus.VERIFIED,
        ]
        return min((e.status for e in self.items), key=order.index)

    @property
    def tier(self) -> SourceTier:
        """Weakest tier across members."""
        if any(e.tier is SourceTier.SECONDARY for e in self.items):
            return SourceTier.SECONDARY
        return SourceTier.PRIMARY

    @property
    def replicates(self) -> bool | None:
        """Whether the sign is preserved across TRAIN and TEST.

        ``None`` when the set does not contain both eras.
        """
        by_era = {e.era: e for e in self.items}
        train, test = by_era.get(Era.TRAIN), by_era.get(Era.TEST)
        if train is None or test is None:
            return None
        return (train.value > 0) == (test.value > 0)

    def __str__(self) -> str:
        return f"{self.label}: " + ", ".join(e.label for e in self.items)

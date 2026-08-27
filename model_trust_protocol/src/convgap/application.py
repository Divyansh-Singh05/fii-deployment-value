"""The unit of analysis: one application of the dataset, and where its value is lost."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from convgap.benchmark import Detectability
from convgap.evidence import Evidence, EvidenceSet
from convgap.friction import CLEARS_ALL, FrictionLevel
from convgap.provenance import SourceTier, SourceTree, VerificationStatus
from convgap.recipe import Recipe
from convgap.role import Role


class Layer(StrEnum):
    """Where in the modelling stack the *significance* half was established.

    The value half is by construction always decision-relevant, so it does not
    partition. Stratifying on the significance side is what supports the
    paper's claim that the gap is not an artifact of one methodology: if every
    channel came from a single layer, the finding would be about that method
    rather than about financial prediction.
    """

    REGRESSION = "conditional-mean regression"
    DENSITY = "density forecast"
    CLASSIFICATION = "classification / hazard"
    ATTRIBUTION = "feature attribution"
    PORTFOLIO = "portfolio"


@dataclass(frozen=True, slots=True)
class ApplicationResult:
    """Verified summary of one application, ready for an exhibit row."""

    key: str
    title: str
    layer: Layer
    mechanism: str
    role: Role
    significance: EvidenceSet
    value: EvidenceSet
    detectability: Detectability | None
    fails_at: FrictionLevel | None
    """Lowest friction level the application does not survive. ``None`` means it
    survives every level."""

    @property
    def clears(self) -> int:
        """Number of friction levels survived."""
        return CLEARS_ALL if self.fails_at is None else int(self.fails_at)

    @property
    def survives(self) -> bool:
        return self.fails_at is None

    @property
    def status(self) -> VerificationStatus:
        order = [
            VerificationStatus.MISSING,
            VerificationStatus.OUTDATED,
            VerificationStatus.MISMATCH,
            VerificationStatus.STALE,
            VerificationStatus.UNVERIFIED,
            VerificationStatus.DERIVED,
            VerificationStatus.VERIFIED,
        ]
        return min((self.significance.status, self.value.status), key=order.index)

    @property
    def tier(self) -> SourceTier:
        if SourceTier.SECONDARY in (self.significance.tier, self.value.tier):
            return SourceTier.SECONDARY
        return SourceTier.PRIMARY

    @property
    def publishable(self) -> bool:
        """Traced to a live artifact *and* sourced from the computation itself.

        Both conditions are required. A verified digest on a prose document
        establishes only that nobody edited the prose.
        """
        return self.status.is_publishable and self.tier is SourceTier.PRIMARY


class Application(ABC):
    """One application of the institutional flow record, under stress.

    The dataset is the object of study. The model each subclass names - a panel
    regression, a regime model, a density engine, a book of strategies - is a
    stress-test environment, not a subject: it exists to impose one level of
    real-world friction on the same underlying data.

    Each subclass supplies two evidence sets measured on the *same object and
    the same data*: what a conventional significance test reported, and what a
    value test reported once that friction was applied. The pairing is the
    design; an entry drawing its two halves from different samples is not
    evidence of anything and must not be registered.
    """

    key: str
    title: str
    layer: Layer
    role: Role = Role.APPLICATION
    fails_at: FrictionLevel | None
    """The lowest friction level this application does not survive."""
    mechanism: str
    """Why it dies there. This is the paper's content; a label is not enough."""

    @abstractmethod
    def significance(self) -> EvidenceSet:
        """Evidence that a conventional test declares the effect present."""

    @abstractmethod
    def value(self) -> EvidenceSet:
        """Evidence on what the effect is worth to a forecast or a decision."""

    @abstractmethod
    def recipes(self) -> tuple[Recipe, ...]:
        """The commands that regenerate this application's artifacts.

        Required, not optional. An application whose numbers cannot be rebuilt
        from this repository is not reproducible, however well it is traced.

        A sequence rather than a single recipe because an application may draw
        its two halves from different computations - and, in one case here, from
        different source trees. Splitting them keeps each artifact attached to
        the command that actually writes it.
        """

    def detectability(self) -> Detectability | None:
        """Implied-versus-observed accounting, where it can be derived.

        Returning ``None`` is permitted and is reported as such: a channel
        without a benchmark contributes a qualitative row, not a conversion
        rate. It should not be silently omitted.
        """
        return None

    def resolve(
        self, trees: dict[str, SourceTree], locks: dict[str, str] | None = None
    ) -> ApplicationResult:
        """Verify every number and return the exhibit row."""
        return ApplicationResult(
            key=self.key,
            title=self.title,
            layer=self.layer,
            role=self.role,
            mechanism=self.mechanism,
            fails_at=self.fails_at,
            significance=self.significance().verify(trees, locks),
            value=self.value().verify(trees, locks),
            detectability=self.detectability(),
        )

    def __repr__(self) -> str:
        where = "survives" if self.fails_at is None else f"dies at {self.fails_at.label}"
        return f"<{type(self).__name__} {self.key} ({where})>"


def evidence_set(label: str, *items: Evidence) -> EvidenceSet:
    """Convenience constructor used throughout :mod:`convgap.applications`."""
    if not items:
        raise ValueError(f"evidence set {label!r} must contain at least one reading")
    return EvidenceSet(label=label, items=items)

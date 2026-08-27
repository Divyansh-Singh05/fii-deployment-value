"""What is an alternative dataset worth under ascending real-world friction?

The object of study is a fourteen-year masked transaction record of foreign
institutional flow. Every model this package refers to - a cross-sectional
regression, a hidden Markov regime model, a stock-day predictive density, a
market-level density engine, a book of long/short strategies - is a
**stress-test environment**, not a subject: each exists to impose one level of
institutional friction on the same underlying data.

The package encodes a single argument. For each application of the record it
pairs the result a conventional significance test reported with the result a
value test reported on the same object and the same data, and places the
application on a friction ladder according to where its value is lost. Every
number carries the artifact that produced it and a digest of that artifact, so
an exhibit can state which of its entries have been traced to a live
computation and which have not.
"""

from convgap.application import Application, ApplicationResult, Layer
from convgap.benchmark import Detectability
from convgap.evidence import Era, Evidence, EvidenceSet, Statistic
from convgap.friction import CLEARS_ALL, FrictionLevel
from convgap.provenance import (
    Source,
    SourceTier,
    SourceTree,
    VerificationStatus,
    load_source_trees,
)
from convgap.registry import (
    all_applications,
    by_friction,
    by_instrument,
    controls,
    coverage_gaps,
    register,
    survivors,
)
from convgap.role import Role

__version__ = "0.1.0"

__all__ = [
    "CLEARS_ALL",
    "Application",
    "ApplicationResult",
    "Detectability",
    "Era",
    "Evidence",
    "EvidenceSet",
    "FrictionLevel",
    "Layer",
    "Role",
    "Source",
    "SourceTier",
    "SourceTree",
    "Statistic",
    "VerificationStatus",
    "all_applications",
    "by_friction",
    "by_instrument",
    "controls",
    "coverage_gaps",
    "load_source_trees",
    "register",
    "survivors",
]

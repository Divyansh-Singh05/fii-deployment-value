from __future__ import annotations

import pytest

from convgap import facts
from convgap.provenance import SourceTier


def test_every_claim_cites_a_primary_source() -> None:
    for claim in facts.CLAIMS:
        assert claim.evidence.source.tier is SourceTier.PRIMARY
        assert claim.evidence.source.producer, (
            "a descriptive computed here must name the code that computes it, "
            "so the freshness gate applies"
        )


def test_every_claim_carries_an_extractor() -> None:
    for claim in facts.CLAIMS:
        assert claim.evidence.extractor is not None, (
            "a data-section numeral without an extractor is only as good as the "
            "transcription that produced it"
        )


@pytest.mark.parametrize("claim", facts.CLAIMS, ids=lambda c: c.evidence.note)
def test_claim_text_is_a_sentence(claim: facts.DataClaim) -> None:
    assert claim.text.endswith("."), "a claim is an assertion the paper makes"
    assert len(claim.text.split()) >= 5


def test_claims_are_not_silently_dropped() -> None:
    assert len(facts.CLAIMS) >= 12, (
        "the data section's numerals are enumerated here; shrinking this list "
        "removes a claim from verification without removing it from the paper"
    )

import pytest

from support_categorizer.pii import pii_findings, pii_rate


@pytest.mark.parametrize("text, kind", [
    ("Bronze customer Rodriguez Domingo asks about a refund", "membership level"),
    ("Gold member wants free shipping", "membership level"),
    ("Customer Chloe Zhang complains about stock", "name"),
    ("Refund sent to cminh730@email.com", "email"),
    ("Call back on (977) 625-2661", "phone"),
    ("Order 3348917502 is late", "long number"),
])
def test_pii_findings_hits(text, kind):
    assert kind in pii_findings(text)


@pytest.mark.parametrize("text", [
    "Customer asks about the refund status of a returned item.",
    "Customer wants free international shipping on their plan.",
    "Customer reports the website is slow; agent clears cookies.",
    "",
    None,
])
def test_pii_findings_clean(text):
    assert pii_findings(text) == []


def test_pii_rate():
    assert pii_rate(["Gold member X", "clean text", None, "Customer Ann Lee called"]) == pytest.approx(2 / 3)
    assert pii_rate([]) == 0.0


def test_instructions_carry_the_pii_rule():
    from support_categorizer.categorizer import INSTRUCTION
    from support_categorizer.sql import AI_QUERY_INSTRUCTION

    for text in (INSTRUCTION, AI_QUERY_INSTRUCTION):
        assert "membership level" in text and "names" in text

from support_categorizer.transcript import build_transcript, real_turn_count


def test_build_transcript_drops_actions_and_collapses_space():
    turns = [("agent", "Hi!"), ("action", "Account pulled"), ("customer", "  need   help ")]
    assert build_transcript(turns) == "Agent: Hi!\nCustomer: need help"


def test_truncates_on_line_boundary():
    turns = [("customer", "a" * 30), ("agent", "b" * 30), ("customer", "c" * 30)]
    out = build_transcript(turns, max_chars=80)
    assert len(out) <= 80
    assert out == "Customer: " + "a" * 30 + "\nAgent: " + "b" * 30


def test_real_turn_count():
    assert real_turn_count([("agent", "x"), ("action", "y"), ("customer", "z")]) == 2

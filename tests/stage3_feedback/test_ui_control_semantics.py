from tokenthrift.ui.controls import CONTROL_EXPLANATION


def test_control_copy_distinguishes_the_three_effect_categories():
    text = CONTROL_EXPLANATION.lower()
    assert "only generation" in text
    assert "adjust the pruning policy" in text
    assert "experimental session-local model update" in text

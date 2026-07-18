"""Tests for Inference.seg_constrained (jm8.9).

The hard guarantee is structural: output must concatenate back to the
input (re-derivability), with no empty padas — for ANY input. Boundary
QUALITY is measured by evals/eval.py --ab (constrained arm), not locked
here, because it shifts with retraining; the guarantee must not.
"""
from slm.infer import Inference

_inf = Inference()

CASES = [
    "tasminvirUpAkzenihate",
    "vanamidaMdurgaMpUrRaM",
    "kamarcantaH",
    "ityaham",
    "rAmaH",                      # single word: must not force a split
    "a",                          # degenerate 1-char input
    "nametURISayAnbARAnsavizAnivapannagAn",  # long val example
]


def test_rederivability_guarantee():
    for text in CASES:
        r = _inf.seg_constrained(text)
        assert "".join(r["padas"]) == text, (text, r["padas"])


def test_no_empty_padas():
    for text in CASES:
        r = _inf.seg_constrained(text)
        assert all(r["padas"]), (text, r["padas"])
        assert len(r["padas"]) >= 1


def test_marks_itself_constrained():
    assert _inf.seg_constrained("rAmaH")["constrained"] is True


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                failed += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)

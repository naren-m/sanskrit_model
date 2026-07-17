"""Tests for SandhiEngine.split() candidate generation.

The split() lattice is a *proposer*: downstream (slm.pipeline.Analyzer)
rescores the pool with lexicon coverage + model logprob. These tests
therefore assert pool MEMBERSHIP, not rank — a correct split that never
enters the pool can never be chosen, no matter how good the scorer.

Bug being locked down (2026-07-17): DFS explored rule-match branches
before the skip branch, so minimal-boundary splits (1 boundary, many
skipped chars) were starved out of the candidate pool by combinatorial
over-split garbage on any input with spurious early rule matches.
"""
from slm.rules import SandhiEngine

_se = SandhiEngine()


def _pool(text: str) -> list[list[str]]:
    return _se.split(text, max_results=20)


def test_rutva_seam_in_pool():
    # prApnuyuH + mAnavAH; CSV encodes pada-final as underlying s (rutva
    # 8.2.66) or surface r (repha-prakfti 8.3.15) — either form is correct.
    pool = _pool("prApnuyurmAnavAH")
    assert (["prApnuyus", "mAnavAH"] in pool
            or ["prApnuyur", "mAnavAH"] in pool), pool


def test_utva_guna_seam_in_pool():
    # rAmaH + asti → rAmo'sti (as,a → o ' — 6.1.113)
    pool = _pool("rAmo'sti")
    assert ["rAmas", "asti"] in pool, pool


def test_guna_vowel_seam_in_pool():
    # sUrya + udayaH → sUryodayaH (a,u → o — guNa 6.1.87)
    pool = _pool("sUryodayaH")
    assert ["sUrya", "udayaH"] in pool, pool


def test_zero_change_m_vowel_boundary_in_pool():
    # kam + arcantaH: word-final m before a vowel is unchanged (8.3.23
    # mo'nusvAraH applies only before hal). Needs identity m+ac rows in
    # sandhi-rules-full.csv for the lattice to propose the boundary.
    pool = _pool("kamarcantaH")
    assert ["kam", "arcantaH"] in pool, pool


# --- regressions: previously-working seams must stay in the pool ---------

def test_regression_vapyekam():
    pool = _pool("vApyekaM")
    assert (["va", "api", "ekaM"] in pool
            or ["vA", "api", "ekaM"] in pool), pool


def test_regression_taddhitam():
    pool = _pool("tadDitam")
    assert ["tat", "hitam"] in pool, pool


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}")
    sys.exit(1 if failed else 0)

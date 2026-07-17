"""Tests for Analyzer scorer (slm.pipeline) — tasks-jm8.13.

split() (the proposer) now puts correct splits IN the candidate pool
(see tests/test_split.py). These tests lock down that the SCORER then
*selects* the correct candidate. Two scorer fixes are exercised:

A. Pausa normalization: split() returns Paninian underlying pada-finals
   (rAmas, prApnuyur); corpus vocab holds surface/pausa forms (rAmaH).
   Word-final s/r -> H (8.3.15 kharavasAnayoH) before lexicon lookup.

B. Frequency-weighted lexicon credit: binary in-vocab coverage can't
   break a tie between a real split and one containing a junk hapax that
   also happens to be in vocab. Log-frequency credit favors the split
   made of high-corpus-frequency words.
"""
from slm.pipeline import Analyzer

_a = Analyzer()


def _best(text: str) -> list[str]:
    return _a.segment(text)["best"]["padas"]


# --- Fix A: pausa normalization -----------------------------------------

def test_pausa_rescues_ramosti():
    # rAmaH+asti -> rAmo'sti. split() proposes underlying rAmas|asti;
    # rAmas is OOV but pausa(rAmas)=rAmaH is freq-738. Must beat the
    # spurious rAma|u'sti (u'sti is OOV).
    assert _best("rAmo'sti") == ["rAmas", "asti"]


# --- Fix B: frequency-weighted lexicon credit ---------------------------

def test_freq_weight_breaks_taddhitam_tie():
    # tat(1137)+hitam(74) must beat tadDi(4)+tam(687): both splits are
    # fully in-vocab (binary lex=1.0 tie), freq credit decides.
    assert _best("tadDitam") == ["tat", "hitam"]


# --- selection regressions (correct split must stay selected) -----------

def test_select_suryodayah():
    assert _best("sUryodayaH") == ["sUrya", "udayaH"]


def test_select_gangodakam():
    assert _best("gaNgodakam") == ["gaNgA", "udakam"]


def test_select_kamarcantah():
    assert _best("kamarcantaH") == ["kam", "arcantaH"]


def test_select_prapnuyur_manavah():
    # prApnuyuH is absent from the corpus entirely (optative, unattested),
    # so this stays lex<1.0 -- but the CORRECT split must still be picked.
    assert _best("prApnuyurmAnavAH") == ["prApnuyur", "mAnavAH"]


def test_select_ityaham():
    assert _best("ityaham") == ["iti", "aham"]


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

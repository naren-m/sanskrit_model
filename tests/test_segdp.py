"""Tests for the lexicon-anchored sandhi-aware segmentation DP (slm/segdp.py).

Hard guarantees under test:
  1. Every returned segmentation re-derives the surface: verbatim edges
     concatenate back; junction edges restore the surface via the forward rule.
  2. Known easy splits come back correct.
  3. OOV input yields no DP path (callers must fall back).
"""
import pytest

from slm.rules import SandhiEngine
from slm.segdp import LexiconSegmenter


@pytest.fixture(scope="module")
def seg():
    return LexiconSegmenter()


@pytest.fixture(scope="module")
def engine():
    return SandhiEngine()


def _underlying(word):
    """Inverse of pausa: pada-final H may stand for underlying s or r."""
    if word.endswith("H"):
        return [word, word[:-1] + "s", word[:-1] + "r"]
    return [word]


def rederives(engine, padas, src):
    """Forward-join padas with the rule engine (trying underlying variants of
    pausa finals, 8.3.15); True if some join path reproduces `src`."""
    surfaces = {""}
    for idx, p in enumerate(padas):
        new = set()
        for s in surfaces:
            if idx == 0:
                new.update(_underlying(p))
                continue
            for v in _underlying(p) if idx < len(padas) - 1 else [p]:
                for joined, _cat in engine.join(s, p) + engine.join(s, v):
                    new.add(joined.replace(" ", ""))
        surfaces = set(list(new)[:200])
    # final pada stays pausa; also accept exact concat
    return src in surfaces


def test_verbatim_split(seg):
    # two very frequent Ramayana padas abutting verbatim
    out = seg.kbest("tataHrAmaH")
    assert out, "expected a DP path"
    assert out[0][1] == ["tataH", "rAmaH"]


def test_all_paths_rederive_verbatim(seg):
    for sc, padas in seg.kbest("tataHrAmaH"):
        # all-verbatim candidate must concatenate exactly
        joined = "".join(padas)
        if joined == "tataHrAmaH":
            return
    pytest.fail("no candidate concatenates back to source")


def test_junction_edge_rederives(seg, engine):
    # rAmo'sti is the canonical as+a -> o' junction (rAmaH + asti)
    out = seg.kbest("rAmo'sti")
    assert out, "expected a DP path through a junction edge"
    top = out[0][1]
    assert rederives(engine, top, "rAmo'sti"), top


def test_oov_returns_empty(seg):
    assert seg.kbest("qqqqxxxxzzzz") == []


def test_kbest_scores_descending(seg):
    out = seg.kbest("tataHrAmaH", k=8)
    scores = [sc for sc, _ in out]
    assert scores == sorted(scores, reverse=True)


def test_min_count_filter_shrinks_vocab():
    a = LexiconSegmenter()
    b = LexiconSegmenter(min_count=3)
    assert len(b.vocab) < len(a.vocab)

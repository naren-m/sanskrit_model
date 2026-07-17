"""evals/baseline.py — trigram + pure-symbolic baselines for the A/B gate
(tasks-jm8.1, spec §7).

The neural model's ONLY job in the segmentation pipeline is to RANK
lattice-licensed candidates (ADR jm8.12: model as ranker, not generator).
So the honest ablation holds everything else fixed and swaps just the
ranker's tie-break term:

  * symbolic : lexicon coverage + log-frequency only (no sequence scorer)
  * trigram  : + interpolated char-trigram logprob of the joined split
  * neural   : + the 10.7M GPT's teacher-forced logprob (production scorer)

All three arms score the IDENTICAL candidate pool (SandhiEngine.split
lattice + no-split fallback). The neural arm's greedy-decode candidate is
deliberately NOT injected here — that would change the pool, not the
ranker, and confound the comparison.

Trigram training data: the same corpus the GPT saw (ramayanam padas +
yoga sutra lines from data/corpus.json), so neither arm has a data
advantage.

Run:  uv run python -m evals.eval --ab
"""
from __future__ import annotations

import math
from collections import defaultdict

from slm import corpus
from slm.rules import SandhiEngine

_BOS = "\x02"  # sentinel chars kept out of SLP1's alphabet


class CharTrigram:
    """Interpolated char trigram LM with add-k smoothing.

    logprob() is mean per-char log P — length-normalized exactly like
    Inference.logprob, so the two plug into the same scoring slot."""

    def __init__(self, lines: list[str], k: float = 0.1,
                 lambdas: tuple[float, float, float] = (0.7, 0.2, 0.1)):
        self.k = k
        self.l3, self.l2, self.l1 = lambdas
        self.tri: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.bi: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.uni: dict[str, int] = defaultdict(int)
        self.total = 0
        for ln in lines:
            s = _BOS + _BOS + ln
            for i in range(2, len(s)):
                self.tri[s[i - 2:i]][s[i]] += 1
                self.bi[s[i - 1]][s[i]] += 1
                self.uni[s[i]] += 1
                self.total += 1
        self.vsize = len(self.uni) + 1

    def _p(self, ctx2: str, ch: str) -> float:
        t = self.tri.get(ctx2)
        p3 = (t[ch] + self.k) / (sum(t.values()) + self.k * self.vsize) if t else 0.0
        b = self.bi.get(ctx2[-1])
        p2 = (b[ch] + self.k) / (sum(b.values()) + self.k * self.vsize) if b else 0.0
        p1 = (self.uni[ch] + self.k) / (self.total + self.k * self.vsize)
        return self.l3 * p3 + self.l2 * p2 + self.l1 * p1

    def logprob(self, text: str) -> float:
        if not text:
            return 0.0
        s = _BOS + _BOS + text
        lp = 0.0
        for i in range(2, len(s)):
            lp += math.log(self._p(s[i - 2:i], s[i]))
        return lp / len(text)


class AblationRanker:
    """Score-identical reimplementation of Analyzer.segment()'s formula with
    a pluggable sequence-scorer arm. Kept separate from slm.pipeline so the
    production Analyzer and the eval harness can't drift silently into
    scoring different things — this file IS the spec of what's compared."""

    def __init__(self, seq_scorers: dict[str, object], min_count: int = 1):
        c = corpus.load_corpus()
        self.vocab: dict[str, int] = {w: n for w, n in c["vocab"].items() if n >= min_count}
        self._log_max = math.log1p(max(self.vocab.values())) if self.vocab else 1.0
        self.sandhi = SandhiEngine()
        self.seq_scorers = seq_scorers  # arm -> obj with .logprob, or None

    # -- mirrors of Analyzer scoring (pausa + freq weighting) -----------------
    def _pausa(self, w: str) -> str:
        return w[:-1] + "H" if w.endswith(("s", "r")) else w

    def _freq(self, p: str) -> int:
        return max(self.vocab.get(p, 0), self.vocab.get(self._pausa(p), 0))

    def _lex(self, padas: list[str]) -> tuple[float, float, int]:
        n = len(padas) or 1
        cov = sum(1 for p in padas if self._freq(p) > 0) / n
        fw = sum(math.log1p(self._freq(p)) for p in padas) / (n * self._log_max)
        oov = sum(1 for p in padas if self._freq(p) == 0)
        return cov, fw, oov

    def candidates(self, text: str, k_lattice: int = 20) -> list[list[str]]:
        cands = list(self.sandhi.split(text, max_results=k_lattice))
        if [text] not in cands:
            cands.append([text])
        return cands

    def rank(self, text: str, arm: str,
             cands: list[list[str]] | None = None) -> list[str]:
        scorer = self.seq_scorers[arm]
        best, best_s = None, -1e18
        for padas in (cands if cands is not None else self.candidates(text)):
            cov, fw, oov = self._lex(padas)
            seq = scorer.logprob(" | ".join(padas)) if scorer else 0.0
            s = 4.0 * cov + 0.5 * fw - 0.6 * oov + 0.15 * seq - 0.05 * len(padas)
            if s > best_s:
                best_s, best = s, padas
        return best or [text]


class NeuralSeqScorer:
    """Adapts Inference.logprob(src, tgt) to the one-arg trigram interface;
    the <seg> src prompt is how the production Analyzer calls it."""

    def __init__(self, inf, src_text_holder: dict):
        self.inf = inf
        self.holder = src_text_holder  # {'src': current input text}

    def logprob(self, joined: str) -> float:
        return self.inf.logprob(f"<seg>{self.holder['src']}", joined)


def word_f1(pred: list[str], gold: list[str]) -> float:
    """Multiset word F1 — partial credit when some padas match."""
    from collections import Counter
    pc, gc = Counter(pred), Counter(gold)
    tp = sum((pc & gc).values())
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gold)
    return 2 * prec * rec / (prec + rec)


def training_lines() -> list[str]:
    c = corpus.load_corpus()
    lines = [" ".join(p) for p in c["ramayanam_padas"]]
    lines += c["yoga_lines"]
    return lines

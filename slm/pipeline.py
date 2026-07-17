"""slm/pipeline.py — the neuro-symbolic analyzer (spec §4 keystone, jm8.11 v0).

Mixes all four layers so a small model suffices (ADR jm8.12):

  1. LATTICE   (symbolic) SandhiEngine.split enumerates every rule-licensed
     segmentation of the sandhied input — exhaustive, never wrong about what a
     rule permits, but over-generates and can't rank.
  2. LEXICON   (symbolic) real corpus vocabulary (slm.corpus) gates candidates:
     a split whose padas are attested words beats one that isn't.
  3. MODEL     (neural)   ranks the survivors — Inference.logprob scores the
     <seg> mapping, and the model's own greedy decode is added as a candidate.
     The net proposes/ranks; it never invents characters outside the lattice.
  4. VERIFY    (symbolic) each chosen pada is checked against the DhatuKosha /
     (future) vidyut round-trip. Nothing is asserted the engine can't confirm.

v0 scope: the SEGMENTATION half is fully wired. Per-pada morphological analysis
(inflected surface -> root+features) needs vidyut-generated data (jm8.3) and is
reported as pending, not faked.

Run:  uv run python -m slm.pipeline
"""
from __future__ import annotations

import math

from slm import corpus
from slm.infer import Inference


def _pausa(word: str) -> str:
    """Pausa (utterance-final) form of a pada: word-final s or r -> visarga H
    (8.3.15 kharavasAnayor visarjanIyaH). split() returns pada-finals in
    Paninian *underlying* form (rAmas, prApnuyur) because that is how
    sandhi-rules-full.csv encodes them; the corpus vocabulary stores the
    *surface/pausa* form (rAmaH). Every pada is by construction utterance-
    final within its candidate, so this rewrite is always the licensed one.
    Words already ending in a vowel / anusvAra / H / other consonant are
    returned unchanged."""
    if word.endswith(("s", "r")):
        return word[:-1] + "H"
    return word


class Analyzer:
    def __init__(self, min_count: int = 1):
        self.inf = Inference()
        c = corpus.load_corpus()
        self.vocab: dict[str, int] = {w: n for w, n in c["vocab"].items() if n >= min_count}
        # normaliser for log-frequency credit (see _lex_weight)
        self._log_max = math.log1p(max(self.vocab.values())) if self.vocab else 1.0

    # -- ranking helpers ------------------------------------------------------
    def _freq(self, pada: str) -> int:
        """Corpus frequency of a pada, trying its pausa (surface-visarga)
        form so an underlying-form pada from split() matches surface vocab."""
        return max(self.vocab.get(pada, 0), self.vocab.get(_pausa(pada), 0))

    def _lex_coverage(self, padas: list[str]) -> float:
        """Fraction of padas attested in the corpus (pausa-normalised). This
        is the binary symbolic GATE — it drives the dominant score term."""
        if not padas:
            return 0.0
        return sum(1 for p in padas if self._freq(p) > 0) / len(padas)

    def _lex_weight(self, padas: list[str]) -> float:
        """Mean log-frequency of the padas, normalised to [0, 1]. A finer
        signal than coverage: among splits with EQUAL coverage it favours the
        one built from high-frequency real words, so a junk hapax that merely
        happens to be in vocab (e.g. 'tadDi', freq 4) cannot tie a genuine
        high-frequency split ('tat' 1137 + 'hitam' 74). Replaces the model's
        copy-biased logprob as the primary tie-breaker."""
        if not padas:
            return 0.0
        return sum(math.log1p(self._freq(p)) for p in padas) / (len(padas) * self._log_max)

    def _candidates(self, text: str, k_lattice: int = 20) -> list[list[str]]:
        cands = list(self.inf.sandhi.split(text, max_results=k_lattice))
        # add the model's own greedy segmentation as a candidate
        seg = self.inf.seg(text)
        if seg["padas"]:
            cands.append(seg["padas"])
        cands.append([text])  # the no-split fallback
        # dedup, preserving order
        seen, uniq = set(), []
        for c in cands:
            key = " ".join(c)
            if key not in seen:
                seen.add(key); uniq.append(c)
        return uniq

    def segment(self, text: str, k_lattice: int = 20) -> dict:
        text = text.strip()
        cands = self._candidates(text, k_lattice)
        scored = []
        for padas in cands:
            lex = self._lex_coverage(padas)
            freqw = self._lex_weight(padas)
            oov = sum(1 for p in padas if self._freq(p) == 0)
            model = self.inf.logprob(f"<seg>{text}", " | ".join(padas))
            # Lexicon coverage DOMINATES (symbolic gate, 4.0); each out-of-vocab
            # pada takes a hard hit so a whole-string non-word can't win on the
            # model's love of copying its input. Among coverage-equal splits,
            # log-frequency (0.5) is the primary tie-break -- it outranks the
            # model (0.15) so a genuine high-frequency split beats a junk-hapax
            # one instead of deferring to the net's copy bias. Small penalty vs
            # over-splitting.
            combined = (4.0 * lex + 0.5 * freqw - 0.6 * oov
                        + 0.15 * model - 0.05 * len(padas))
            scored.append({"padas": padas, "lex_coverage": round(lex, 3),
                           "lex_weight": round(freqw, 3), "oov": oov,
                           "model_logprob": round(model, 3),
                           "score": round(combined, 3)})
        scored.sort(key=lambda s: -s["score"])
        return {"input": text, "n_candidates": len(scored),
                "best": scored[0], "ranked": scored[:6]}

    def analyze(self, text: str) -> dict:
        """Full analyze: segment, then annotate each pada (lexicon + morph stub)."""
        seg = self.segment(text)
        padas = seg["best"]["padas"]
        annotated = []
        for p in padas:
            entry = {"pada": p, "in_lexicon": p in self.vocab,
                     "corpus_freq": self.vocab.get(p, 0)}
            # morphology of an inflected surface needs vidyut (jm8.3); only a
            # bare root is verifiable today via the DhatuKosha.
            hits = self.inf.kosha.lookup(p)
            if hits:
                entry["root_analysis"] = [{"code": h["code"], "gana": h["gana"],
                                           "artha": h["artha_slp1"]} for h in hits[:2]]
            annotated.append(entry)
        return {"input": text, "segmentation": padas,
                "lex_coverage": seg["best"]["lex_coverage"],
                "padas": annotated,
                "note": "per-pada morphology of inflected forms pending vidyut (jm8.3)"}


def _demo():
    a = Analyzer()
    print(f"model val_bpb={a.inf.val_bpb:.3f}  lexicon={len(a.vocab)} words\n")
    tests = ["ityaham", "tadDitam", "rAmo'sti", "devendra", "sUryodayaH",
             "gaNgodakam", "tapovanam"]
    for t in tests:
        r = a.segment(t)
        b = r["best"]
        print(f"{t:14s} → {' | '.join(b['padas']):28s}  "
              f"lex={b['lex_coverage']}  lp={b['model_logprob']}  ({r['n_candidates']} cands)")
    print("\n--- full analyze ---")
    import json
    print(json.dumps(a.analyze("tadDitam"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()

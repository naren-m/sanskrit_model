"""slm/segdp.py — lexicon-anchored, sandhi-aware segmentation DP (jm-improve-5pt).

Tier 1 of the segmentation product. A semi-Markov Viterbi over the surface
string with two edge types:

  * verbatim edge: a substring that is a known pada (corpus lexicon, scored by
    log unigram frequency);
  * junction edge: invert ONE non-identity sandhi rule (first, second) -> result
    at the point where `result` occurs in the surface, but ONLY when both
    reconstructed sides are themselves lexicon padas. This anchoring is what
    the failed rule-inversion lattice lacked: rules alone over-license 11.4x
    (evals A1d); rules *conditioned on the lexicon* license almost nothing
    wrong.

Re-derivability is preserved by construction: every verbatim edge concatenates
back to the surface, and every junction edge consumes exactly `result` while
emitting `...first` / `second...`, i.e. the forward rule application restores
the surface span.

k-best paths are returned so the caller (Inference.seg_dp) can rescore with
the char-LM: joint score = dp_score + lm_mean_logprob * len(candidate_text).
The LM resolves what frequency cannot — e.g. whether this corpus writes a
pada-final nasal as `m` or anusvara `M` (gold is idiosyncratic per word).

When no full-cover path exists (OOV text), callers fall back to the
copy-constrained neural decode (Inference.seg_constrained) — measured 0% DP
coverage under a gold-pada-ban stress test, so the neural tier is the
robustness floor, not an ornament.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAX_WORD = 30          # longest lexicon pada considered
RULE_PENALTY = 1.0     # per junction-edge log-space penalty (favors verbatim)
DEFAULT_K = 8
# lexicon hygiene default: entries with freq<=10 that decompose into >=2 other
# entries of freq>=50 are multi-word junk spans, not padas. Tuned on a 500-row
# dev split (fresh windows, deduped against val); dev plateau is flat around
# (penalty 1.0, prune (10,50)) — see docs/plan/debate-notes.md.
DEFAULT_PRUNE = (10, 50)


_VOWELS = set("aAiIuUfFxXeEoO")
# Panini-licensed pada-final nasal alternations (8.3.23 anusvara, 8.4.58
# homorganic). OFF by default: the dev gain (+5/500 with LM rescoring) did not
# replicate on val (-13/400) — the extra candidates displace gold in k-best
# more often than they supply the missing spelling. See debate-notes.md.
_NASAL_VARIANTS = {"m": ("M", "n"), "M": ("m", "n"), "n": ("M", "m")}


def pausa(word: str) -> str:
    """Pausa (utterance-final) form: word-final s/r -> visarga H (8.3.15).
    Sandhi rules encode pada-finals underlying (rAmas); the corpus lexicon and
    the gold segmentations store the pausa surface (rAmaH)."""
    if word.endswith(("s", "r")):
        return word[:-1] + "H"
    return word


class LexiconSegmenter:
    def __init__(self, vocab: dict[str, int] | None = None, rules=None,
                 min_count: int = 1,
                 prune_spans: tuple[int, int] | None = DEFAULT_PRUNE):
        if vocab is None:
            vocab = json.loads((ROOT / "data" / "corpus.json").read_text())["vocab"]
        if min_count > 1:
            vocab = {w: c for w, c in vocab.items() if c >= min_count}
        if prune_spans is not None:
            # lexicon hygiene: the corpus vocabulary contains multi-word spans
            # captured as single "padas" (e.g. a verb phrase glued by the
            # source's Devanagari run detection). A rare entry that decomposes
            # verbatim into >=2 much-more-frequent entries is such a span, and
            # it crowds out the correct split. (max_freq, min_part_freq)
            max_f, min_pf = prune_spans
            vocab = self._prune_span_entries(vocab, max_f, min_pf)
        self.vocab = vocab
        self._logtotal = math.log(sum(vocab.values()))
        if rules is None:
            from slm.rules import SandhiEngine
            rules = SandhiEngine().rules
        # result-string -> [(first, second), ...], identity rules excluded
        # (verbatim edges already cover zero-change abutments)
        self.redges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in rules:
            res, f, s = r["_res_ns"], r["first_slp1"], r["second_slp1"]
            if not res or not f or not s or res == f + s:
                continue
            self.redges[res].append((f, s))
        self.results = sorted(self.redges, key=len, reverse=True)

    @staticmethod
    def _prune_span_entries(vocab: dict[str, int], max_freq: int,
                            min_part_freq: int) -> dict[str, int]:
        keep = dict(vocab)
        suspects = [w for w, c in vocab.items() if c <= max_freq and len(w) >= 8]
        for w in suspects:
            n = len(w)
            # can w be covered verbatim by >=2 frequent OTHER entries?
            reach = [False] * (n + 1)
            parts = [0] * (n + 1)
            reach[0] = True
            for j in range(1, n + 1):
                for i in range(max(0, j - MAX_WORD), j):
                    if not reach[i]:
                        continue
                    part = w[i:j]
                    if part != w and vocab.get(part, 0) >= min_part_freq:
                        reach[j] = True
                        parts[j] = max(parts[j], parts[i] + 1)
            if reach[n] and parts[n] >= 2:
                del keep[w]
        return keep

    def _score(self, w: str) -> float | None:
        c = self.vocab.get(w)
        return math.log(c) - self._logtotal if c else None

    def kbest(self, src: str, k: int = DEFAULT_K,
              oov_penalty: float | None = None,
              nasal_variants: bool = False) -> list[tuple[float, list[str]]]:
        """Top-k (score, padas) full-cover segmentations of `src`; [] if none.

        With `oov_penalty` set, an unknown span may be consumed as a single
        OOV pada at that (heavy) per-char penalty, so one unseen word degrades
        only its own span instead of dumping the whole line to the caller's
        fallback tier."""
        n = len(src)
        # state (pos, pending-initial) -> top-k [(score, padas-tuple)]
        best: dict[tuple[int, str], list] = defaultdict(list)
        best[(0, "")] = [(0.0, ())]
        order = [(0, "")]
        seen = {(0, "")}

        def push(state, item):
            lst = best[state]
            lst.append(item)
            # deterministic tie-break: score desc, then candidate text
            lst.sort(key=lambda t: (-t[0], t[1]))
            del lst[k:]
            if state not in seen:
                seen.add(state)
                order.append(state)

        qi = 0
        while qi < len(order):
            st = order[qi]
            qi += 1
            pos, pend = st
            if pos == n:
                continue
            for sc, words in list(best[st]):
                for j in range(pos + 1, min(n, pos + MAX_WORD) + 1):
                    w = pend + src[pos:j]
                    ws = self._score(w)
                    if ws is not None:
                        push((j, ""), (sc + ws, words + (w,)))
                    # nasal-variant edges (see _NASAL_VARIANTS note; opt-in)
                    if (nasal_variants and w and w[-1] in _NASAL_VARIANTS
                            and j < n
                            and src[j] not in _VOWELS and src[j] not in "MH'"):
                        for fin in _NASAL_VARIANTS[w[-1]]:
                            alt = w[:-1] + fin
                            ws2 = self._score(alt)
                            if ws2 is not None:
                                push((j, ""), (sc + ws2, words + (alt,)))
                    for res in self.results:
                        if not src.startswith(res, j):
                            continue
                        for f, s in self.redges[res]:
                            w2 = pend + src[pos:j] + f
                            # rules carry underlying finals (as/ur); the lexicon
                            # and gold carry pausa forms (aH/uH) — try both,
                            # emit the attested one
                            for cand in (w2, pausa(w2)):
                                ws2 = self._score(cand)
                                if ws2 is None:
                                    continue
                                push((j + len(res), s),
                                     (sc + ws2 - RULE_PENALTY, words + (cand,)))
                                break
                if oov_penalty is not None:
                    # unknown-span edge: consume src[pos:j] as one OOV pada
                    w = pend + src[pos:j]
                    if w not in self.vocab:
                        push((j, ""), (sc + oov_penalty * len(w), words + (w,)))
        return [(sc, list(ws)) for sc, ws in best.get((n, ""), [])]

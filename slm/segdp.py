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


# trie terminal marker: a sentinel object never equal to any 1-char key, so a
# node's word-end flag can never collide with a child edge (SLP1 chars).
_TERMINAL = object()


def _build_trie(keys, value=True):
    """Char trie over `keys`. Each terminal node maps _TERMINAL -> `value`, or,
    when `value` is callable, -> value(key) (used to stash the result string)."""
    root: dict = {}
    for key in keys:
        node = root
        for ch in key:
            nxt = node.get(ch)
            if nxt is None:
                nxt = {}
                node[ch] = nxt
            node = nxt
        node[_TERMINAL] = value(key) if callable(value) else value
    return root


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
        # rank each result by its position in `self.results` so per-position
        # matches can be replayed in the exact order the old linear scan used
        # (push() re-sorts, but preserving order keeps state-discovery — hence
        # k-best output — byte-identical).
        self._res_rank = {res: i for i, res in enumerate(self.results)}
        # trie over lexicon keys: walk it from a position to enumerate only the
        # spans that are known padas (verbatim) or prefixes of one (so a
        # junction edge `pref+first` could still complete), instead of scanning
        # all MAX_WORD lengths. A falling-off walk provably yields no further
        # verbatim/junction edge, so the DP can stop early.
        self._trie = _build_trie(self.vocab)
        # trie over junction-result strings (<=4 chars); terminal stashes the
        # result so a single left-to-right walk collects every rule result that
        # begins at each surface position.
        self._rtrie = _build_trie(self.redges, value=lambda k: k)

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

        # results starting at each surface position, in self.results order —
        # replaces the per-(pos,item,j) `for res in self.results` scan (792
        # results) with one left-to-right rtrie walk reused across all states.
        res_at: list[list[str]] = [[] for _ in range(n + 1)]
        for p in range(n):
            node = self._rtrie
            q = p
            while q < n:
                node = node.get(src[q])
                if node is None:
                    break
                q += 1
                res = node.get(_TERMINAL)
                if res is not None:
                    res_at[p].append(res)
            if len(res_at[p]) > 1:
                res_at[p].sort(key=self._res_rank.__getitem__)

        # the nasal edge (opt-in) can fire even where the span is not a pada
        # prefix, so it needs the full length scan; every other edge stops being
        # possible past a trie miss, so the common path breaks out early.
        trie = self._trie

        qi = 0
        while qi < len(order):
            st = order[qi]
            qi += 1
            pos, pend = st
            if pos == n:
                continue
            jmax = min(n, pos + MAX_WORD)
            # trie node for the pending-initial prefix; None => `pend` is not a
            # prefix of any pada, so no span from here can be/extend to a pada.
            base = trie
            for ch in pend:
                base = base.get(ch)
                if base is None:
                    break
            for sc, words in list(best[st]):
                node = base
                j = pos
                while j < jmax:
                    node = None if node is None else node.get(src[j])
                    j += 1
                    w = pend + src[pos:j]
                    # edge order below mirrors the old scan exactly (verbatim,
                    # nasal, junction, oov) so state-discovery order — and thus
                    # the k-best result — is byte-identical.
                    if node is not None and _TERMINAL in node:
                        push((j, ""), (sc + self._score(w), words + (w,)))
                    if nasal_variants and w[-1] in _NASAL_VARIANTS \
                            and j < n \
                            and src[j] not in _VOWELS and src[j] not in "MH'":
                        for fin in _NASAL_VARIANTS[w[-1]]:
                            alt = w[:-1] + fin
                            ws2 = self._score(alt)
                            if ws2 is not None:
                                push((j, ""), (sc + ws2, words + (alt,)))
                    if node is not None:
                        for res in res_at[j]:
                            for f, s in self.redges[res]:
                                w2 = w + f
                                # rules carry underlying finals (as/ur); lexicon
                                # and gold carry pausa forms (aH/uH) — try both,
                                # emit the attested one
                                for cand in (w2, pausa(w2)):
                                    ws2 = self._score(cand)
                                    if ws2 is None:
                                        continue
                                    push((j + len(res), s),
                                         (sc + ws2 - RULE_PENALTY,
                                          words + (cand,)))
                                    break
                    # once the span stops being a pada prefix, no verbatim or
                    # junction edge can fire further; only the nasal edge (which
                    # replaces the span's last char) can still match past here.
                    if node is None and not nasal_variants:
                        break
                if oov_penalty is not None:
                    # unknown-span edge: consume the maximal span src[pos:jmax]
                    # as a single OOV pada (fires once per item, as the old scan
                    # did outside the length loop with j at its final value).
                    w = pend + src[pos:jmax]
                    if w not in self.vocab:
                        push((jmax, ""), (sc + oov_penalty * len(w),
                                          words + (w,)))
        return [(sc, list(ws)) for sc, ws in best.get((n, ""), [])]

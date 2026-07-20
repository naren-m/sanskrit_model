# Improvement Plan — Segmentation +5 pts (jm-improve-5pt) — FINAL

**Goal:** improve segmentation exact match by ≥5 points over the 40.8% copy-constrained baseline.
**Outcome:** **87.8% exact / 93.4% F1** (351/400) vs **43.0%** same-checkpoint baseline — **+44.8 points**, re-derivability still 100% by construction on both tiers.
**Process:** literature survey + 2-round debate (sanskrit expert / developer / architect) + moderator prototyping; full record in `debate-notes.md`.

## What shipped

1. **`slm/segdp.py` — LexiconSegmenter** (tier 1): semi-Markov Viterbi over the surface with (a) verbatim edges = corpus padas scored by log unigram frequency, (b) junction edges inverting one non-identity sandhi rule, licensed only when both reconstructed sides are lexicon padas (pausa-normalized per 8.3.15), (c) optional penalized OOV-span edge, (d) lexicon hygiene (prune freq≤10 entries decomposing into ≥2 freq≥50 entries), (e) deterministic tie-breaks, (f) k-best output.
2. **`Inference.seg_dp`** (tiering): k-best DP paths rescored jointly (`dp_score + lm_mean_logprob × len`); fallback to `seg_constrained` when no DP path (3/400 rows).
3. **`train.py` fixes**: BUG-A (cosine horizon re-estimated from measured step rate — LR no longer parks at min after step 3000), BUG-B (best-val_bpb weights saved, not final), `--out` flag (smoke runs can't clobber the production ckpt). Retrained: best_bpb 0.4723 → **0.4635**; neural baseline 40.8% → 43.0%.
4. **`tests/test_segdp.py`**: re-derivability (incl. junction rows), OOV empty-result, k-best ordering, hygiene filter. 22 tests green repo-wide.
5. **Tuning protocol**: all hyperparameters frozen on a 500-row dev split (fresh windows, deduped against val); single final val run. Sensitivity curves flat around the chosen point.

## Measured evidence chain (chronological)

| configuration | exact | ckpt |
|---|---:|---|
| constrained neural greedy | 40.8% | old |
| beam-4 | 41.8% | old |
| pure lexicon-unigram DP | 73.8% | — |
| sandhi-aware lexicon DP (untuned) | 84.8% | — |
| + k-best LM joint rescore | 89.8% | old |
| + pausa junction fix (shipped, untuned) | 90.0% | old |
| constrained neural greedy | **43.0%** | new |
| tiered seg_dp, dev-frozen config | **87.8%** | new |

Diagnostics: gold-in-kbest@8 recall 94.5% (tier-1 ceiling); ambiguous-subset ablation: LM rescoring +22 rows over DP-argmax (n=337) — the neural term genuinely disambiguates. OOV stress (gold padas banned): DP 0% → the neural tier is the generalization floor, not decoration.

**Honest label (debate-mandated):** 87.8% is a closed-corpus, in-lexicon number — the 85K lexicon and the val set derive from the same 33,128 Rāmāyaṇa slokas. It is the *product* number for in-corpus text, not an open-domain generalization claim.

## Backlog (final round-2 votes; architect's hard phase boundary adopted)

**Post-vote experiment:** nasal-variant candidate edges (expert's & architect's joint #1) were implemented, gained +5/500 on dev, but **failed to replicate on val** (−13/400, recall unchanged — variants displace gold in k-best instead of supplying missing spellings). Kept in code behind `nasal_variants=False`; full record in debate-notes.md. The always-LM-vote fix (architect's live-bug find) is kept — it is a no-op until >1 candidate exists.

### Phase 1 — frozen-400 metric (small, lexicon/DP-side)
1. **Batch LM rescoring** (`logprob_batch`): ~2,500 sequential forwards → padded batches; 25 s → ~4 s. (developer #3)
2. **Rescorer headroom**: rows with gold in k-best but mis-ranked — pada-bigram prior in the DP score, or morphological-agreement signal (expert), only if misses persist after variants. (architect #5)
3. **Over-segmentation cluster** (19 residual): revisit prune aggressiveness vs RULE_PENALTY jointly on dev; architect's widen-not-switch rule (low joint margin → k 8→24 + re-rescore, never tier-switch on in-lexicon rows).
3b. **m/M cluster (15 residual), second attempt**: global variant edges rejected (see above); retry as *targeted* injection — variants only on low-margin rows during the widen step, dev-gated with a stricter promotion bar (≥2 pp on dev before any val run).
4. **Perf hardening for full slokas**: trie/Aho-Corasick over rule results; MAX_WORD sweep. (developer #4)

### Phase 2 — OOV/deployment metric (build the metric FIRST)
1. **OOV-injected dev set** (architect blocker): held-out slokas + deliberately unseen padas; calibrates the coverage tier-gate; yields the honest open-vocab number. Nothing else in phase 2 proceeds without it.
2. **DCS ingestion (CC BY 4.0, ~650K sentences)**: out-of-domain lexicon (shrinks OOV) + contamination-free genre-diverse eval; later forward-sandhi augmentation. Expert's caveat: bigger lexicon ⇒ more accidental both-sides-valid licensing (vibhakti homophony) ⇒ LM rescoring must strengthen, not retire.
3. **Neural-tier training scale** (seg 8k→50k, 20-min budget, task-weighted sampling, dedup vs val by source): raises the fallback floor. BUG-A/B already fixed as prerequisite. Expert's hard-junction curriculum folds in here.
4. **Tier-2 hardening**: propose-then-verify on fallback outputs; bidirectional pass (both rescoped by their own authors to fallback-only).
5. **Rule-chain composition in junction edges**: expert's diagnosis that some "no-path" rows are compositional-rule gaps (chained sandhi), not vocabulary gaps — distinguish before spending lexicon budget on them.
6. **Standing eval hygiene**: always report in-lexicon and lexicon-ablated numbers side by side. (expert #4, developer #2)

## Paper impact

- Committed draft's decoder numbers (40.8/47.9 projections) were measured on the old checkpoint, which was overwritten during this session (see incident log in debate-notes.md) — re-verify every model-dependent figure against the new ckpt before submission.
- Thesis sharpens, does not break: "capacity isn't the problem; architecture is" now has a third act — restoring the symbolic layer *as a lexicon-anchored lattice* (instead of the failed rule-only lattice) recovers what both pure arms miss; the LM's remaining role is exactly the idiosyncrasy frequency can't decide (+22 ambiguous rows). Fits TransLIST/MeCab/Krishna-EBM lineage (see debate-notes literature table).

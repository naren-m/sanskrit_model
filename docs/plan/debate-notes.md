# Debate Notes — "+5 points exact match" (jm-improve-5pt)

**Date:** 2026-07-18
**Participants:** Sanskrit Expert · Developer · Architect (moderated)
**Baseline:** copy-constrained greedy decode, 40.8% exact (163/400), F1 57.9%.
**Target:** ≥45.8% exact on the same fixed 400-row val set, 100% re-derivability preserved.

## Shared brief (facts on the table)

- 10.7M char GPT (6L/6H/384d, SLP1, block 256), multi-task mixture, 5-min MPS training budget; val_bpb still falling at cutoff → **undertrained**.
- seg supervision: only **7,600** synthetic windows (2–4 padas, ≤60 chars) vs corpus of **33,128 slokas / 308K pada tokens** → ~10× data headroom.
- Failure decomposition (A5): clean rows 47.9% exact, hard rows 24.2% — gap is **boundary knowledge on hard rows**, not decode search (τ=0 optimal; beam-4 only +1.0 at 10× cost).
- Lattice-rank path dead (6.8% ceiling). Lexicon of 85,715 padas + frequencies available at decode time.
- Constraint: val.json's 400 seg rows are frozen; any new training data must dedup against their `src` strings. Re-derivability guarantee must survive every change.

## Moderator diagnostics (measured before round 1 closed)

1. Baseline reconfirmed: constrained-greedy 163/400 (40.8%), F1 57.9%, 24 s for the full 400-row eval.
2. **100% of val gold padas (1,157/1,157) are present in the 85,715-pada corpus frequency lexicon**; all 400 rows fully in-lexicon.
3. Val pada-count distribution: 2-pada ×157, 3-pada ×129, 4-pada ×114 (matches training window range 2–4).
4. **Verbatim-copy ceiling: 312/400 (78.0%).** Any decoder that replays the surface verbatim (incl. the current constrained decoder) can never exact-match the 88 rows where a sandhi transform rewrote a junction — gold there restores underlying forms.
5. **Pure lexicon-unigram Viterbi DP** (segment src into lexicon words maximizing Σ log unigram-freq, no neural model): **295/400 = 73.8% exact** — +33.0 points over the neural constrained decoder, 94.6% of the verbatim ceiling. Runtime negligible.
   - Caveat for debate: lexicon is built from the same Ramayana corpus the val windows were sampled from. Not label leakage (a deployment lexicon would exist for in-domain text), but out-of-domain generalization would see OOV padas. Hybrid with the LM should degrade gracefully.

6. **Sandhi-aware lexicon DP** (add junction edges that invert one of the 1,042 non-identity rules, licensed only when the reconstructed left word `…+first` and right word `second+…` are both in the lexicon; small per-rule penalty): **339/400 = 84.8% exact, 2 s total**. Verbatim rows 294/312, transform rows 45/88 — the verbatim ceiling is broken. Re-derivability holds by construction (every transform edge is a licensed forward rule; every verbatim edge concatenates). This resurrects the "dead" lattice idea by anchoring rule inversion to the lexicon, eliminating the 11.4× over-licensing that killed it.

7. **Miss anatomy of the 61 DP errors:** transform|same-count 33 (dominated by pada-final `m`↔`M` pausa-form choice, where corpus gold is inconsistent per word — needs the LM's corpus-specific spelling knowledge), verbatim|under 10 + transform|under 10 (noisy lexicon contains multi-word spans as single "padas", e.g. `vAkyamabravIt`, crowding out splits), verbatim|over 3, verbatim|same 2, no-path 3. → Obvious hybrid: k-best DP paths rescored by the 10.7M LM (`Inference.logprob` already exists).

8. **k-best DP + LM joint rescoring:** k=8 paths per row (316/400 rows have >1 candidate); rescoring candidates by `dp_score + lm_mean_logprob × len(tgt)` gives **359/400 = 89.8% exact** (25 s incl. LM passes). LM-only rescoring *hurts* (81.5% vs DP's 84.8%) — the lexicon prior carries real signal the LM alone lacks; the joint score wins. Neural contribution over pure DP: **+5.0 pts**.
9. **OOV stress test:** banning each row's gold padas from the lexicon (simulating unseen text) collapses DP to **0% exact, 323/400 no-path**. The lexicon path is brittle out-of-domain; the neural constrained decoder (40.8% with no lexicon at all) is the necessary robustness fallback. → Tiered product: joint DP+LM when a path exists, constrained neural decode otherwise.

## Round 1 — independent proposals

### Sanskrit Expert

Opened with corpus evidence: only ~15 SLP1 chars ever occur pada-finally (a 23.6%, H 15.8%, m 15.7%, A 10.3%, M 8.1%, …) — Pāṇini's padānta restriction empirically. Boundary types: VC 48.0% (nearly always zero-change; only 5 narrow tuk-āgama rules), CC 39.7%, CV 6.9%, VV 5.5%; within consonant-final junctures, 88.4% are m/H/M (anusvāra assimilation + visarga sandhi). `build_sandhi_coverage` treats all 1,481 rules uniformly (per_rule=2–3) despite this skew.

1. **Pada-final legality + lexicon prior as decode-time rescoring** (no retrain): blend logit(space) with log-freq of char-as-pada-final + lexicon membership of resulting pada; avagraha `'` in surface = near-100% boundary signal, hard prior. Soft prior only (m/n can be medial).
2. **Hard-junction oversampling curriculum in build_seg**: stratify windows to 70–80% containing ≥1 CC/CV boundary (corpus rate ~47%) — reallocates signal to exactly the hard-row deficit. Top pick for raw points. Dedup vs val required.
3. **Frequency-matched per-rule weighting in sandhi coverage**: visarga/anusvāra families 10–20× rare vowel rules (88% of hard-boundary mass); keep ac-sandhi nonzero.
4. Pada-final-legality auxiliary task (only if #1 saturates; task-interference risk at 10.7M).
5. (For architect) allocate search budget to CC/CV positions only; VC is unambiguous — beam-4's +1.0 is diluted on positions that need no search.
Flag: verify `engine.join()` has no underlying-vs-surface (as/H) form mismatch at H-final junctures — 34.8% of hard-boundary mass.

### Developer

Found two load-bearing bugs + one negative result:
- **BUG-A**: train.py cosine schedule hardcodes `est_total=3000`; budget runs ~3,800 steps → LR parks at min_lr from step 3000. "Train longer" without fixing this silently learns nothing after 3k steps.
- **BUG-B**: train.py saves the FINAL model, not the tracked `best_bpb` state — free ~0.5–1 pt.
- Negative: val seg distribution (2–4 padas, ≤60 chars) matches train — window-length/block expansion is a non-lever for this benchmark.
- Verified: uniform sampling makes seg only ~1/5 of the mixture.

1. **Scale seg 8k→50–60k + fix BUG-A + ~20-min budget**: +4 to +8 expected; #1 risk in whole debate = **val contamination** — must dedup new windows against the 400 val surface strings and report the count.
2. **Task-weighted sampling** (seg ~40–50% of batches, ~10 lines): +2 to +4, overlaps #1.
3. **Lexicon-frequency rescoring of top-k copy-constrained candidates** (no retrain): model_logprob + λ·Σ log freq(pada) − μ·#OOV-padas; the 6.8% lattice ceiling doesn't apply to the copy-constrained pool. Tune λ/μ off-val.
4. **Save best-bpb checkpoint** (~5 lines, near-free).
5. Boundary-adjacent loss upweighting (3–5× on space/pipe positions): +1 to +3, riskiest, test separately.

### Architect

Key structural finding: seg_constrained makes isolated local argmax per gap — no global objective, no lexicon at decode time, left-context only — while a lexicon+freq ranker already exists in pipeline.py, unwired.

1. **Semi-Markov DP over gap scores + pada lexicon** (Viterbi over gap logits s_i = lp[space]−lp[copy] + Σ log P_lex(pada)): the single change converting local greed into global segmentation; ~40 lines; DP subsumes τ. Worst case ties greedy.
2. **Bidirectional gap evidence**: second pass on char-reversed input → right-context boundary logit; attacks hard rows (left-only LM commits where right context would veto). 2× inference vs beam's 10×. Risk: reversed input off-distribution — gate to near-threshold gaps.
3. **Propose-then-verify**: re-split OOV padas with lowered boundary bias; accept only if lexicon coverage strictly improves.
4. **10× seg data + longer budget**: highest ceiling, least cheap; force-multiplier for #1–#3. Verify hard-row lift specifically.
5. Boundary-tagger head on frozen backbone (distill DP into 1-pass tagger) — productionization move, last.
All five provably preserve re-derivability (gap-subset selection can't emit new chars).

## Literature scan (similar projects) — researcher

| project | approach | data | numbers | transferable |
|---|---|---|---|---|
| Hellwig & Nehrdich 2018 rcNN-SS | char CNN+RNN per-char boundary tag | DCS ~560K sent. | SIGHUM 96.84 F1 / 87.08 PM | per-char tag = our exact task; CNN junction window |
| ByT5-Sanskrit 2024 | 582M byte T5, pretrain 6.5B tokens → multitask fine-tune | 601K DCS | DCS2018 90.11 PM; Hackathon 94.29 PM | LM pretraining scale is the biggest driver; pseudo-paragraph context |
| Krishna et al. 2018 EBM | energy-based global scoring over SHR lattice | <1/10 task data | 96.92 F1 | whole-sentence global score; re-rank candidates |
| Aralikatte 2018 seq2(seq)² | split-location decoder (95%) + word decoder (79.5%) | UoH | +20% over prior | decouple "where" from "what" |
| SHR (Goyal & Huet) | symbolic FST lattice | lexicon | 59.9 SPA alone | pada trie as boundary gate; powered EBM/TransLIST |
| TransLIST 2022 | char labeling + lexicon candidates via soft-masked attention + path ranking | SIGHUM/Hackathon | 98.86 F1 / 93.97 PM (SOTA) | **lexicon injection + path re-ranking without abandoning char decoder** |
| CWS small-model tricks | bigram feats, CRF/Viterbi, pointwise | — | — | transition layer / global decode over boundary tags |
| DCS corpus | 650K tagged sentences, **CC BY 4.0** | github OliverHellwig/sanskrit | — | 20× training data; forward-sandhi augmentation |

Researcher's ranked transferables: (1) forward-sandhi augmentation from DCS; (2) char-LM pretraining on raw DCS/GRETIL; (3) TransLIST-style lexicon injection; (4) CRF/Viterbi global boundary decode; (5) beam re-ranking by pada n-gram LM; (6) wider context; (7) junction bigram features; (8) two-stage where/what decode.

**Convergence note (moderator):** researcher's #3/#4/#5 (lexicon injection + global Viterbi + path re-ranking = TransLIST recipe), architect's #1 (semi-Markov DP + lexicon), sanskrit-expert's #1 (lexicon/legality prior at decode), and developer's #3 (lexicon rescoring of top-k) are four independent arrivals at the same idea — which the moderator prototype already measured at **84.8–89.8%**. Strong mutual validation.

## Incident log (moderator)

While landing the developer's train.py fixes (BUG-A schedule, BUG-B best-ckpt), a 3-step smoke run **overwrote `data/ckpt.pt`** (gitignored, no backup, no local snapshots). Repair: retrained with the same seed/data under the fixed schedule; a `--out` flag was added so smoke tests can never target the production checkpoint again. Consequence: the retrained model is not bit-identical to the one behind the committed paper numbers (40.8% etc.) — all before/after comparisons below are re-measured on the new checkpoint, and the paper's decoder numbers need re-verification against it before any future submission.

## Round 1.5 — revised positions after the DP evidence

All three teammates revised once the moderator's DP numbers (84.8/89.8/90.0, OOV 0%) landed:

- **Architect** reranked: (1) ship k-best DP+LM as tier-1; (2) constrained-neural fallback is "the single most important architectural piece" — the only thing between 89.8% and the OOV cliff; (3) **OOV-injected dev set is BLOCKER-level** — current val (100% in-lexicon) structurally cannot observe the failure mode the tiered design exists for, so the router is untestable on it; (4) 10× seg data now serves the *fallback floor*, not the DP; (5) tagger head last. Also: ablate DP-argmax vs DP+LM on the ambiguous subset only ("make sure the LM term genuinely disambiguates, doesn't launder lexicon confidence"), and measure **gold-in-kbest recall** — the tier-1 analog of the old lattice ceiling. Routing rule: hard symbolic gate (empty lattice / zero-edge pada → tier-2) first, soft top1–top2 margin second, fallback-on-doubt (tier-2 degrades gracefully, tier-1 fails catastrophically).
- **Developer** reframed: "+5 goal met ~10× over; improvement now = hardening + honest measurement, not chasing the neural number." New top item: **partial-coverage OOV edge in the DP** (consume an unknown span as one penalized OOV pada so one unseen word no longer dumps the whole sloka to the fallback — routes per-span, not per-sentence). Flagged: k-best per-state pruning can drop a globally-best path; tie-breaking by insertion order is nondeterministic across runs; `startswith` scan over 792 result strings is O(n·MAX_WORD·|results|) — trie it before scaling to full slokas; MAX_WORD=30 and RULE_PENALTY=2.0 are magic constants to sweep on a dev split. **Contamination audit is a blocker for how the number is labeled**: the lexicon and val share a generative process, so 90.0% is a closed-corpus product number, not a generalization claim.
- **Sanskrit expert** delivered the deepest critique of the DP: (a) 0%-under-ban proves it does *exact-match retrieval, not morphological decomposition* — and Sanskrit's defining property is unbounded productive compounding (samāsa) + taddhita derivation, which no finite lexicon closes; (b) **pada-final homophony** (vibhakti ending reuse + sandhi collapsing distinct underlying finals) means "both sides in lexicon" can license a lexically-valid but grammatically wrong split — real disambiguation uses syntactic agreement, not frequency; (c) optional (vibhāṣā) sandhi + metrical variants mean one canonical rule table will misfire on texts following different conventions; (d) the 100%-in-lexicon fact is largely circular (val generated from the same slokas that built the lexicon); (e) pausa-normalization assumes every editorial word division is a true pausa juncture — edition-dependent. Verdict: "treat 84.8/89.8 as an in-domain ceiling under a closed-vocabulary assumption, not a production estimate." If scope = closed corpus, DP-primary is right; if generalization matters, the fallback's quality on DP's blind spot is the real lever.
- **Researcher** (focused follow-up): the hybrid is not an anomaly — it re-discovers the dominant paradigm: **MeCab** (lexicon lattice + Viterbi + CRF scores, 20 years of Japanese production), **semi-Markov CRF** (the formal name for span-scored segmentation), **Krishna et al. EBM** (SHR lattice + global scoring, 96.92 F1 with <1/10 data — closest published analog), **TransLIST** (lexicon candidates + path re-ranking, SIGHUM SOTA). ByT5 shows pure-neural needs ~50× the params + 6.5B-token pretraining to match what the hybrid gets for free at this scale. DCS (650K sentences, CC BY 4.0) is the key external input — as an *out-of-domain lexicon + contamination-free eval*, more than as pretraining data.

## Round 2 — rebuttals & votes

### Developer (full round 2)
- **Retrain is no longer the next investment.** Ranked next points: (1) lexicon hygiene — min_count + drop entries that are themselves DP-segmentable into ≥2 known padas (attacks the 20 under-segmentation misses, est +2–4, zero GPU); (2) per-word pausa (m/M) attested-form lookup — deterministic corpus lookup beats a better LM tie-breaker for the 33-miss cluster (est +2–3); (3) retrain drops to insurance for the OOV tier — but still fix BUG-A/B (~10 lines) since any future retrain is silently broken without them.
- **Overfit protocol:** carve a dev split from slokas *not* among the val sources; tune RULE_PENALTY/k/min_count/λ there; freeze; report once on val; publish sensitivity curves (flat plateau ⇒ number is real). Reassurance in hand: LM-only rescoring *hurts* (81.5 < 84.8) — the win comes from the DP prior, not a fitted weight, "exactly the low-overfit signature." Measure gold-in-kbest@8: recall ≈ 90% ⇒ k-limited; recall ≫ 90% ⇒ headroom is in the rescorer.
- **Perf:** batch the ~2,500 sequential logprob forwards (`logprob_batch`) → 25 s → ~4 s.
- Rebuttals: endorse avagraha-as-forced-edge (cheap, near-lossless); endorse m/M diagnosis but rebut curriculum/prior remedies (per-word lookup instead); demote expert's oversampling + rule-coverage weighting to the OOV-neural tier; rebut architect's bidirectional pass (DP already has global bidirectional context; 2× cost for fallback-only value); propose-then-verify subsumed by hygiene + OOV edge; rebut "pretraining scale dominates" — "we just watched a zero-pretraining symbolic DP beat the neural model by +44."
- **Vote:** 1. lexicon hygiene · 2. clean tuning protocol + honest in-vocab/OOV reporting · 3. logprob batching · 4. m/M attested-form lookup + avagraha edge · 5. DCS out-of-domain lexicon + clean eval (with BUG-A/B as prerequisite).

## Implementation session (moderator, post-round-2)

Consensus items landed:
- `slm/segdp.py`: deterministic tie-break (score desc, then text — developer's reproducibility flag); optional per-span **OOV edge** (penalized unknown-span consumption, developer's #1 — off by default, for open-vocab use); **lexicon hygiene** `prune_spans` (drop freq≤max entries that decompose verbatim into ≥2 freq≥min entries).
- `train.py`: BUG-A fixed (schedule re-estimates total steps from measured rate at step 100); BUG-B fixed (best-val_bpb weights snapshotted and saved); `--out` flag added after the checkpoint-overwrite incident.
- **Dev-split tuning protocol** (developer's): 500 fresh windows, new seed, deduped against all 400 val `src` strings. Sweeps (DP-1best exact on dev): RULE_PENALTY {0.5: 393, **1.0: 395**, 1.5: 390, 2: 390, 3: 387, 4: 386}; min_count {**1: 390**, 2: 210, 3: 155} — hapax padas are essential, min_count stays 1; prune_spans {(none): 395, (5,50): 400, (10,50): **403**, (8,30): 403} at penalty 1.0. Flat plateau ⇒ low overfit risk. Frozen: penalty 1.0, prune (10,50), k=8, min_count 1.
- Retrain (after checkpoint incident): best_bpb **0.4635** vs original run's 0.4723 — the BUG-A/B fixes alone improved the LM.

### Architect (full round 2)
- **The repricing:** every neural-training proposal was priced against the 40.8% baseline "that is no longer the product" — on the frozen 400 they can move at most the 3 fallback rows. Value migrated to the OOV regime; they're follow-up items. Exceptions credited: BUG-A/B (correctness regardless), expert's m/M diagnosis ("GOLD — a tier-1 candidate-generation gap, not a training problem").
- **Next bottleneck, mechanically split:** (1) under-seg misses = k-best RECALL failures (noisy lexicon span outscores the split; LM can't fix what DP never proposed) → lexicon hygiene #1; (2) m/M misses = CANDIDATE-GENERATION failures (`pausa()` covers s/r→H but never emits the m↔M variant, so the LM is never offered the right spelling) → nasal-variant generation, ~15 lines.
- **Routing correction:** never route low-margin in-lexicon rows to the neural tier (strictly worse oracle there, negative expected value). Tier gate stays coverage-only; low joint margin triggers WIDENING (k 8→24 + variant injection + re-rescore), not a tier switch.
- **Live bug found:** `seg_dp`'s `len(cands)==1` branch skips LM rescoring — a single full-cover path can still be the wrong m/M spelling; LM must always vote once variants exist.
- Retired his own round-1 bidirectional pass for tier-1 ("the DP is already a global bidirectional optimization") — survives only as tier-2 polish. Propose-then-verify: subsumed in tier-1 by static hygiene; genuinely useful on tier-2 outputs (not lexicon-verified at all).
- **Vote:** 1. ship tiered seg_dp (done) · 2. lexicon hygiene (verify no fallback-count regression) · 3. nasal-variant candidate-gen + always-LM · 4. BUG-A/B · 5. pada-bigram in DP only if 2–3 leave misses.
- **Hard phase boundary:** OOV work (DCS, retrain, curriculum, tier-2 hardening) is a SEPARATE plan with its own metric — "refusing to let the OOV agenda hide inside a benchmark that can't see it."

### Sanskrit expert (full round 2)
- Withdrew his own round-1 legality-prior/auxiliary-task items ("the shipped lexicon DP is a strictly better instantiation of the same idea").
- **m/M/homorganic-nasal is vibhāṣā, not noise:** Pāṇini licenses THREE surface realizations of pada-final -m (unchanged m; anusvāra M per 8.3.23; homorganic class nasal per 8.4.58) — genuinely optional, which is *why* gold is inconsistent per word. Generate all licensed variants (homorganic nasal only before class stops) and let the LM disambiguate — direct extension of the shipped H fix. His #1.
- Compound-decomposition hygiene filter = his round-1 "future-proofing" item, now the direct fix for the second error bucket.
- **Junction-edge limits on real text:** (a) single-rule-per-edge can't represent chained rule applications — those become no-path rows that *look* like OOV but are compositional-rule gaps (different fix: rule-chain composition); (b) samāsa-internal junctures aren't covered by "both sides lexicon-valid" (compound-member allomorphs differ from citation forms) — scope boundary, not a bug today; (c) **bigger lexicons raise accidental-licensing risk** (dense vibhakti homophony ⇒ more coincidental both-sides-valid wrong splits) — LM rescoring must *strengthen* as the lexicon grows, not be retired.
- DCS: right source precisely for genre/source diversity; imports more realization variance (m/M conventions vary by scribal tradition) — a feature for OOV robustness, but demands stricter eval hygiene.
- **Vote:** 1. m/M/nasal 3-way candidate-gen · 2. compound-decomposition hygiene · 3. BUG-A/B then DCS re-targeted at diversity · 4. standing in-lexicon + lexicon-ablated dual reporting · 5. architect's mechanisms rescoped to fallback path only.

## Moderator synthesis

**Points of consensus (all three + literature):**
1. Ship the tiered segmenter: lexicon-anchored sandhi-aware Viterbi DP (tier 1) + LM k-best joint rescoring + copy-constrained neural fallback (tier 2). Four independent proposals converged on this shape before the prototype numbers were shared; MeCab/Krishna-EBM/TransLIST are the same paradigm in production/literature.
2. Re-derivability guarantee survives every accepted change — both tiers only insert boundaries (plus rule-licensed junction restorations).
3. **Honest reporting is mandatory**: the headline is a *closed-corpus, in-lexicon* number (val and lexicon share a generative process — sanskrit-expert's circularity point, developer's contamination audit, architect's "the val set cannot observe the OOV failure mode"). Report in-lexicon exact, OOV behavior, and traffic split separately.
4. Neural training investment (data scale, schedule fixes) is re-scoped from "the headline lever" to "the OOV/fallback floor."

**Points of genuine disagreement (resolved by moderator):**
- *Bidirectional reversed-pass* (architect) vs *"DP already has global context"* (developer): developer wins for tier-1; idea parked for the fallback tier only.
- *Decode-time legality prior / curricula* (sanskrit-expert R1) vs *deterministic per-word lookup* (developer R2): for the m/M miss cluster, the deterministic attested-form direction is cheaper; but the LM joint rescore already captures much of it empirically — measured, not assumed.
- *Retrain now* (developer R1) vs *retrain later* (developer R2, architect): later — except BUG-A/B fixes, which were landed immediately (and, forced by the checkpoint incident, validated: best_bpb improved 0.4723 → 0.4635).
- *Prune aggressiveness*: sanskrit-expert warned legitimate rare compounds die with aggressive pruning; dev sweep confirmed — (1,5) hurts (379), mild (10,50) helps (403). Expert's caution empirically vindicated.

**What the debate changed vs the moderator's solo prototype:** deterministic tie-break, per-span OOV edge, dev-split tuning protocol + sensitivity sweep (the 90.0% would otherwise have carried silent hyperparameter overfit), lexicon hygiene (+13 dev rows), BUG-A/B train fixes, the three-number honest-reporting frame, gold-in-kbest recall + ambiguous-subset ablation as required diagnostics, and the DCS roadmap for open-vocab work.

See `improvement-plan.md` for the final ranked plan and measured outcome.

## Final measured outcome (retrained checkpoint, dev-frozen config, single val run)

Checkpoint: retrained under fixed schedule + best-ckpt saving, best_bpb 0.4635. Config frozen on dev only (penalty 1.0, prune (10,50), k=8): no val peeking.

| arm | exact | note |
|---|---:|---|
| [1] constrained neural greedy (same ckpt) | **43.0%** (172/400) | old ckpt scored 40.8% — BUG-A/B fixes alone bought +2.2 |
| [2] DP-1best (tuned) | 82.2% (329/400) | |
| [3] **tiered seg_dp — headline** | **87.8%** (351/400), F1 93.4% | +44.8 pts over [1]; goal was +5 |
| [4] gold-in-kbest@8 recall | 94.5% (378/400) | tier-1 ceiling; 27 rows of rescorer headroom |
| [5] ambiguous-subset ablation (n=337) | DP-argmax 270 → DP+LM 292 | **LM delta +22** — architect's "confidence laundering" concern answered: the LM term genuinely disambiguates |

Fallback engaged on 3 rows. Both tiers re-derivable by construction. Honest-reporting frame: 87.8% is a **closed-corpus, in-lexicon** number; open-vocab floor is the neural tier (43.0% lexicon-free); OOV measurement requires the planned OOV dev set (plan §4).

Note on the earlier 90.0%: measured on the (now-overwritten) old checkpoint with the untuned a-priori config. After the checkpoint incident and the debate-mandated no-peeking protocol, the defensible number is 87.8% — selecting the old config *because* it scores higher on val would be exactly the overfitting round 2 prohibited.

## Post-vote experiment: nasal-variant candidate edges — REJECTED

Both round-2 #1 votes (expert: Pāṇini 8.3.23/8.4.58 3-way realization; architect: "candidate-generation gap, ~15 lines") were implemented: vocab-attested m/M/n variant edges before consonants, plus the always-LM-vote fix. Results:

| set | without variants | with variants |
|---|---:|---:|
| dev tiered (500) | 417 | 422 (+5) |
| val tiered (400) | **351 (87.8%)** | 338 (84.5%, −13) |
| val DP-1best | 82.2% | 78.5% |
| val gold-in-k8 recall | 94.5% | 94.5% (unchanged) |

The dev gain did not replicate. Mechanism: recall didn't move — the variants weren't adding missing gold spellings on val; they injected confusable candidates that displaced gold within k-best and degraded both the DP argmax and the LM's pick rate. Feature kept in code behind `nasal_variants=False` (off by default). Honesty note: this rejection consumed one bit of val adaptivity (the feature was evaluated on val once and declined for non-replication); the 87.8% headline config itself was frozen before the feature existed. Follow-up for the m/M cluster moves to phase-1 backlog (widen-k + targeted variant injection only on low-margin rows, per architect's widen-not-switch rule — evaluate on dev with a stricter promotion bar first).

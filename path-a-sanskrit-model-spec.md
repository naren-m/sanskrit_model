# Path A — Sanskrit Specialist Model Training Spec

### Companion to the Prakriyā Engine design: the neural proposer/scorer

**Role in the system.** This model fills stages 5–6 of the Prakriyā Engine (morphological hypothesis ranking + lattice disambiguation) and provides the sandhi-split proposer. It never certifies an analysis — every candidate it emits is round-tripped through vidyut-prakriya (Layer-A discipline). Neural proposes, Pāṇini disposes.

---

## 1. Model

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | Encoder-decoder, ByT5-style (T5 v1.1 architecture, byte/char vocab) | Seq2seq fits all four tasks; encoder gives bidirectional context for scoring |
| Size | Start **~120M** (byt5-small class); scale to ~300M only if evals demand | Trains on one 24–80GB GPU; fast iteration beats size at this data scale |
| Tokenization | **Character-level over SLP1** (vocab ≈ 100: SLP1 alphabet + digits + punctuation + task/control tokens) | Sandhi and morphology are character phenomena; SLP1 is 1 char = 1 phoneme, no subword pathology. Skip raw UTF-8 bytes — Devanagari is 3 bytes/char, wasteful |
| Context length | 512 chars train / 1024 eval (RoPE or T5 relative positions extrapolate) | A śloka ≈ 130–180 SLP1 chars; 512 covers verse + analysis output |
| Init | From scratch. (Optional ablation: init from `google/byt5-small` — its byte embeddings are useless for SLP1, likely no benefit) | Vocab mismatch makes transfer marginal |
| Framework | HF `transformers` + `accelerate` (or plain PyTorch + Lightning); bf16; no apex/DeepSpeed needed at this size | Replaces the THUDM/GLM stack entirely |

Control tokens (single reserved chars/ids): `<seg>`, `<morph>`, `<sandhi>`, `<denoise>`, task separators, `[MASK_k]` sentinel ids (T5-style), and feature-field markers (`<dhAtu>`, `<lakAra>`, `<viBakti>`, `<liNga>`, `<vacana>`, `<kft>`, `<taddhita>`).

---

## 2. Data builders (the real work)

All builders emit JSONL: `{"task": ..., "src": ..., "tgt": ..., "provenance": ...}` in SLP1. Everything is verifiable or gold — no scraped silver labels in v1.

### 2.1 `build_vidyut_pairs.py` — synthetic form ⇄ analysis (the unfair advantage)

Forward-generate with vidyut-prakriya over:

- All ~2,000 dhātus × 10 lakāras × 3 puruṣas × 3 vacanas × kartari/karmaṇi → **tiṅantas** (~10⁶ forms with upasarga combinations sampled, not exhaustive)
- Kṛdantas: {kta, ktavatu, Satf, SAnac, tavya, anIyar, lyuw, GaY, ktvA, lyap, tumun} × dhātus → declined subantas
- Sanādi: ṇic, san, yaṅ over a high-frequency dhātu subset
- Subantas: prātipadikas from the kosha × 8 vibhaktis × 3 vacanas × liṅga

Target encoding example:

```
src: <morph> Bavati
tgt: <dhAtu> BU 01.0001 <gaRa> 1 <lakAra> law <puruza> praTama <vacana> eka <prayoga> kartari
```

Deduplicate by (surface, analysis). **Ambiguity is signal**: one surface with n analyses becomes n pairs; the model learns the distribution, top-k decoding surfaces alternatives.
Frequency-weight sampling by DCS lemma frequencies (uniform sampling over the Dhātupāṭha would drown the model in forms of roots attested twice in the corpus).
Expected yield: 5–20M pairs; sample ~10M into the mixture.

### 2.2 `build_sandhi_pairs.py` — segmentation data, two sources

- **Gold:** DCS (~4.8M tokens) gives sandhied text aligned to segmented+lemmatized analyses. Primary supervised source. Export: `src: <seg> <sandhied line>` → `tgt: pada1 | pada2 | ...`
- **Synthetic:** take segmented pada sequences (from DCS, or grammatical pada sequences sampled from vidyut output) and **apply sandhi forward** using `rules.csv` (first+second→result). Free unlimited (sandhied, split) pairs, and by construction every boundary is licensed by a known rule. Apply with probability <1.0 per junction so the model also sees unsandhied text.

### 2.3 `build_denoise.py` — raw-text span infilling (the GLM idea)

Corpora: GRETIL, SARIT, Muktabodha, sa.wikipedia, Vedabase (laukika only for v1; Vedic/accented text excluded — flagged-mode later, per the standing decision).
Pipeline: transliterate → SLP1, normalize daṇḍas/avagraha, dedup (MinHash), filter non-Sanskrit lines. Expect ~0.5–1.5 GB clean SLP1 text.
Objective: T5/GLM span corruption — mask spans (mean length 6 chars, ~15% of text) with sentinel ids; predict spans. Verse-aware variant: mask a whole pāda 10% of the time (`[sMASK]` analogue) — this is what later powers metrical restoration.

### 2.4 `build_meter_lines.py` — light metrical conditioning

Prefix denoising examples from verse corpora with the identified meter name (from the Phase-1 chandas module): `<Candas> anuzwuB <denoise> ...`. Cheap conditioning; the hard metrical guarantees still come from symbolic constrained decoding at inference, not from the model.

### 2.5 Explicitly out of v1

- Generated prakriyā traces (sūtra sequences) as a training *target* — traces come from the verifier only; a model that emits sūtra numbers will hallucinate them (source-accountability principle).
- Vedic/accented material.
- Manuscript-noise robustness (add as augmentation in v2: OCR-style character noise on denoising inputs).

---

## 3. Training mixture & schedule

**Stage 1 — Pretraining (denoising), ~1 epoch-equivalent over mixture:**

| Task | Share |
|---|---|
| 2.3 span denoising (raw corpus) | 60% |
| 2.2 synthetic sandhi pairs | 20% |
| 2.1 morph pairs | 20% |

**Stage 2 — Supervised multi-task finetune:**

| Task | Share |
|---|---|
| 2.2 gold DCS segmentation | 35% |
| 2.1 morph analysis (freq-weighted) | 35% |
| 2.2 synthetic sandhi | 15% |
| 2.3/2.4 denoising (retain, prevents forgetting) | 15% |

Hyperparameters (starting point, 120M): AdamW, lr 3e-4 → cosine to 3e-5, batch ≈ 0.5M chars, bf16, dropout 0.1, Stage 1 ~200k steps, Stage 2 ~50k steps with early stopping on dev segmentation F1. One A100-80GB ≈ 3–5 days total; a 4090 works with gradient accumulation at ~2–3× wall clock.

---

## 4. Inference integration (Prakriyā Engine stages 5–6)

- **Segmentation:** constrained beam search (k=16) over the sandhi lattice from stage 4 — the model may only emit splits that exist as lattice edges (i.e., licensed by rules.csv/lexicon). Model = ranker over symbolically valid paths, never a free generator.
- **Morphology:** for each candidate pada, decode top-k analyses; **filter: keep only analyses vidyut-prakriya regenerates to the observed surface**; renormalize scores over survivors. Verified survivors carry their rule trace into the display layer.
- **Scoring API:** also expose encoder log-likelihoods for whole-lattice path scoring (replaces/augments the trigram baseline in the engine spec — build the trigram anyway; it is the honesty baseline the neural model must beat).
- **ārṣa-prayoga case:** if zero analyses survive verification, report "no Pāṇinian derivation; nearest verified analyses: …" with the model's raw candidates clearly labeled unverified.

---

## 5. Evaluation suite (`evals/`)

Held out **by work, not by line** (DCS texts overlap heavily; leakage by line split would inflate scores):

1. **Segmentation:** word-level F1 + sentence exact-match on held-out DCS works; compare against published Hackathon-SIGHUM splits so numbers are comparable to prior sandhi-split literature.
2. **Dhātu identification accuracy** (top-1 / top-5): held-out vidyut pairs *and* a 500-token hand-checked slice of real verse (Gītā chapters excluded from training).
3. **Verification survival rate:** % of top-1 analyses that pass vidyut round-trip — the metric that matters most for the engine.
4. **Metrical restoration:** mask one pāda in held-out verses; measure exact restoration and %-metrically-valid under constrained decoding.
5. **Golden verses:** the engine's fixed test suite (BG 2.47, sahasranāma lines, one āryā, one vasantatilakā) end-to-end.

---

## 6. Repo layout

```
sanskrit-lm/
├── data/
│   ├── raw/            # gretil, dcs, corpora dumps
│   ├── builders/       # build_vidyut_pairs.py, build_sandhi_pairs.py,
│   │                   # build_denoise.py, build_meter_lines.py
│   └── mixtures/       # stage1.jsonl.zst, stage2.jsonl.zst
├── tokenizer/          # slp1_vocab.json + control tokens
├── train/              # HF training loop, configs (120m.yaml, 300m.yaml)
├── evals/
├── serve/              # scorer API consumed by prakriya-engine
└── tests/
```

## 7. Build order

1. Tokenizer + `build_vidyut_pairs.py` (unblocks everything; also independently useful as a lookup-table sanity check against vidyut-kosha)
2. `build_sandhi_pairs.py` (DCS export + rules.csv forward application)
3. Corpus cleaning + `build_denoise.py`
4. Stage 1 training → Stage 2 → eval loop
5. `serve/` scorer API + engine integration behind the trigram baseline A/B

**Go/no-go gate:** the model earns its place only if it beats the trigram baseline on verification-survival rate and segmentation F1. If it doesn't, the engine still works — the symbolic core never depended on it.

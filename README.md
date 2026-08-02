# sanskrit-lm — Path A, autoresearch harness

A character-level SLP1 **Sanskrit specialist model**, built as an
[autoresearch](https://github.com/karpathy/autoresearch)-style harness: a
single editable `train.py`, a fixed wall-clock budget, and one headline metric
(`val_bpb`). The guiding principle from the design spec
([`path-a-sanskrit-model-spec.md`](path-a-sanskrit-model-spec.md)) is:

> **Neural proposes, Pāṇini disposes.** Every analysis the model emits is
> round-tripped through the symbolic rule engine; the net is a *proposer/ranker*,
> never a certifier.

## Architecture note (why GPT, not ByT5 yet)

The spec targets an encoder-decoder **ByT5**. This harness starts with a
decoder-only **GPT** because it trains on Apple-Silicon MPS/CPU in minutes — you
can actually *play*. Every task is framed as one causal sequence
`<task> src <sep> tgt <eos>`, loss taken only on the `tgt` span, so a decoder LM
learns the same seq2seq maps. ByT5 is the scale-up target, earned once the
pipeline beats the symbolic baseline (spec §7 go/no-go gate). The verifier gate
is architecture-agnostic, so this costs nothing on correctness.

## The rules ("all the rules")

`slm/rules.py` is the symbolic core — pure stdlib, loaded from the SLP1 CSVs:

| Asset | Rows | Feeds |
|---|---|---|
| `sandhi-rules-full.csv` | ~1468 | `SandhiEngine.join` / `.split` |
| `dhatus-full.csv` / `dhatus-core.csv` | 2259 / 294 | `DhatuKosha.lookup` (morph + verify) |
| `meters-full.csv` | 145 | `ChandasEngine.identify` (chandas) |

## The corpus (gold text)

`slm/corpus.py` transliterates real Sanskrit → SLP1 and caches it:

- **Ramāyaṇa** (`../ramayanam`): 33k slokas → ~308k padas, word-segmented in
  original order → **gold segmentation** data.
- **Yoga Sūtras** (`../yoga_sutras`): 196 sutras, clean sandhied lines → denoising.

## Tasks in the mixture (`slm/datagen.py`)

| task | src → tgt | source |
|---|---|---|
| `morph` | `<morph>gam` → `<dhAtu>ga\mx~<gaRa>1<artha>...` | dhātupāṭha (deterministic) |
| `seg` | `<seg>`sandhied → `pada1 \| pada2` | Ramāyaṇa gold word order |
| `sandhi` | `<sandhi>a<sep>b` → joined | forward sandhi over real vocab |
| `meter` | `<Candas><wt>`L/G-pattern → `<meter>`name | chandas table |
| `denoise` | `<denoise>`span-masked → spans | Yoga Sūtras + joined ramayana |

## Run it

```bash
uv sync                       # torch (MPS) + indic-transliteration
uv run python -m slm.corpus   # build the SLP1 corpus cache (once)
uv run prepare.py             # tokenizer + mixture + train/val split
uv run train.py               # ~5 min budget; prints val_bpb + decode demos
uv run python -m evals.eval   # seg exact-match, dhatu top-1, verify-survival
```

Quick smoke: `uv run prepare.py --quick && uv run train.py --budget-min 1`.

## Demo

```bash
uv run demo.py            # scripted showcase (morph / sandhi / seg / chandas)
uv run demo.py --repl     # interactive: morph gam | sandhi rAma asti | seg rAmo'sti
uv run serve.py           # web UI at http://127.0.0.1:8008
```

The demo runs the loop live: the net proposes an analysis, the symbolic engine
confirms it against the real Dhātupāṭha (or **rejects a hallucination** — try
`morph zzzq`).

## Results (5-min MPS run, 10.7M-param GPT)

| task | metric | value |
|---|---|---|
| morph | **verify-survival** (proposals Pāṇini confirms) | **96.4%** |
| seg | exact-match on held-out Ramāyaṇa | 38.5% |
| sandhi | exact-match (in-distribution) | 91–96% |
| — | `val_bpb` | ~0.46 |

Classic sandhi cases all correct: `rAma+asti→rAmAsti`, `tat+hitam→tadDitam`,
`deva+indra→devendra`, `gaNgA+udakam→gaNgodakam`.

## Tests

```bash
uv run pytest                      # everything (dev group installs pytest)
uv run python -m slm.rules         # symbolic-engine self-test, real verses
```

Meter identification is graded against `tests/data/golden_meters.json` — 20
attested verses (Rāmāyaṇa, Gītā, Kālidāsa, Bhartṛhari, Śaṅkara, Sāṃkhyakārikā)
covering anuṣṭubh pathyā/vipulā, 8 vṛttas, upajāti and āryā. Ground truth is
derived from Piṅgala's *gaṇa* definitions, deliberately **not** from
`meters-full.csv`, so a bad row in the table shows up as a failure instead of
being silently confirmed.

An optional second opinion cross-checks the whole golden set against the
independent MIT-licensed [`sanskrit/chandas`](https://github.com/sanskrit/chandas):

```bash
git clone --depth 1 https://github.com/sanskrit/chandas /tmp/chandas
SANSKRIT_CHANDAS_PATH=/tmp/chandas uv run pytest tests/test_chandas_crossvalidate.py
```

It is not a dependency (no `setup.py`, Python 2 `__init__`), so those tests
skip unless the env var is set. The two known divergences are asserted with
their reasons rather than tolerated — see the module docstring.

## Files

```
slm/tokenizer.py   SLP1 char tokenizer (~120 ids) + control tokens
slm/model.py       nanoGPT-style decoder-only GPT (MPS/CPU)
slm/rules.py       symbolic engine: sandhi / dhatu / chandas
slm/corpus.py      Devanagari->SLP1 corpus loaders (ramayana, yoga sutras)
slm/datagen.py     multi-task mixture builder
slm/infer.py       structured inference API (model + verifier)
prepare.py         data + tokenizer stage (autoresearch prepare.py)
train.py           THE editable trainer (autoresearch train.py)
demo.py            CLI showcase / REPL
serve.py           local web UI (stdlib http.server)
evals/eval.py      task metrics + verification-survival
tests/data/golden_meters.json          20 attested verses + their meters
tests/test_chandas_golden.py           meter identification, graded
tests/test_chandas_crossvalidate.py    optional 2nd opinion (sanskrit/chandas)
program.md         autonomous-research instructions for a coding agent
```

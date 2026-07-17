# program.md — Autonomous Research Org (Sanskrit LM)

_Adapted from karpathy/autoresearch. You are a coding agent running training
experiments to improve a character-level SLP1 Sanskrit specialist model._

## The objective

Minimize **`val_bpb`** (bits/token over target spans, printed by `train.py`)
**within a fixed wall-clock budget** (default 5 min on this Mac), WITHOUT
regressing the per-task decode demos. `val_bpb` is the fast proxy; the metrics
that actually matter for the engine (Path A spec §5) are:

1. **Segmentation** exact-match on `seg` val examples.
2. **Dhātu-id** top-1 on `morph` val examples.
3. **Verification-survival** — % of decoded `morph` analyses that the symbolic
   engine (`slm/rules.py`) can regenerate. *Neural proposes, Pāṇini disposes.*

The go/no-go gate (spec §7): the model earns its place only if it beats a
trigram/rule baseline on segmentation + verification-survival. If it can't, the
symbolic core in `slm/rules.py` still works — it never depended on the net.

## The rules of the game

- **Edit only `train.py`.** Everything else (model.py, tokenizer, datagen,
  rules engine, data) is the fixed substrate. Keep the diff small and reviewable.
- **Fixed budget = comparable runs.** Don't extend the budget to win; tune what
  fits inside it (arch size, lr schedule, batch size, data mix, curriculum).
- **No leakage.** val is task-stratified in `prepare.py`. Don't train on val.
- **No hallucinated authority.** The model may propose analyses; it must never
  emit sūtra numbers or claim a derivation the verifier didn't confirm.
- **Self-contained.** CPU/MPS only. No CUDA-only ops, no new heavy deps.

## Workflow each experiment

1. Read the last run's `val_bpb` + decode demos.
2. Form ONE hypothesis (e.g. "wider n_embd helps morph more than denoise").
3. Make the minimal `train.py` change.
4. `uv run train.py --budget-min 5` (or `--max-steps N` for determinism).
5. Record: change, val_bpb, demo quality, keep/revert.

## Idea backlog (unordered)

- Curriculum: denoise-heavy early, then up-weight seg + morph (spec stages 1→2).
- Per-task loss weighting; the mixture is imbalanced (meter is small, morph large).
- Constrained decode for `seg`: restrict to boundaries the lattice licenses.
- Add a verification-survival eval hook that round-trips `morph` decodes through
  `slm/rules.DhatuKosha` and reports survival % alongside val_bpb.
- Try n_layer/n_embd sweeps under the budget; find the compute-optimal point.
- Weight tying / dropout / warmup sweeps.
- ByT5 encoder-decoder ablation (the spec's real target) once the pipeline holds.

## What NOT to do

- Don't grow the budget, add val to train, or special-case the demo prompts.
- Don't add a dependency on a GPU or a cloud service.
- Don't make the model emit prakriyā traces as targets (source-accountability,
  spec §2.5) — traces come from the verifier only.

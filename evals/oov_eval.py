"""evals/oov_eval.py — honest open-vocabulary segmentation metric (plan §Phase-2 #1).

WHY THIS EXISTS
---------------
The headline 87.8% exact (docs/plan/improvement-plan.md) is a CLOSED-CORPUS,
IN-LEXICON number: the ~85K-pada lexicon (data/corpus.json "vocab") and the seg
val split (data/val.json) both derive from the same 33,128 Rāmāyaṇa ślokas, so
the lexicon-anchored DP tier (slm/segdp.py, tier 1) is scored on padas it was
built from. That val set structurally CANNOT observe the failure mode the tiered
design exists for — the OOV cliff — so the coverage tier-gate (when seg_dp
should fall back from the lexicon-DP tier to the neural seg_constrained tier) is
untestable on it. The architect flagged this as a Phase-2 hard blocker.

WHAT THIS BUILDS
----------------
An OOV-injected dev set: we deliberately DELETE a controlled set of gold padas
from the segmenter's lexicon so a target fraction of eval rows contain >=1 pada
that is absent from the lexicon (simulated unseen vocabulary). We keep the gold
segmentations for scoring, run the FULL tiered pipeline (Inference.seg_dp) with
the ablated lexicon, and report, side by side (standing eval-hygiene requirement,
expert #4 / developer #2):
  * in-lexicon exact / F1  — rows whose gold padas all survive the ablation
  * OOV-row  exact / F1    — rows with >=1 held-out pada: the open-vocab number
  * tier attribution       — how often each tier fired and its accuracy, split
                             by in-lexicon vs OOV: this calibrates the tier-gate.

INJECTION (deterministic, controlled)
-------------------------------------
Reconstruct (surface, gold-padas) from each seg val row exactly as evals/eval.py
does. Deterministically designate ceil(oov_frac * N) rows as injection sites
(seeded shuffle). At each site we hold out ONE gold pada — the RAREST gold pada
of len>=3 that is currently in the lexicon — from the vocab. Rarest-first (a)
biases toward content words rather than high-frequency function words (a
plausible "unseen word"), and (b) minimizes collateral: a rare pada makes few
OTHER rows OOV. The union of held-out padas defines the ablated vocab; we then
RECLASSIFY every row against that union, so the realized OOV fraction (reported
alongside the requested one) reflects incidental sharing honestly.

LEAK CHECK (the point — the DP must genuinely have no path for a held-out pada)
------------------------------------------------------------------------------
Every pada the DP can emit — verbatim edge OR sandhi junction edge (which emits
its reconstructed `first`/`second` side) OR pausa variant — is gated through
LexiconSegmenter._score(), a plain vocab membership lookup; the only non-lexicon
edge is the penalized OOV-span edge, which seg_dp() never enables (kbest default
oov_penalty=None). Therefore a pada string P with P∉vocab is UNEMITTABLE by the
DP as a unit. We (1) build the segmenter from vocab-minus-held-out so removal is
by construction, (2) assert each held-out pada is absent from segmenter.vocab
AND that _score() returns None for it (no verbatim/junction path), and (3)
empirically confirm zero held-out padas appear in ANY predicted segmentation
across the whole run (leak count must be 0). A DP that still produced a held-out
pada would be cheating; leak=0 proves it cannot.

HONEST-LABELING CAVEAT
----------------------
This is a SEMI-SYNTHETIC open-vocab proxy, not a natural out-of-domain test. The
held-out padas are real Rāmāyaṇa padas we hid from the lexicon, not genuinely
unseen words from another genre — the surface text is still in-domain, only the
lexicon is impoverished. It isolates the tier-gate + fallback-floor behavior
under vocabulary gaps; the fully honest open-domain number awaits the DCS
out-of-domain eval (plan §Phase-2 #2). "In-lexicon" numbers here are still the
closed-corpus product number; the OOV-row number is the open-vocab floor for the
tiered pipeline under injected gaps.

Run:  uv run python -m evals.oov_eval
      uv run python -m evals.oov_eval --oov-frac 0.5 --n 400 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from evals.baseline import word_f1
from slm.infer import Inference
from slm.segdp import LexiconSegmenter
from slm.tokenizer import SLP1Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_seg_rows(tok: SLP1Tokenizer, n_cap: int | None) -> list[dict]:
    """Reconstruct (surface, gold padas) from the seg val split, exactly as
    evals/eval.py does: strip BOS/SEP/EOS, drop the '<seg>' src tag, split gold
    on '|'."""
    val = json.loads((DATA / "val.json").read_text())
    rows = []
    for e in val:
        if e["task"] != "seg":
            continue
        ids, ls = e["ids"], e["loss_start"]
        src = tok.decode(ids[1:ls - 1], skip_special=False)
        gold = tok.decode(ids[ls:-1], skip_special=False)
        text = src.removeprefix("<seg>")
        gold_padas = [p.strip() for p in gold.split("|") if p.strip()]
        if not gold_padas:
            continue
        rows.append({"text": text, "gold": gold_padas})
    if n_cap is not None:
        rows = rows[:n_cap]
    return rows


def choose_holdout(gold: list[str], vocab: dict[str, int]) -> str | None:
    """Pick the pada to hold out for an injection-site row: the RAREST in-vocab
    gold pada of len>=3. Rarest-first biases to content words and minimizes
    collateral OOV. Returns None if the row has no removable content pada (it is
    then left as-is — possibly already OOV)."""
    cands = [p for p in gold if len(p) >= 3 and p in vocab]
    if not cands:
        return None
    # freq asc, then longer, then lexicographic — fully deterministic
    return min(cands, key=lambda p: (vocab[p], -len(p), p))


def inject_oov(rows: list[dict], full_vocab: dict[str, int], oov_frac: float,
               seed: int) -> set[str]:
    """Deterministically hold out padas so ~oov_frac of rows go OOV. Returns the
    set of held-out pada strings (the ablation)."""
    import random
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_sites = math.ceil(oov_frac * len(rows))
    held: set[str] = set()
    for i in idx[:n_sites]:
        p = choose_holdout(rows[i]["gold"], full_vocab)
        if p is not None:
            held.add(p)
    return held


def classify(rows: list[dict], held: set[str]) -> None:
    """Tag each row: oov=True iff >=1 gold pada was held out of the lexicon."""
    for r in rows:
        r["held"] = [p for p in r["gold"] if p in held]
        r["oov"] = bool(r["held"])


def leak_check(seg: LexiconSegmenter, held: set[str]) -> list[str]:
    """Structural guarantee: a held-out pada must have NO verbatim/junction path.
    Both edge kinds route through _score() (a vocab lookup); seg_dp never enables
    the OOV-span edge. So P∉vocab ⇒ P unemittable. Return any leaks (must be [])."""
    leaks = []
    for p in held:
        if p in seg.vocab or seg._score(p) is not None:
            leaks.append(p)
    return leaks


def fmt(exact: int, n: int, f1: float) -> str:
    p = 100 * exact / n if n else 0.0
    fp = 100 * f1 / n if n else 0.0
    return f"{exact:4d}/{n:<4d} ({p:5.1f}%)   F1 {fp:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oov-frac", type=float, default=0.3,
                    help="target fraction of eval rows carrying >=1 OOV pada")
    ap.add_argument("--n", type=int, default=None, help="cap seg rows (default all)")
    ap.add_argument("--seed", type=int, default=0, help="injection RNG seed")
    ap.add_argument("--k", type=int, default=8, help="DP k-best width")
    args = ap.parse_args()

    tok = SLP1Tokenizer.load(ROOT / "tokenizer" / "slp1_vocab.json")
    rows = load_seg_rows(tok, args.n)

    full_vocab = json.loads((DATA / "corpus.json").read_text())["vocab"]
    held = inject_oov(rows, full_vocab, args.oov_frac, args.seed)
    classify(rows, held)

    # ablated lexicon = full minus held-out; same hygiene (prune_spans) as prod
    ablated = {w: c for w, c in full_vocab.items() if w not in held}
    seg = LexiconSegmenter(vocab=ablated)
    leaks = leak_check(seg, held)

    # run the FULL tiered pipeline with the ablated segmenter injected
    inf = Inference()
    inf._segdp = seg  # seg_dp() reuses this instead of building the full-vocab one

    n_oov = sum(r["oov"] for r in rows)
    print(f"OOV-injected open-vocab segmentation eval "
          f"(semi-synthetic proxy — see module docstring)")
    print(f"  device={inf.device}  ckpt val_bpb={inf.val_bpb:.4f}  k={args.k}")
    print(f"  rows={len(rows)}  full-vocab={len(full_vocab)}  ablated-vocab={len(seg.vocab)}")
    print(f"  held-out padas={len(held)} unique  "
          f"oov-frac requested={args.oov_frac:.2f}  "
          f"realized={n_oov/len(rows):.2f} ({n_oov}/{len(rows)})")
    print(f"  leak check (held-out padas emittable by DP): {len(leaks)}  "
          f"[{'PASS' if not leaks else 'FAIL: ' + ', '.join(leaks[:5])}]")
    if held:
        ex = min(held, key=lambda p: (full_vocab[p], p))
        print(f"  e.g. held out '{ex}' (corpus freq {full_vocab[ex]})")

    # aggregate buckets
    agg = {b: {"exact": 0, "f1": 0.0, "n": 0} for b in ("in_lex", "oov", "all")}
    # tier attribution: per (tier) and per (tier, bucket)
    tier = defaultdict(lambda: {"exact": 0, "n": 0})
    tier_bucket = defaultdict(lambda: {"exact": 0, "n": 0})
    dp_leak = 0        # held-out pada emitted by the LEXICON-DP tier — must be 0
    fb_recover = 0     # held-out pada recovered by the neural fallback — expected

    for r in rows:
        res = inf.seg_dp(r["text"], k=args.k)
        pred, t = res["padas"], res.get("tier", "?")
        ok = pred == r["gold"]
        f1 = word_f1(pred, r["gold"])
        bucket = "oov" if r["oov"] else "in_lex"
        for b in (bucket, "all"):
            agg[b]["exact"] += ok
            agg[b]["f1"] += f1
            agg[b]["n"] += 1
        tier[t]["exact"] += ok
        tier[t]["n"] += 1
        tier_bucket[(t, bucket)]["exact"] += ok
        tier_bucket[(t, bucket)]["n"] += 1
        n_held_in_pred = sum(p in held for p in pred)
        if t == "lexicon-dp":
            dp_leak += n_held_in_pred        # cheat: DP must not emit held-out padas
        else:
            fb_recover += n_held_in_pred      # legit: copy-constrained surface recovery

    # The DP tier is vocab-gated, so it must NEVER emit a held-out pada. The
    # neural fallback is copy-constrained on the surface, so it CAN reproduce a
    # held-out pada (that is the generalization floor doing its job) — those are
    # counted separately as recoveries, not leaks.
    print(f"\n  empirical DP leak (held-out padas emitted by lexicon-dp tier): "
          f"{dp_leak}  [{'PASS' if dp_leak == 0 else 'FAIL'}]")
    print(f"  neural-fallback surface recoveries of held-out padas: {fb_recover} "
          f"(expected — copy-constrained tier is not vocab-gated)")

    print(f"\nsegmentation accuracy (ablated lexicon, full tiered seg_dp)")
    print(f"  {'in-lexicon rows':20s} {fmt(agg['in_lex']['exact'], agg['in_lex']['n'], agg['in_lex']['f1'])}")
    print(f"  {'OOV rows (open-vocab)':20s} {fmt(agg['oov']['exact'], agg['oov']['n'], agg['oov']['f1'])}")
    print(f"  {'all rows':20s} {fmt(agg['all']['exact'], agg['all']['n'], agg['all']['f1'])}")

    print(f"\ntier attribution (calibrates the coverage tier-gate)")
    print(f"  {'tier':22s} {'fired':>10s} {'exact-acc':>11s}")
    for t in sorted(tier):
        s = tier[t]
        print(f"  {t:22s} {s['n']:5d}/{len(rows):<4d} "
              f"{100*s['exact']/max(1,s['n']):9.1f}%")
    print(f"\n  tier x bucket (did OOV rows route to the fallback as intended?)")
    print(f"  {'tier':22s} {'bucket':8s} {'fired':>8s} {'exact-acc':>11s}")
    for (t, b) in sorted(tier_bucket):
        s = tier_bucket[(t, b)]
        print(f"  {t:22s} {b:8s} {s['n']:6d}   {100*s['exact']/max(1,s['n']):9.1f}%")

    # tier-gate summary: of OOV rows, what fraction fell back?
    oov_fb = tier_bucket[("constrained-fallback", "oov")]["n"]
    inlex_dp = tier_bucket[("lexicon-dp", "in_lex")]["n"]
    print(f"\n  gate summary: {oov_fb}/{n_oov} OOV rows fell back to neural; "
          f"{inlex_dp}/{agg['in_lex']['n']} in-lexicon rows stayed on lexicon-dp")


if __name__ == "__main__":
    main()

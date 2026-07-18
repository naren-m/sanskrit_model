"""evals/eval.py — task metrics beyond val_bpb (Path A spec §5).

Loads a trained checkpoint and scores the val split per task:

  * seg   : exact-match of the ' | '-joined pada segmentation.
  * morph : dhatu top-1 accuracy (did the decoded <dhAtu> field match gold?).
  * verify-survival : of decoded morph analyses, what % name a root the symbolic
    DhatuKosha actually knows (neural proposes, Panini disposes — the metric that
    gates the model's place in the engine).
  * sandhi / meter / denoise : exact-match of the full target string.

Run:  uv run python -m evals.eval            (uses data/ckpt.pt)
      uv run python -m evals.eval --n 300    (cap examples per task)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from slm.model import GPT, GPTConfig
from slm.tokenizer import SLP1Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device):
    # self-generated checkpoint; weights_only=True still allows the plain-dict
    # cfg/meta payload and avoids arbitrary-code unpickling.
    ck = torch.load(DATA / "ckpt.pt", map_location=device, weights_only=True)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def decode(model, tok, device, src: str, max_new=64) -> str:
    ids = [tok.bos_id] + tok.encode(src) + [tok.sep_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, eos_id=tok.eos_id,
                         temperature=1e-6, top_k=1)  # greedy
    gen = out[0, len(ids):].tolist()
    if tok.eos_id in gen:
        gen = gen[:gen.index(tok.eos_id)]
    return tok.decode(gen, skip_special=False)


_DHATU_RE = re.compile(r"<dhAtu>([^<]*)")


def ab_main(n_cap: int):
    """A/B gate (tasks-jm8.1, spec §7): does the neural ranker beat a char
    trigram and a pure lexicon+frequency ranker at picking the gold
    segmentation out of the SAME lattice candidate pool?

    Arms (identical pools, only the sequence-scorer term differs):
      symbolic — lexicon coverage + log-freq only
      trigram  — + char-trigram logprob (trained on the GPT's own corpus)
      neural   — + GPT teacher-forced logprob (production scorer)
      generate — GPT free decode (production fallback path; own output, no
                 pool), with 'rederiv' = does its split concat back to src?
    """
    from evals.baseline import (AblationRanker, CharTrigram, NeuralSeqScorer,
                                training_lines, word_f1)
    from slm.infer import Inference

    tok = SLP1Tokenizer.load(ROOT / "tokenizer" / "slp1_vocab.json")
    inf = Inference()
    print(f"device={inf.device}  ckpt val_bpb={inf.val_bpb:.4f}")
    print("training trigram on corpus...", flush=True)
    tri = CharTrigram(training_lines())

    holder: dict = {"src": ""}
    ranker = AblationRanker({
        "symbolic": None,
        "trigram": tri,
        "neural": NeuralSeqScorer(inf, holder),
    })

    val = json.loads((DATA / "val.json").read_text())
    rows = [e for e in val if e["task"] == "seg"][:n_cap]
    arms = ["symbolic", "trigram", "neural", "generate", "constrained"]
    stats = {a: {"exact": 0, "f1": 0.0, "rederiv": 0, "hit_pool": 0} for a in arms}
    in_pool = 0  # oracle ceiling: gold present in the shared candidate pool

    for e in rows:
        ids, ls = e["ids"], e["loss_start"]
        src = tok.decode(ids[1:ls - 1], skip_special=False)
        gold = tok.decode(ids[ls:-1], skip_special=False)
        text = src.removeprefix("<seg>")
        gold_padas = [p.strip() for p in gold.split("|") if p.strip()]
        holder["src"] = text

        cands = ranker.candidates(text)
        gold_avail = gold_padas in cands
        in_pool += gold_avail
        for arm in arms:
            if arm == "generate":
                pred = [p.strip() for p in
                        decode(inf.model, tok, inf.device, src).split("|") if p.strip()]
            elif arm == "constrained":
                pred = inf.seg_constrained(text)["padas"]
            else:
                pred = ranker.rank(text, arm, cands)
            s = stats[arm]
            s["exact"] += pred == gold_padas
            s["f1"] += word_f1(pred, gold_padas)
            s["rederiv"] += "".join(pred) == text
            s["hit_pool"] += gold_avail and pred == gold_padas

    n = len(rows)
    print(f"\nA/B seg ablation on {n} held-out examples "
          f"(identical lattice pools for rank arms)")
    print(f"pool ceiling (gold in lattice pool): {in_pool}/{n} "
          f"({100*in_pool/max(1,n):.1f}%) — rank arms cannot exceed this; "
          f"'hit|pool' is accuracy on just those rows")
    print(f"{'arm':10s} {'exact':>12s} {'word-F1':>9s} {'rederiv':>9s} {'hit|pool':>10s}")
    for arm in arms:
        s = stats[arm]
        hp = f"{s['hit_pool']}/{in_pool}" if arm not in ("generate", "constrained") else "—"
        print(f"{arm:10s} {s['exact']:5d}/{n:<4d} {100*s['exact']/n:5.1f}% "
              f"{100*s['f1']/n:8.1f}% {100*s['rederiv']/n:8.1f}% {hp:>10s}")

    nx, tx, sx = (stats[a]["exact"] for a in ("neural", "trigram", "symbolic"))
    nf, tf, sf = (stats[a]["f1"] for a in ("neural", "trigram", "symbolic"))
    print(f"\nneural beats trigram baseline:  "
          f"exact {'YES' if nx > tx else 'NO'} ({nx} vs {tx}), "
          f"F1 {'YES' if nf > tf else 'NO'} ({100*nf/n:.1f} vs {100*tf/n:.1f})")
    print(f"neural beats pure-symbolic:     "
          f"exact {'YES' if nx > sx else 'NO'} ({nx} vs {sx}), "
          f"F1 {'YES' if nf > sf else 'NO'} ({100*nf/n:.1f} vs {100*sf/n:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="cap examples per task")
    ap.add_argument("--ab", action="store_true",
                    help="A/B gate: neural vs trigram vs symbolic seg ranking")
    args = ap.parse_args()
    if args.ab:
        ab_main(args.n)
        return

    device = pick_device()
    tok = SLP1Tokenizer.load(ROOT / "tokenizer" / "slp1_vocab.json")
    model, ck = load_model(device)
    print(f"device={device}  ckpt val_bpb={ck.get('val_bpb'):.4f}")

    val = json.loads((DATA / "val.json").read_text())
    by_task = defaultdict(list)
    for e in val:
        by_task[e["task"]].append(e)

    # optional symbolic verifier for morph survival
    try:
        from slm import rules
        kosha = rules.DhatuKosha()
    except Exception as e:
        kosha = None
        print(f"  (verifier unavailable: {e})")

    print(f"\n{'task':10s} {'n':>5s} {'exact':>8s}  extra")
    for task, rows in sorted(by_task.items()):
        rows = rows[: args.n]
        exact = 0
        survive = 0
        for e in rows:
            # reconstruct src/tgt from ids using the sep boundary
            ids = e["ids"]
            ls = e["loss_start"]
            src = tok.decode(ids[1:ls - 1], skip_special=False)  # after BOS, before SEP
            gold = tok.decode(ids[ls:-1], skip_special=False)     # tgt, drop EOS
            pred = decode(model, tok, device, src)
            if pred.strip() == gold.strip():
                exact += 1
            if task == "morph" and kosha is not None:
                m = _DHATU_RE.search(pred)
                if m:
                    root = rules.strip_anubandhas(m.group(1))
                    if root and kosha.lookup(root):
                        survive += 1
        n = len(rows)
        extra = ""
        if task == "morph" and kosha is not None:
            extra = f"verify-survival={survive}/{n} ({100*survive/max(1,n):.1f}%)"
        print(f"{task:10s} {n:5d} {exact:5d}/{n:<3d} {100*exact/max(1,n):5.1f}%  {extra}")


if __name__ == "__main__":
    main()

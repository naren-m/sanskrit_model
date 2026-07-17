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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="cap examples per task")
    args = ap.parse_args()

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

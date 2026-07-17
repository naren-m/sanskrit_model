"""train.py — autoresearch-style single-file trainer for the Sanskrit LM.

This is THE file to edit when doing autonomous research (see program.md). Keep
it self-contained and the diff reviewable. Fixed wall-clock budget makes runs
comparable; the headline metric is val_bpb (bits per byte over tgt spans).

Device: Apple-Silicon MPS if available, else CPU. No CUDA-only code paths.

Run:  uv run train.py                      (default ~5 min budget)
      uv run train.py --budget-min 1       (quick smoke)
      uv run train.py --max-steps 200      (step-bounded instead of time-bounded)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from slm.model import GPT, GPTConfig
from slm.tokenizer import SLP1Tokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LN2 = math.log(2.0)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(name: str) -> list[dict]:
    return json.loads((DATA / f"{name}.json").read_text())


def make_batch(rows, idxs, pad_id, device):
    """Pad a list of examples; build next-token targets with loss only on the
    tgt span (positions before loss_start and padding are set to ignore=-1)."""
    batch = [rows[i] for i in idxs]
    maxlen = max(len(b["ids"]) for b in batch)
    B = len(batch)
    x = torch.full((B, maxlen), pad_id, dtype=torch.long)
    y = torch.full((B, maxlen), -1, dtype=torch.long)
    tgt_tokens = 0
    for bi, ex in enumerate(batch):
        ids = ex["ids"]
        n = len(ids)
        x[bi, :n] = torch.tensor(ids, dtype=torch.long)
        ls = ex["loss_start"]
        # target at position t is ids[t+1]; supervise only when t+1 is in tgt span
        for t in range(n - 1):
            if (t + 1) >= ls:
                y[bi, t] = ids[t + 1]
                tgt_tokens += 1
    return x.to(device), y.to(device), tgt_tokens


@torch.no_grad()
def evaluate(model, rows, cfg, pad_id, device, batch_size, max_batches=50):
    model.eval()
    tot_nats, tot_tok = 0.0, 0
    for start in range(0, min(len(rows), max_batches * batch_size), batch_size):
        idxs = list(range(start, min(start + batch_size, len(rows))))
        x, y, ntok = make_batch(rows, idxs, pad_id, device)
        _, loss = model(x, targets=y, ignore_index=-1)
        if ntok:
            tot_nats += loss.item() * ntok
            tot_tok += ntok
    model.train()
    if tot_tok == 0:
        return float("nan"), float("nan")
    val_loss = tot_nats / tot_tok            # nats / token
    val_bpb = val_loss / LN2                  # bits / token (~byte for SLP1)
    return val_loss, val_bpb


def sample_demos(model, tok, device, cfg):
    """Greedy-ish decode a few held-in task prompts to eyeball behavior."""
    prompts = [
        "<morph>gam", "<morph>BU", "<sandhi>rAma<sep>asti",
        "<seg>rAmo'sti", "<Candas><wt>LGLGLLLG",
    ]
    model.eval()
    outs = []
    for p in prompts:
        ids = [tok.bos_id] + tok.encode(p) + [tok.sep_id]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=48, eos_id=tok.eos_id,
                             temperature=0.7, top_k=20)
        gen = out[0, len(ids):].tolist()
        outs.append((p, tok.decode(gen, skip_special=False).split("<eos>")[0]))
    model.train()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=5.0)
    ap.add_argument("--max-steps", type=int, default=0, help="if >0, overrides budget")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    meta = json.loads((DATA / "meta.json").read_text())
    tok = SLP1Tokenizer.load(ROOT / "tokenizer" / "slp1_vocab.json")
    train_rows, val_rows = load_split("train"), load_split("val")
    print(f"device={device}  vocab={meta['vocab_size']}  block={meta['block_size']}"
          f"  train={len(train_rows)}  val={len(val_rows)}")

    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=meta["block_size"],
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"model params: {model.num_params()/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    pad_id = meta["pad_id"]
    rng = torch.Generator().manual_seed(args.seed)

    def lr_at(step, total):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        if total <= args.warmup:
            return args.min_lr
        prog = (step - args.warmup) / max(1, total - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))

    # estimate total steps for the cosine schedule
    est_total = args.max_steps if args.max_steps > 0 else 3000
    t0 = time.time()
    best_bpb = float("inf")
    step = 0
    model.train()
    while True:
        if args.max_steps > 0 and step >= args.max_steps:
            break
        if args.max_steps == 0 and (time.time() - t0) > args.budget_min * 60:
            break

        lr = lr_at(step, est_total)
        for g in opt.param_groups:
            g["lr"] = lr
        idxs = torch.randint(0, len(train_rows), (args.batch_size,), generator=rng).tolist()
        x, y, _ = make_batch(train_rows, idxs, pad_id, device)
        _, loss = model(x, targets=y, ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0:
            vl, vbpb = evaluate(model, val_rows, cfg, pad_id, device, args.batch_size)
            el = time.time() - t0
            print(f"step {step:5d} | {el:5.1f}s | lr {lr:.2e} | "
                  f"train {loss.item():.3f} | val {vl:.3f} | val_bpb {vbpb:.4f}")
            best_bpb = min(best_bpb, vbpb)
        step += 1

    vl, vbpb = evaluate(model, val_rows, cfg, pad_id, device, args.batch_size)
    best_bpb = min(best_bpb, vbpb)
    print(f"\nFINAL  steps={step}  time={(time.time()-t0):.1f}s  "
          f"val_loss={vl:.4f}  val_bpb={vbpb:.4f}  best_bpb={best_bpb:.4f}")

    print("\n--- decode demos ---")
    for p, o in sample_demos(model, tok, device, cfg):
        print(f"  {p:24s} -> {o!r}")

    ckpt = ROOT / "data" / "ckpt.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "meta": meta,
                "val_bpb": vbpb}, ckpt)
    print(f"\nsaved {ckpt}")


if __name__ == "__main__":
    main()

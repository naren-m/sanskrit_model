"""prepare.py — autoresearch-style data + tokenizer stage for the Sanskrit LM.

Mirrors karpathy/autoresearch's prepare.py role: build the tokenizer, generate
the training corpus, and serialize encoded tensors the trainer consumes.

Steps:
  1. Build + save the SLP1 char tokenizer (tokenizer/slp1_vocab.json).
  2. Generate the multi-task mixture from the CSV rule assets (slm/datagen).
  3. Encode each example as  [BOS] <task> src <sep> tgt <eos>  and record the
     index where the tgt span begins, so training takes loss only on tgt.
  4. Task-stratified split into train/val, saved as data/{train,val}.pkl plus
     data/meta.json.

Run:  uv run prepare.py            (or: uv run prepare.py --quick  for a smoke set)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from slm import datagen
from slm.tokenizer import SLP1Tokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TOK_PATH = ROOT / "tokenizer" / "slp1_vocab.json"


def encode_example(tok: SLP1Tokenizer, ex: dict) -> dict:
    """Return {ids, loss_start, task}. Loss applies from the <sep> boundary's
    tgt side through <eos>; the src (prompt) region is masked out in training."""
    src_ids = tok.encode(ex["src"])
    tgt_ids = tok.encode(ex["tgt"])
    ids = [tok.bos_id] + src_ids + [tok.sep_id] + tgt_ids + [tok.eos_id]
    loss_start = 1 + len(src_ids) + 1  # first tgt token position
    return {"ids": ids, "loss_start": loss_start, "task": ex["task"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test set")
    ap.add_argument("--block-size", type=int, default=256)
    args = ap.parse_args()

    n_sandhi = 400 if args.quick else 8000
    n_denoise = 300 if args.quick else 6000
    morph_cap = 800 if args.quick else None

    print("[1/4] building tokenizer")
    tok = SLP1Tokenizer.build()
    tok.save(TOK_PATH)
    print(f"      vocab_size={tok.vocab_size}  -> {TOK_PATH}")

    print("[2/4] generating mixture from CSV rules")
    mix_path = DATA / ("mixture_quick.jsonl" if args.quick else "mixture.jsonl")
    counts = datagen.build_all(args.seed, mix_path, n_sandhi, n_denoise, morph_cap)
    print("      " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    print("[3/4] encoding")
    examples, dropped = [], 0
    with mix_path.open() as f:
        for line in f:
            ex = json.loads(line)
            enc = encode_example(tok, ex)
            if len(enc["ids"]) > args.block_size:
                dropped += 1
                continue
            examples.append(enc)
    print(f"      encoded={len(examples)}  dropped(> {args.block_size})={dropped}")

    print("[4/4] task-stratified train/val split")
    import random
    rng = random.Random(args.seed)
    by_task: dict[str, list] = {}
    for e in examples:
        by_task.setdefault(e["task"], []).append(e)
    train, val = [], []
    for task, rows in by_task.items():
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_frac))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train); rng.shuffle(val)

    DATA.mkdir(parents=True, exist_ok=True)
    # JSON (not pickle) so the encoded splits carry no code-execution risk.
    (DATA / "train.json").write_text(json.dumps(train))
    (DATA / "val.json").write_text(json.dumps(val))
    max_len = max(len(e["ids"]) for e in examples)
    meta = {
        "vocab_size": tok.vocab_size,
        "block_size": args.block_size,
        "max_len": max_len,
        "pad_id": tok.pad_id,
        "n_train": len(train),
        "n_val": len(val),
        "train_task_counts": dict(Counter(e["task"] for e in train)),
        "val_task_counts": dict(Counter(e["task"] for e in val)),
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"      train={len(train)}  val={len(val)}  max_len={max_len}")
    print(f"      -> {DATA/'train.json'} , {DATA/'val.json'} , {DATA/'meta.json'}")
    print("done.")


if __name__ == "__main__":
    main()

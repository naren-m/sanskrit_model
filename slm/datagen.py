"""Build the multi-task training mixture from the SLP1 CSV rule assets.

This is the v0 "basic model with all the rules" data builder. It produces every
kind of pair the spec (section 2) asks for *that is derivable from the CSVs
alone* — no vidyut-prakriya dependency yet:

  * morph   : root -> dhatu analysis      (deterministic, from dhatus-*.csv)
  * sandhi  : pada1 + pada2 -> joined     (forward sandhi via SandhiEngine)
  * seg     : joined -> pada1 | pada2      (inverse of the above; supervised split)
  * meter   : L/G pattern -> meter name    (from meters-full.csv)
  * denoise : span-corrupted text -> spans (T5-style, over synthetic sandhied text)

Genuine inflected tinanta/subanta generation needs vidyut-prakriya and is left
as the v1 upgrade (see README / program.md). Everything here is rule-licensed.

Output: JSONL of {"task","src","tgt","provenance"} in SLP1.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from slm import corpus, rules

ROOT = Path(__file__).resolve().parent.parent

# Fallback hand-checked seed padas, used only if the real corpus cache is
# unavailable. Normally the vocabulary comes from ../ramayanam via slm.corpus.
SEED_PADAS = [
    "rAmaH", "devaH", "guruH", "vanam", "jalam", "nadI", "sItA", "saH", "sA",
    "tat", "aham", "tvam", "ca", "eva", "iti", "api", "tu", "hi", "tataH",
    "atra", "gacCati", "asti", "Bavati", "gatvA", "kftvA", "uvAca", "namaH",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# --- task builders -----------------------------------------------------------
def build_morph(kosha: rules.DhatuKosha, cap: int | None = None) -> list[dict]:
    rows = _read_csv(ROOT / "dhatus-full.csv")
    out = []
    for r in rows:
        root = rules.strip_anubandhas(r["dhatu_slp1"])
        if not root:
            continue
        tgt = (f"<dhAtu>{r['dhatu_slp1']}<gaRa>{r['gana']}"
               f"<artha>{r['artha_slp1']}")
        out.append({"task": "morph", "src": f"<morph>{root}", "tgt": tgt,
                    "provenance": f"dhatupatha:{r['code']}"})
    random.shuffle(out)
    return out[:cap] if cap else out


# clean SLP1 pseudo-word fragments to carry a junction (teach sandhi as a LOCAL
# transformation, independent of the carrier). Prefixes END IN A CONSONANT so
# prefix+first stays a natural CV syllable (r+a -> "ra", not vowel hiatus).
_CARRIER_PRE = ["r", "m", "d", "s", "n", "v", "k", "g", "j", "h", "p", "t",
                "gr", "pr", "str", "dh", "S", "rAm", "dev"]
_CARRIER_SUF = ["ma", "na", "ka", "ta", "ya", "va", "sya", "ti", "tu", "H", "m"]


def build_sandhi(engine: rules.SandhiEngine, vocab: list[str], n_pairs: int) -> list[dict]:
    """Synthetic join task: two real padas -> sandhied surface. Identity (no
    sandhi fires) kept only rarely so the model does not learn concatenation."""
    out = []
    for _ in range(n_pairs):
        a, b = random.choice(vocab), random.choice(vocab)
        joined, cat = engine.join(a, b)[0]
        joined = joined.replace(" ", "")
        if joined == a + b and cat == "no-sandhi" and random.random() > 0.05:
            continue
        out.append({"task": "sandhi", "src": f"<sandhi>{a}<sep>{b}",
                    "tgt": joined, "provenance": f"forward-sandhi:{cat}"})
    return out


def build_sandhi_coverage(engine: rules.SandhiEngine, per_rule: int) -> list[dict]:
    """Uniform coverage of EVERY sandhi rule. For rule (first,second,result),
    build junctions  <carrier>+first | second+<carrier>  so each rule type
    (esp. rare vowel sandhis) is seen ~per_rule times, not at its Zipfian rate."""
    rows = _read_csv(ROOT / "sandhi-rules-full.csv")
    out = []
    for r in rows:
        f, s = r["first_slp1"], r["second_slp1"]
        if not f or not s:
            continue
        for _ in range(per_rule):
            left = random.choice(_CARRIER_PRE) + f
            right = s + random.choice(_CARRIER_SUF)
            joined = engine.join(left, right)[0][0].replace(" ", "")
            out.append({"task": "sandhi", "src": f"<sandhi>{left}<sep>{right}",
                        "tgt": joined, "provenance": f"rule-cover:{r['category']}"})
    return out


def build_seg(engine: rules.SandhiEngine, sloka_seqs: list[list[str]],
              n_examples: int) -> list[dict]:
    """Gold segmentation from REAL Ramayana word order: take a window of 2-4
    consecutive padas, forward-apply sandhi across the whole window to get the
    sandhied surface, and supervise the split back to the original padas."""
    out = []
    tries = 0
    while len(out) < n_examples and tries < n_examples * 4:
        tries += 1
        seq = random.choice(sloka_seqs)
        if len(seq) < 2:
            continue
        w = random.randint(2, min(4, len(seq)))
        start = random.randint(0, len(seq) - w)
        window = seq[start:start + w]
        surface = window[0]
        for nxt in window[1:]:
            surface = engine.join(surface, nxt)[0][0]
        # CSV sandhi results can carry spaces (e.g. "tad Ditam"); strip them so
        # the seg input is genuinely continuous and does not leak boundaries.
        surface = surface.replace(" ", "")
        if len(surface) > 60:
            continue
        out.append({"task": "seg", "src": f"<seg>{surface}",
                    "tgt": " | ".join(window), "provenance": "ramayanam-gold"})
    return out


def build_meter() -> list[dict]:
    rows = _read_csv(ROOT / "meters-full.csv")
    out = []
    for r in rows:
        pat = r.get("pattern", "").strip()
        name = r.get("name_slp1", "").strip()  # SLP1 (ASCII); name_iast has diacritics
        if not pat or not name:
            continue
        out.append({"task": "meter", "src": f"<Candas><wt>{pat}",
                    "tgt": f"<meter>{name}", "provenance": "meters-full"})
    return out


def build_denoise(lines: list[str], n_examples: int, mask_rate: float = 0.15,
                  provenance: str = "corpus") -> list[dict]:
    """T5-style span corruption over real sandhied lines (yoga sutras + joined
    ramayana windows)."""
    out = []
    pool = [s for s in lines if len(s) >= 8]
    if not pool:
        return out
    for _ in range(n_examples):
        s = random.choice(pool)
        chars = list(s)
        span = max(2, int(round(len(chars) * mask_rate)))
        start = random.randint(0, max(0, len(chars) - span))
        masked = "".join(chars[start:start + span])
        src = "".join(chars[:start]) + "[MASK_0]" + "".join(chars[start + span:])
        out.append({"task": "denoise", "src": f"<denoise>{src}",
                    "tgt": f"[MASK_0]{masked}", "provenance": provenance})
    return out


def _denoise_lines(engine: rules.SandhiEngine, sloka_seqs: list[list[str]],
                   yoga_lines: list[str], n_joined: int) -> list[str]:
    """Real sandhied text: yoga sutra lines + sandhi-joined ramayana windows."""
    lines = list(yoga_lines)
    for _ in range(n_joined):
        seq = random.choice(sloka_seqs)
        w = random.randint(2, min(4, len(seq)))
        start = random.randint(0, len(seq) - w)
        s = seq[start]
        for nxt in seq[start + 1:start + w]:
            s = engine.join(s, nxt)[0][0]
        lines.append(s.replace(" ", ""))
    return lines


# --- driver ------------------------------------------------------------------
def build_all(seed: int, out_path: Path, n_sandhi: int, n_denoise: int,
              morph_cap: int | None) -> dict:
    random.seed(seed)
    engine = rules.SandhiEngine()
    kosha = rules.DhatuKosha()

    # real corpus-grounded vocabulary + word sequences (falls back to seed list)
    try:
        cdata = corpus.load_corpus()
        sloka_seqs = [s for s in cdata["ramayanam_padas"] if len(s) >= 2]
        yoga_lines = cdata["yoga_lines"]
        vocab = corpus.top_padas(n=6000, min_count=2) or SEED_PADAS
    except Exception as e:  # corpus optional; degrade gracefully
        print(f"  [datagen] corpus unavailable ({e}); using seed lexicon")
        sloka_seqs = [[random.choice(SEED_PADAS) for _ in range(3)] for _ in range(2000)]
        yoga_lines = []
        vocab = SEED_PADAS

    denoise_lines = _denoise_lines(engine, sloka_seqs, yoga_lines,
                                   n_joined=max(0, n_denoise - len(yoga_lines)))
    # keep every rule covered but do not let sandhi dominate the mixture
    per_rule = 2 if n_sandhi < 1000 else 3
    parts = {
        "morph": build_morph(kosha, cap=morph_cap),
        "sandhi": build_sandhi(engine, vocab, n_sandhi)
                  + build_sandhi_coverage(engine, per_rule),
        "seg": build_seg(engine, sloka_seqs, n_sandhi),
        "meter": build_meter(),
        "denoise": build_denoise(denoise_lines, n_denoise, provenance="corpus"),
    }
    all_rows = [r for v in parts.values() for r in v]
    random.shuffle(all_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {k: len(v) for k, v in parts.items()}
    counts["TOTAL"] = len(all_rows)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "mixture.jsonl"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-sandhi", type=int, default=8000)
    ap.add_argument("--n-denoise", type=int, default=6000)
    ap.add_argument("--morph-cap", type=int, default=None)
    args = ap.parse_args()
    counts = build_all(args.seed, Path(args.out), args.n_sandhi, args.n_denoise,
                       args.morph_cap)
    print("wrote", args.out)
    for k, v in counts.items():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()

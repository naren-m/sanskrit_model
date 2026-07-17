"""demo.py — the Sanskrit engine, live. Neural proposes, Panini disposes.

Runs the trained GPT and the symbolic rule engine side by side:

  * morph  : model proposes a dhatu analysis; DhatuKosha verifies it.
  * sandhi : model joins two padas; SandhiEngine gives the rule-licensed answer.
  * seg    : model segments continuous SLP1 into padas.
  * meter  : ChandasEngine scans an SLP1 line into laghu/guru + names the meter.

Usage:
  uv run demo.py                 # scripted showcase
  uv run demo.py --repl          # interactive: type  morph gam / sandhi rAma asti /
                                 #   seg rAmo'sti / meter Darmakzetrekurukzetre
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from slm import rules
from slm.model import GPT, GPTConfig
from slm.tokenizer import SLP1Tokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
_DHATU_RE = re.compile(r"<dhAtu>([^<]*)")
_GANA_RE = re.compile(r"<gaRa>([^<]*)")
_ARTHA_RE = re.compile(r"<artha>([^<]*)")

G = "\033[92m"; R = "\033[91m"; B = "\033[94m"; DIM = "\033[2m"; X = "\033[0m"


class Engine:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = SLP1Tokenizer.load(ROOT / "tokenizer" / "slp1_vocab.json")
        ck = torch.load(DATA / "ckpt.pt", map_location=self.device, weights_only=True)
        self.model = GPT(GPTConfig(**ck["cfg"])).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.val_bpb = ck.get("val_bpb")
        self.sandhi = rules.SandhiEngine()
        self.kosha = rules.DhatuKosha()
        self.chandas = rules.ChandasEngine()

    @torch.no_grad()
    def _gen(self, src: str, max_new=64) -> str:
        ids = [self.tok.bos_id] + self.tok.encode(src) + [self.tok.sep_id]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(x, max_new_tokens=max_new, eos_id=self.tok.eos_id,
                                  temperature=1e-6, top_k=1)
        gen = out[0, len(ids):].tolist()
        if self.tok.eos_id in gen:
            gen = gen[:gen.index(self.tok.eos_id)]
        return self.tok.decode(gen, skip_special=False)

    # -- tasks ----------------------------------------------------------------
    def morph(self, root: str):
        pred = self._gen(f"<morph>{root}")
        m, gn, ar = _DHATU_RE.search(pred), _GANA_RE.search(pred), _ARTHA_RE.search(pred)
        upa = m.group(1) if m else "?"
        gana = gn.group(1) if gn else "?"
        artha = ar.group(1) if ar else "?"
        clean = rules.strip_anubandhas(upa) if m else ""
        hits = self.kosha.lookup(clean) if clean else []
        ok = bool(hits)
        print(f"  {B}model proposes{X}: dhātu={upa!r} gaṇa={gana} artha={artha!r}")
        if ok:
            print(f"  {G}✓ Pāṇini confirms{X}: '{clean}' is a known root "
                  f"({len(hits)} dhātupāṭha entr{'y' if len(hits)==1 else 'ies'})")
            for h in hits[:3]:
                print(f"      {DIM}{h['code']}  gaṇa {h['gana']} ({h['gana_name']})"
                      f"  {h['dhatu_slp1']}  '{h['artha_slp1']}'{X}")
        else:
            print(f"  {R}✗ no Pāṇinian derivation{X} for {clean!r} — proposal rejected")

    def sandhi_join(self, a: str, b: str):
        pred = self._gen(f"<sandhi>{a}<sep>{b}")
        gold, cat = self.sandhi.join(a, b)[0]
        gold = gold.replace(" ", "")
        mark = f"{G}match{X}" if pred.strip() == gold else f"{R}differs{X}"
        print(f"  {B}model{X}: {a} + {b} → {pred!r}")
        print(f"  {DIM}rule ({cat}): → {gold!r}  [{mark}]{X}")

    def seg(self, text: str):
        pred = self._gen(f"<seg>{text}")
        print(f"  {B}model segments{X}: {text} → {pred}")

    def meter(self, line: str):
        scan = self.chandas.identify(line)
        w = rules.syllable_weights(line)
        print(f"  {B}chandas{X}: {line}")
        print(f"      weights = {w}  ({len(w)} syllables)")
        if scan:
            top = scan[0]
            print(f"      best meter: {top.get('name_iast', top)}  "
                  f"(distance {top.get('distance','?')})")
        pred = self._gen(f"<Candas><wt>{w}")
        print(f"      {DIM}model guess from L/G: {pred}{X}")


def showcase(e: Engine):
    print(f"\n{'='*66}\n  Sanskrit engine — trained GPT (val_bpb={e.val_bpb:.3f}) + Pāṇini\n{'='*66}")
    print(f"\n{B}## MORPHOLOGY — neural proposes, Pāṇini disposes{X}")
    for r in ["gam", "BU", "kf", "vad", "zzzq"]:  # last is a nonsense root
        print(f"\n>>> morph {r}")
        e.morph(r)
    print(f"\n{B}## SANDHI{X}")
    for a, b in [("rAma", "asti"), ("tat", "hitam"), ("deva", "indra"), ("gaNgA", "udakam")]:
        print(f"\n>>> sandhi {a} {b}")
        e.sandhi_join(a, b)
    print(f"\n{B}## SEGMENTATION{X}")
    for t in ["rAmo'sti", "tapasvI", "sUryodayaH"]:
        print(f"\n>>> seg {t}")
        e.seg(t)
    print(f"\n{B}## CHANDAS (meter){X}")
    for line in ["rAmAya", "vAgarTAviva"]:
        print(f"\n>>> meter {line}")
        e.meter(line)
    print()


def repl(e: Engine):
    print("commands: morph <root> | sandhi <a> <b> | seg <text> | meter <line> | quit")
    while True:
        try:
            line = input(f"{B}sanskrit>{X} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line or line in ("quit", "exit", "q"):
            break
        parts = line.split()
        cmd, args = parts[0], parts[1:]
        try:
            if cmd == "morph" and args:
                e.morph(args[0])
            elif cmd == "sandhi" and len(args) >= 2:
                e.sandhi_join(args[0], args[1])
            elif cmd == "seg" and args:
                e.seg(args[0])
            elif cmd == "meter" and args:
                e.meter(args[0])
            else:
                print("  ?  usage: morph gam | sandhi rAma asti | seg rAmo'sti | meter rAmAya")
        except Exception as ex:
            print(f"  error: {ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repl", action="store_true")
    args = ap.parse_args()
    e = Engine()
    (repl if args.repl else showcase)(e)


if __name__ == "__main__":
    main()

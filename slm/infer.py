"""slm/infer.py — structured inference API (returns dicts, not prints).

Shared by demo.py (CLI) and serve.py (web). Wraps the trained GPT + the symbolic
rule engine so every result carries both the model's proposal and Panini's
verdict.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from slm import rules
from slm.model import GPT, GPTConfig
from slm.tokenizer import SLP1Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
_DHATU_RE = re.compile(r"<dhAtu>([^<]*)")
_GANA_RE = re.compile(r"<gaRa>([^<]*)")
_ARTHA_RE = re.compile(r"<artha>([^<]*)")
_METER_RE = re.compile(r"<meter>([^<]*)")


class Inference:
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
    def _gen(self, src: str, max_new: int = 64) -> str:
        ids = [self.tok.bos_id] + self.tok.encode(src) + [self.tok.sep_id]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(x, max_new_tokens=max_new, eos_id=self.tok.eos_id,
                                  temperature=1e-6, top_k=1)
        gen = out[0, len(ids):].tolist()
        if self.tok.eos_id in gen:
            gen = gen[:gen.index(self.tok.eos_id)]
        return self.tok.decode(gen, skip_special=False)

    @torch.no_grad()
    def logprob(self, src: str, tgt: str) -> float:
        """Scoring API (spec §4): mean per-token log-prob of `tgt` given `src`
        under teacher forcing. Used to RANK symbolically-licensed candidates
        (segmentation paths, verified analyses) — the model as ranker, not
        generator. Length-normalized so candidates of different lengths compare."""
        src_ids = self.tok.encode(src)
        tgt_ids = self.tok.encode(tgt) + [self.tok.eos_id]
        ids = [self.tok.bos_id] + src_ids + [self.tok.sep_id] + tgt_ids
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)
        logits, _ = self.model(x)
        logp = torch.log_softmax(logits[0], dim=-1)
        start = len(src_ids) + 1  # first position whose target is a tgt token
        tot, n = 0.0, 0
        for t in range(start, y.size(1)):
            tot += logp[t, y[0, t]].item()
            n += 1
        return tot / max(1, n)

    def morph(self, root: str) -> dict:
        raw = self._gen(f"<morph>{root}")
        m, gn, ar = _DHATU_RE.search(raw), _GANA_RE.search(raw), _ARTHA_RE.search(raw)
        upa = m.group(1) if m else ""
        clean = rules.strip_anubandhas(upa) if upa else ""
        hits = self.kosha.lookup(clean) if clean else []
        return {
            "task": "morph", "input": root, "raw": raw,
            "proposal": {"dhatu": upa, "gana": gn.group(1) if gn else "",
                         "artha": ar.group(1) if ar else ""},
            "verified": bool(hits),
            "clean_root": clean,
            "entries": [{"code": h["code"], "gana": h["gana"],
                         "gana_name": h["gana_name"], "dhatu": h["dhatu_slp1"],
                         "artha": h["artha_slp1"]} for h in hits[:5]],
        }

    def sandhi_join(self, a: str, b: str) -> dict:
        pred = self._gen(f"<sandhi>{a}<sep>{b}").strip()
        gold, cat = self.sandhi.join(a, b)[0]
        gold = gold.replace(" ", "")
        return {"task": "sandhi", "input": [a, b], "model": pred,
                "rule": gold, "category": cat, "match": pred == gold}

    def seg(self, text: str) -> dict:
        pred = self._gen(f"<seg>{text}")
        padas = [p.strip() for p in pred.split("|") if p.strip()]
        return {"task": "seg", "input": text, "model": pred, "padas": padas}

    def meter(self, line: str) -> dict:
        # scan() transliterates IAST/Devanagari/loose-roman -> SLP1, splits
        # padas, and applies the Anustubh rules before the vrtta table.
        result = self.chandas.scan(line)
        weights = "".join(p["weights"] for p in result["padas"])
        best = result["verse_meter_guess"] or {}
        raw = self._gen(f"<Candas><wt>{result['padas'][0]['weights']}") if result["padas"] else ""
        mm = _METER_RE.search(raw)
        return {
            "task": "meter", "input": line, "weights": weights,
            "syllables": len(weights),
            "verse_meter": result["verse_meter"],
            "symbolic_best": {"name": best.get("name_iast", best.get("name_slp1", "")),
                              "distance": best.get("distance")},
            "model_name": mm.group(1) if mm else raw,
        }

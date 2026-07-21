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

    @torch.no_grad()
    def logprob_batch(self, src: str, tgts: list[str]) -> list[float]:
        """Batched equivalent of `logprob` for a shared `src` and many `tgts`.

        Encodes `src` once and runs a SINGLE padded forward over all candidates
        instead of one teacher-forced pass per candidate — the per-candidate
        mean log-probs are identical (within fp tolerance) to
        ``[logprob(src, t) for t in tgts]``. Used by `seg_dp` to rescore k-best
        DP paths in one shot. Right-padding is safe because attention is causal
        (a real position never attends to a later pad), and padded target
        positions are masked out so they never enter any sequence's mean."""
        if not tgts:
            return []
        src_ids = self.tok.encode(src)
        prefix = [self.tok.bos_id] + src_ids + [self.tok.sep_id]
        start = len(src_ids) + 1  # shared across all: first tgt-target position
        # full_i = prefix + encode(tgt_i) + [eos]; score positions [start, real_len_i - 2]
        fulls = [prefix + self.tok.encode(t) + [self.tok.eos_id] for t in tgts]
        lens = [len(f) for f in fulls]
        max_len = max(lens)
        pad = self.tok.pad_id
        batch = [f + [pad] * (max_len - len(f)) for f in fulls]
        seq = torch.tensor(batch, dtype=torch.long, device=self.device)
        x, y = seq[:, :-1], seq[:, 1:]
        logits, _ = self.model(x)
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(2, y.unsqueeze(-1)).squeeze(-1)  # (B, T)
        T = y.size(1)
        pos = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
        # last scored y-index for seq i is real_len_i - 2 (predicts its eos)
        last = torch.tensor([l - 2 for l in lens], device=self.device).unsqueeze(1)
        mask = (pos >= start) & (pos <= last)
        tot = (tok_lp * mask).sum(dim=1)
        n = mask.sum(dim=1).clamp(min=1)
        return (tot / n).tolist()

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

    @torch.no_grad()
    def seg_constrained(self, text: str) -> dict:
        """Copy-constrained segmentation decode (spec §4 hard guarantee,
        jm8.9): the decoder must replay the input's characters verbatim and
        may only INSERT ' | ' boundary markers between them, so the output
        always concatenates back to the input — re-derivability 100% by
        construction, killing the ~30% hallucination rate of free decode.

        At each position the model's next-token distribution is consulted
        only to compare P(' ') (opening a boundary) against P(next input
        char) (continuing the word); everything else is masked. After a
        boundary, '|' and ' ' are forced and an immediate second boundary is
        forbidden (no empty padas).

        Deliberately NOT restricted to sandhi-lattice-licensed junctures:
        the A/B ablation (evals/eval.py --ab) measured the lattice pool
        ceiling at 6.8% because most real boundaries are zero-change word
        abutments no rewrite rule licenses. Copy-constraint is the licensing
        that matters — every emitted split is re-derivable."""
        tok = self.tok
        prompt = [tok.bos_id] + tok.encode(f"<seg>{text}") + [tok.sep_id]
        space_id, pipe_id = tok.stoi[" "], tok.stoi["|"]
        block = self.model.cfg.block_size
        out: list[int] = []
        ptr, n = 0, len(text)
        last_boundary = True  # no boundary before the first char
        while ptr < n:
            ids = (prompt + out)[-block:]
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits, _ = self.model(x)
            lp = logits[0, -1]
            copy_id = tok.stoi.get(text[ptr], tok.unk_id)
            if not last_boundary and 0 < ptr and lp[space_id] > lp[copy_id]:
                out += [space_id, pipe_id, space_id]
                last_boundary = True
                continue
            out.append(copy_id)
            ptr += 1
            last_boundary = False
        pred = tok.decode(out, skip_special=False)
        padas = [p.strip() for p in pred.split("|") if p.strip()]
        return {"task": "seg", "input": text, "model": pred, "padas": padas,
                "constrained": True}

    @torch.no_grad()
    def seg_dp(self, text: str, k: int = 8) -> dict:
        """Tiered segmentation (jm-improve-5pt): lexicon-anchored sandhi-aware
        DP proposes k-best full-cover paths; the char-LM breaks ties via the
        joint score dp + lm_mean_logprob * len(tgt); if no DP path covers the
        input (OOV text), fall back to the copy-constrained neural decode.
        Both tiers re-derive the surface by construction."""
        if not hasattr(self, "_segdp"):
            from slm.segdp import LexiconSegmenter
            self._segdp = LexiconSegmenter()
        cands = self._segdp.kbest(text, k=k)
        if not cands:
            out = self.seg_constrained(text)
            out["tier"] = "constrained-fallback"
            return out
        # LM always votes (even a single full-cover path can carry the wrong
        # licensed spelling variant once variant edges are generated). All k
        # candidates are rescored in ONE padded batch forward (P1.1) rather than
        # k sequential teacher-forced passes; the joint score and tie-break
        # (first-wins, candidate order) are unchanged.
        if len(cands) > 1:
            tgts = [" | ".join(padas) for _, padas in cands]
            lps = self.logprob_batch(f"<seg>{text}", tgts)
            scores = [cands[i][0] + lps[i] * len(tgts[i]) for i in range(len(cands))]
            best = cands[max(range(len(cands)), key=lambda i: scores[i])][1]
        else:
            best = cands[0][1]
        return {"task": "seg", "input": text, "model": " | ".join(best),
                "padas": best, "constrained": True, "tier": "lexicon-dp"}

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

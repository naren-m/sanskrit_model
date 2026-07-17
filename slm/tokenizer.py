"""Character-level SLP1 tokenizer with control tokens (Path A, section 1).

SLP1 is a 1-char = 1-phoneme ASCII transliteration of Sanskrit, so a character
vocab is the natural unit for sandhi and morphology (both are character
phenomena). Vocab is ~100 ids: the SLP1 alphabet + digits + punctuation +
reserved control/task tokens + T5-style [MASK_k] sentinels.

Encoding is longest-match: multi-char control tokens like ``<dhAtu>`` and
``[MASK_3]`` map to a single id; everything else is one id per character.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# --- SLP1 base alphabet (1 char = 1 phoneme) ---------------------------------
SLP1_VOWELS = list("aAiIuUfFxXeEoO")
SLP1_CONSONANTS = list("kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh")
SLP1_MARKS = list("MH'")          # anusvara, visarga, avagraha
# upadesha accent / anubandha marks that occur in dhatupatha forms (ga\mx~, asa~^):
# \ = anudatta, ^ = svarita (ubhayapada), ~ = anunasika/it-nasal.
SLP1_ACCENT = list("\\^~")
SLP1_EXTRA = list("L|")           # vedic la, pada separator used in targets
DIGITS = list("0123456789")
PUNCT = list(" .,-")               # space, danda-as-period, etc.

# --- Reserved control tokens (single id each) --------------------------------
TASK_TOKENS = ["<seg>", "<morph>", "<sandhi>", "<denoise>", "<Candas>"]
FEATURE_TOKENS = [
    "<dhAtu>", "<gaRa>", "<lakAra>", "<puruza>", "<vacana>", "<viBakti>",
    "<liNga>", "<prayoga>", "<kft>", "<taddhita>", "<artha>", "<meter>", "<wt>",
]
SPECIAL_CORE = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
N_MASK = 32  # T5-style sentinels [MASK_0]..[MASK_31]
MASK_TOKENS = [f"[MASK_{i}]" for i in range(N_MASK)]

# Order matters only for stable ids; kept fixed once a vocab is saved.
CONTROL_TOKENS = SPECIAL_CORE + TASK_TOKENS + FEATURE_TOKENS + MASK_TOKENS

# Regex that matches any multi-char control token (longest first) at a position.
_CTRL_RE = re.compile("|".join(re.escape(t) for t in sorted(CONTROL_TOKENS, key=len, reverse=True)))


class SLP1Tokenizer:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
        self.pad_id = stoi["<pad>"]
        self.bos_id = stoi["<bos>"]
        self.eos_id = stoi["<eos>"]
        self.sep_id = stoi["<sep>"]
        self.unk_id = stoi["<unk>"]

    # -- construction ---------------------------------------------------------
    @classmethod
    def build(cls) -> "SLP1Tokenizer":
        chars = (SLP1_VOWELS + SLP1_CONSONANTS + SLP1_MARKS + SLP1_ACCENT
                 + SLP1_EXTRA + DIGITS + PUNCT)
        # de-dup preserving order
        seen, ordered = set(), []
        for c in chars:
            if c not in seen:
                seen.add(c); ordered.append(c)
        vocab = CONTROL_TOKENS + ordered
        stoi = {tok: i for i, tok in enumerate(vocab)}
        return cls(stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    # -- (de)serialization ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SLP1Tokenizer":
        return cls(json.loads(Path(path).read_text()))

    # -- codec ----------------------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = [self.bos_id] if add_bos else []
        i = 0
        while i < len(text):
            m = _CTRL_RE.match(text, i)
            if m:
                ids.append(self.stoi[m.group(0)])
                i = m.end()
            else:
                ch = text[i]
                ids.append(self.stoi.get(ch, self.unk_id))
                i += 1
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        specials = set(self.stoi[t] for t in SPECIAL_CORE)
        out = []
        for i in ids:
            if skip_special and i in specials:
                continue
            out.append(self.itos.get(i, ""))
        return "".join(out)


if __name__ == "__main__":
    tok = SLP1Tokenizer.build()
    print("vocab_size =", tok.vocab_size)
    s = "<morph>gam<dhAtu>ga\\mx~<gaRa>1[MASK_0]"
    ids = tok.encode(s, add_bos=True, add_eos=True)
    print("sample     =", s)
    print("ids        =", ids)
    print("roundtrip  =", tok.decode(ids, skip_special=True))
    assert tok.decode(tok.encode("<sandhi>rAma<sep>asti")) == "<sandhi>rAma<sep>asti"
    print("roundtrip OK")

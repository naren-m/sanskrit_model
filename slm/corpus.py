"""Real Sanskrit corpus loaders -> SLP1, for the denoise / seg / sandhi tasks.

Two gold sources (siblings of this repo):

  * Ramayanam  (../ramayanam/data/slokas/Slokas/**/*.txt): each line is
    ``kanda::sarga::sloka::<body>`` where <body> interleaves Devanagari padas
    with English glosses. We keep only the Devanagari runs -> an *ordered,
    word-segmented* pada sequence per sloka. Ideal seg supervision.

  * Yoga Sutras (../yoga_sutras/data/yoga_sutras.json): clean Devanagari
    ``content`` per sutra -> continuous (already-sandhied) lines. Ideal denoise.

Everything is transliterated Devanagari -> SLP1 once and cached to
``data/corpus.json`` (spec §2.3: "transliterate -> SLP1").
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT.parent
RAMAYANAM = PROJECTS / "ramayanam" / "data" / "slokas" / "Slokas"
YOGA = PROJECTS / "yoga_sutras" / "data" / "yoga_sutras.json"
CACHE = ROOT / "data" / "corpus.json"

_DEVA_RUN = re.compile(r"[ऀ-ॿ]+")
# valid SLP1 chars for a *clean* pada (no accents/marks needed from corpus text)
_SLP1_OK = set("aAiIuUfFxXeEoOkKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshMHL'")


def _to_slp1(deva: str) -> str:
    return transliterate(deva, sanscript.DEVANAGARI, sanscript.SLP1)


def _clean_pada(p: str) -> str | None:
    p = p.strip("'|")
    if len(p) < 2:
        return None
    if any(c not in _SLP1_OK for c in p):
        return None
    return p


def load_ramayanam_padas(limit_files: int | None = None) -> list[list[str]]:
    """Per-sloka lists of SLP1 padas (Devanagari runs, in order)."""
    files = sorted(glob.glob(str(RAMAYANAM / "**" / "*.txt"), recursive=True))
    if limit_files:
        files = files[:limit_files]
    out: list[list[str]] = []
    for fp in files:
        for line in Path(fp).read_text(errors="ignore").splitlines():
            body = line.split("::")[-1]
            padas = []
            for run in _DEVA_RUN.findall(body):
                p = _clean_pada(_to_slp1(run))
                if p:
                    padas.append(p)
            if len(padas) >= 2:
                out.append(padas)
    return out


def load_yoga_sutras_lines() -> list[str]:
    """Continuous SLP1 lines (already-sandhied sutra text)."""
    data = json.loads(YOGA.read_text())
    lines: list[str] = []
    for sec in data.get("sections", []):
        for blk in sec.get("blocks", []):
            content = blk.get("content", "").strip()
            for run in _DEVA_RUN.findall(content):
                s = _clean_pada(_to_slp1(run))
                if s and len(s) >= 4:
                    lines.append(s)
    return lines


def build_cache(limit_files: int | None = None) -> dict:
    rama = load_ramayanam_padas(limit_files)
    yoga = load_yoga_sutras_lines()
    vocab: dict[str, int] = {}
    for padas in rama:
        for p in padas:
            vocab[p] = vocab.get(p, 0) + 1
    for line in yoga:
        vocab[line] = vocab.get(line, 0) + 1
    payload = {
        "ramayanam_padas": rama,
        "yoga_lines": yoga,
        "vocab": vocab,
        "stats": {
            "ramayanam_slokas": len(rama),
            "ramayanam_pada_tokens": sum(len(p) for p in rama),
            "yoga_lines": len(yoga),
            "unique_padas": len(vocab),
        },
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload, ensure_ascii=False))
    return payload


def load_corpus(rebuild: bool = False, limit_files: int | None = None) -> dict:
    if rebuild or not CACHE.exists():
        return build_cache(limit_files)
    return json.loads(CACHE.read_text())


def top_padas(n: int = 4000, min_count: int = 2) -> list[str]:
    """Frequency-ranked real padas — a corpus-grounded vocabulary for sandhi."""
    corpus = load_corpus()
    items = [(w, c) for w, c in corpus["vocab"].items() if c >= min_count]
    items.sort(key=lambda x: -x[1])
    return [w for w, _ in items[:n]]


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = build_cache(limit_files=lim)
    print("corpus cache:", CACHE)
    for k, v in payload["stats"].items():
        print(f"  {k:22s} {v}")
    print("  sample padas:", top_padas(12))
    print("  sample yoga :", payload["yoga_lines"][:3])

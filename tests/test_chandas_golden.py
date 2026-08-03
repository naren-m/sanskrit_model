"""Golden-set tests for ChandasEngine — real verses, attested meters.

Why a golden set at all: the old chandas self-test (``python -m slm.rules``)
scanned a *synthesised* line built by pasting "kA"/"ka" together to match a
pattern it had just read out of meters-full.csv. That is a tautology — it can
only ever confirm that a string matches itself. It cannot catch a wrong CSV
row, a lost pada boundary, or a missing prosodic rule, because no real verse
ever goes through it.

``data/golden_meters.json`` holds 18 real verses (Ramayana, Gita, Kalidasa,
Bhartrhari, Shankara, stotra literature) with their traditional meters. Each is
independently cross-checked here by expanding the canonical Pingala *gana*
definition into an L/G string — so the golden truth never depends on
meters-full.csv being correct. When the two disagree, that is a CSV bug and
:func:`test_csv_pattern_matches_canonical_ganas` reports it as one.
"""
import json
from pathlib import Path

from slm import rules

_HERE = Path(__file__).resolve().parent
GOLDEN = json.loads((_HERE / "data" / "golden_meters.json").read_text(encoding="utf-8"))
VERSES = GOLDEN["verses"]

#: Pingala's eight trisyllabic ganas + the two single-syllable markers.
GANAS = {
    "ma": "GGG", "na": "LLL", "bha": "GLL", "ya": "LGG",
    "ja": "LGL", "ra": "GLG", "sa": "LLG", "ta": "GGL",
    "ga": "G", "la": "L",
}

_ENGINE = rules.ChandasEngine()
_CSV_PATTERNS = {m["name_iast"]: m["_clean_pattern"] for m in _ENGINE.meters}


def _pattern_from_ganas(spec: str) -> str:
    """'ta ta ja ga ga' -> 'GGLGGLLGLGG'. Independent of meters-full.csv."""
    return "".join(GANAS[g] for g in spec.split())


def _padas(verse: dict) -> list[str]:
    return verse["text"].split("\n")


def _pada_weights(pada: str) -> str:
    return rules.syllable_weights(rules.to_slp1(pada))


def _matches_with_anceps(weights: str, pattern: str) -> bool:
    """Sanskrit prosody treats the final syllable of a pada as *anceps*: a
    short syllable there still scans as guru. Patterns are written with the
    final position guru, so a real verse ending a pada in a short vowel must
    still match."""
    return len(weights) == len(pattern) and weights[:-1] == pattern[:-1]


# --- 1. the golden data itself is internally consistent -------------------

def test_golden_padas_match_canonical_ganas():
    """Every fixed-pattern verse scans to its canonical gana expansion.

    This is the guard on the *dataset*: if a verse's text were mistyped, its
    syllable weights would stop matching the meter it claims, and this fails
    before any engine test can be misled by bad ground truth."""
    bad = []
    for v in VERSES:
        if not v["ganas"]:
            continue  # rule-defined family (anustubh / upajati)
        canon = _pattern_from_ganas(v["ganas"])
        for i, pada in enumerate(_padas(v)):
            w = _pada_weights(pada)
            if not _matches_with_anceps(w, canon):
                bad.append(f"{v['id']} pada{i}: {w} != {canon}")
    assert not bad, "golden text/meter disagreement:\n  " + "\n  ".join(bad)


def test_declared_syllable_count_matches_ganas():
    for v in VERSES:
        if not v["ganas"]:
            continue
        assert len(_pattern_from_ganas(v["ganas"])) == v["syllables"], v["id"]


# --- 2. meters-full.csv agrees with Pingala ------------------------------

def test_csv_pattern_matches_canonical_ganas():
    """meters-full.csv rows referenced by the golden set must equal the gana
    expansion. Catches data bugs (a wrong row is invisible to any test that
    reads its ground truth *from* the CSV)."""
    bad = []
    for v in VERSES:
        if not v["ganas"]:
            continue
        canon = _pattern_from_ganas(v["ganas"])
        csv_pattern = _CSV_PATTERNS.get(v["meter"])
        if csv_pattern is None:
            bad.append(f"{v['meter']}: absent from meters-full.csv")
        elif csv_pattern != canon:
            bad.append(f"{v['meter']}: csv={csv_pattern} ({len(csv_pattern)} syl) "
                       f"!= pingala={canon} ({len(canon)} syl)")
    assert not bad, "meters-full.csv disagrees with Pingala:\n  " + "\n  ".join(bad)


def test_every_samavrtta_row_is_self_consistent():
    """All 124 single-pada rows: pattern == ganas expansion == syllables count,
    and the pattern contains nothing but L and G.

    This catches data rot across the whole table, not just the rows the golden
    verses happen to touch. It is how the pañcacāmara row was found to be
    wrong: it carried ja-ra-ja-ra (12 syllables) under a name whose canonical
    definition is ja-ra-ja-ra-ja-ga (16)."""
    bad = []
    for m in _ENGINE.meters:
        if not m["_samavrtta"]:
            continue  # multi-pada rows: see test_multipada_rows_parse
        pattern = m["_clean_pattern"]
        expanded = "".join(GANAS[g] for g in m["ganas"].split("-") if g)
        if expanded != pattern:
            bad.append(f"{m['name_iast']}: ganas {m['ganas']} -> {expanded} != {pattern}")
        if int(m["syllables"]) != len(pattern):
            bad.append(f"{m['name_iast']}: syllables={m['syllables']} != {len(pattern)}")
        if set(pattern) - {"L", "G"}:
            bad.append(f"{m['name_iast']}: stray chars in pattern {pattern!r}")
    assert not bad, "meters-full.csv row errors:\n  " + "\n  ".join(bad)


def test_multipada_rows_parse_into_clean_pada_patterns():
    """The 21 ardhasama/vishama rows give one L/G string per pada, separated by
    '/'. They must split into >=2 clean L/G segments.

    Before this was handled, ``_clean_pattern`` stripped only the yati '|' and
    left the '/' in place, so e.g. akhyaniki was compared as the 23-character
    string 'GGLGGLLGLGG/LGLGGLLGLGG' — a pattern no verse can ever match.

    Their ``ganas`` and ``syllables`` columns are known-lossy (the trailing
    ganas of each non-final pada are dropped and replaced by a '?'), so those
    two columns are deliberately NOT asserted here."""
    multi = [m for m in _ENGINE.meters if not m["_samavrtta"]]
    assert multi, "expected some ardhasamavrtta rows"
    for m in multi:
        pats = m["_pada_patterns"]
        assert len(pats) >= 2, m["name_iast"]
        for p in pats:
            assert p and not (set(p) - {"L", "G"}), f"{m['name_iast']}: {p!r}"


# --- 3. pada boundaries survive transliteration --------------------------

def test_newlines_survive_to_slp1():
    """A verse typed one pada per line must stay one pada per line through
    to_slp1(), for IAST, loose roman and Devanagari alike. Losing the newline
    silently welds all four padas into one 76-syllable 'pada' that can never
    match anything."""
    for text in ("rāmāya\nrāmabhadrāya",
                 "ramaya\nramabhadraya",
                 "रामाय\nरामभद्राय"):
        assert rules.to_slp1(text).count("\n") == 1, repr(text)


def test_scan_recovers_pada_count():
    for v in VERSES:
        result = _ENGINE.scan(v["text"])
        assert result["pada_count"] == len(_padas(v)), \
            f"{v['id']}: got {result['pada_count']} padas"


# --- 4. the engine names the right meter ---------------------------------

def _named(verse: dict) -> str:
    return _ENGINE.scan(verse["text"])["meter_name"]


def test_engine_identifies_every_golden_verse():
    """The headline metric: name the meter of each of the 18 golden verses."""
    wrong = []
    for v in VERSES:
        got = _named(v)
        if got != v["meter"]:
            wrong.append(f"{v['id']}: expected {v['meter']!r}, got {got!r}")
    assert not wrong, (f"{len(wrong)}/{len(VERSES)} golden verses misidentified:\n  "
                       + "\n  ".join(wrong))


# --- 5. the specific prosodic rules, isolated ----------------------------

def test_pada_final_anceps_is_honoured():
    """BG 8.28 pada a ends in short 'va'; the indravajra pattern ends guru.
    Without anceps this verse — and 13 of the 15 fixed-pattern golden verses —
    can never match."""
    assert _named(next(v for v in VERSES if v["id"] == "bg-8.28")) == "indravajrā"


def test_anustubh_vipula_accepted():
    """Ramayana 1.1.1 pada a has L L L at syllables 5-7 (na-vipula). A
    pathya-only check rejects the epic's own opening verse."""
    r = _ENGINE.scan(next(v for v in VERSES if v["id"] == "ramayana-1.1.1")["text"])
    assert r["meter_name"] == "anuṣṭubh"
    assert "vipulā" in r["verse_meter"], r["verse_meter"]


def test_anustubh_pathya_still_labelled_pathya():
    r = _ENGINE.scan(next(v for v in VERSES if v["id"] == "raghuvamsha-1.1")["text"])
    assert "pathyā" in r["verse_meter"], r["verse_meter"]


def test_upajati_detected_as_mixed_tristubh():
    """Upajati (indravajra/upendravajra mixed per pada) is 49 of the Gita's 55
    non-anustubh verses and has no fixed pattern, so it cannot live in
    meters-full.csv — it needs its own rule."""
    for vid in ("bg-2.22", "kumarasambhava-1.1"):
        r = _ENGINE.scan(next(v for v in VERSES if v["id"] == vid)["text"])
        assert r["meter_name"] == "upajāti", f"{vid}: {r['verse_meter']}"


def test_pure_tristubh_not_called_upajati():
    """All-indravajra and all-upendravajra verses must keep their own names —
    upajati means *mixed*."""
    assert _named(next(v for v in VERSES if v["id"] == "bg-15.5")) == "indravajrā"
    assert _named(next(v for v in VERSES if v["id"] == "bg-11.28")) == "upendravajrā"


def test_arya_jati_identified():
    """Arya fixes no syllable pattern at all — only per-pada mora totals
    (12/18/12/15). No amount of L/G string matching finds one, which is why it
    needs the separate mora test. The two golden aryas have 35 and 32
    syllables respectively and are the same meter."""
    for vid in ("samkhyakarika-1", "samkhyakarika-2"):
        v = next(x for x in VERSES if x["id"] == vid)
        r = _ENGINE.scan(v["text"])
        assert r["meter_name"] == "āryā", f"{vid}: {r['verse_meter']}"
        assert r["meter_detail"] == "jāti"


def test_jati_does_not_shadow_a_vrtta():
    """A mora rule is far looser than a fixed pattern, so the jati test must
    run last. If it ran first it would swallow fixed-pattern verses whose mora
    totals happen to line up — this asserts every non-jati golden verse still
    gets its own name."""
    for v in VERSES:
        if v["meter"] in rules.JATIS:
            continue
        r = _ENGINE.scan(v["text"])
        assert r["meter_detail"] != "jāti", f"{v['id']} swallowed by jati"


def test_nonsense_is_not_identified():
    """Negative control: a line that is not verse must not be handed a meter."""
    r = _ENGINE.scan("ka ka ka ka ka")
    assert r["meter_name"] is None, r["verse_meter"]


def test_empty_and_junk_input_is_not_identified():
    """Empty input must not be handed a meter.

    ``all(...)`` over an empty pāda list is vacuously true, so before this was
    guarded an empty verse matched *every* CSV row and came back as whichever
    one sorted first ('śrī', confidently, at distance 0). Found by smoke-testing
    the /api/meter endpoint with a mis-named query parameter."""
    for junk in ("", "   ", "\n\n", "12345", "..."):
        r = _ENGINE.scan(junk)
        assert r["meter_name"] is None, f"{junk!r} -> {r['verse_meter']}"


def test_irregular_verses_are_honestly_unidentified():
    """Verses in ``golden_meters.json['irregular']`` follow the loose epic
    tristubh licence, not a classical fixed pattern. A strict identifier must
    say so rather than snap them to the nearest name — silently reporting the
    nearest neighbour is exactly the failure mode this whole test file exists
    to prevent."""
    for v in GOLDEN["irregular"]:
        r = _ENGINE.scan(v["text"])
        assert r["meter_name"] is None, f"{v['id']}: claimed {r['verse_meter']}"


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                failed += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)

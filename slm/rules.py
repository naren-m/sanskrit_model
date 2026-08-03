"""rules.py — the "all the rules" symbolic Sanskrit baseline (Path A, Phase 0/1).

A single, dependency-free (stdlib-only) module that answers four questions
about SLP1-encoded Sanskrit text purely from lookup tables + hand-written
algorithms, with no neural network involved:

  1. Phonology  — syllabify a word, score its syllables laghu/guru (chandas).
  2. Sandhi     — join two padas at their junction; propose splits of a
                  sandhied string back into padas.
  3. Dhātu      — look up a verbal root's gaṇa/meaning from the dhātupāṭha.
  4. Chandas    — identify the meter of a whole verse: fixed-pattern vṛttas
                  from the CSV, plus the rule-defined families no table can
                  hold (anuṣṭubh pathyā/vipulā, upajāti, and the moraic jātis).

This is the "Pāṇini disposes" half of the project's "neural proposes, Pāṇini
disposes" philosophy (see path-a-sanskrit-model-spec.md): a trained model may
later *propose* candidate segmentations or analyses, but everything here is
symbolic, deterministic, and independently auditable against the CSV data
files at the repo root. Nothing in this module requires pandas/numpy/torch —
it must import in milliseconds so it can gate every training/eval step.

Data files (SLP1 encoding, repo root, sibling of this ``slm/`` package):
    sandhi-rules-full.csv   ~1468 first+second -> result sandhi rules
    dhatus-full.csv         ~2259 verbal roots (upadesha form + gana + artha)
    dhatus-core.csv         294 of the above, hand-curated with a clean
                             ``core_root`` column (no anubandhas)
    meters-full.csv         145 named meters. 124 are samavrtta (one L/G
                             pattern); the other 21 are ardhasama/vishama and
                             give one pattern per pada, '/'-separated. '|'
                             marks a yati and is not a syllable.

The chandas half is graded against tests/data/golden_meters.json — 20 attested
verses whose ground truth is derived from Pingala's gana definitions, i.e.
independently of meters-full.csv. The moraic (jati) matcher reimplements the
algorithm from the MIT-licensed sanskrit/chandas package
<https://github.com/sanskrit/chandas>, which tests/test_chandas_crossvalidate.py
also uses as an independent second opinion on the whole golden set.

Naming note: constants/classes below (VOWELS, CONSONANTS, SLP1_ALPHABET,
SandhiEngine, DhatuKosha, ChandasEngine) are a fixed public API relied on by
other modules in this project — do not rename without updating call sites.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. SLP1 phonology
# ---------------------------------------------------------------------------

#: Repo root = parent of this ``slm/`` package. Data CSVs live here.
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: SLP1 vowels (ac). Short: a i u f x. Long/diphthong (always guru): A I U F X e E o O.
VOWELS: set[str] = set("aAiIuUfFxXeEoO")

#: Vowels that are inherently heavy (guru) regardless of what follows: the
#: long monophthongs A/I/U/F/X plus the diphthongs e/E/o/O, which are always
#: historically long/bimoraic in classical prosody even though SLP1 gives
#: e and o their own single letters (there is no short e/o in Sanskrit).
LONG_VOWELS: set[str] = set("AIUFXeEoO")

#: SLP1 consonants (hal), the 5x5 varga grid + semivowels + sibilants + h.
#: Matches slm/tokenizer.py's SLP1_CONSONANTS for consistency across the repo.
CONSONANTS: set[str] = set("kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh")

#: Anusvara (nasalization) and visarga (aspiration) -- neither vowel nor a
#: full consonant; each alone after a vowel makes that syllable guru.
_ANUSVARA = "M"
_VISARGA = "H"
NASAL_VISARGA: set[str] = {_ANUSVARA, _VISARGA}

#: Avagraha -- marks elision of an initial short 'a' after o/e in sandhi.
#: Silent: never counted as a phoneme for syllabification/weight purposes.
AVAGRAHA = "'"

#: Ordered list of every valid SLP1 character this module knows about:
#: vowels, then consonants, then anusvara/visarga/avagraha. Order follows
#: traditional varnamala order (vowels, then velar..labial stops+nasal,
#: semivowels, sibilants+h, then the three extra marks).
SLP1_ALPHABET: list[str] = list("aAiIuUfFxXeEoO") + list(
    "kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh"
) + [_ANUSVARA, _VISARGA, AVAGRAHA]


# ---------------------------------------------------------------------------
# 1a. Transliteration -> SLP1  (input normalisation)
#
# The phonology/chandas code below is defined purely over SLP1. Real users
# type IAST (caritaṃ), loose romanisation (charitaṃ), Dravidian-style long
# vowels (ō/ē), or Devanagari (चरितं) -- all of which must be folded to SLP1
# *before* syllabification, or every downstream weight is garbage. to_slp1()
# is that shim. It is deliberately weight-preserving: aspiration (ch vs c) is
# collapsed only where it does not change laghu/guru, and dandas become line
# breaks so a verse typed on one line still splits into padas.
# ---------------------------------------------------------------------------

#: IAST / loose-roman digraphs and diacritics -> SLP1. Longest keys are tried
#: first (see _ROMAN_KEYS) so "kh"/"ai" win over "k"/"a". Loose romanisation
#: (charita for carita) maps ch->C, which is a different phoneme than c but
#: identical in weight, so meter scanning is unaffected.
_ROMAN_TO_SLP1: dict[str, str] = {
    # aspirated stops + diphthongs (2 code points each)
    "kh": "K", "gh": "G", "ch": "C", "jh": "J", "ṭh": "W", "ḍh": "Q",
    "th": "T", "dh": "D", "ph": "P", "bh": "B", "ai": "E", "au": "O",
    # long vowels / vocalic liquids
    "ā": "A", "ī": "I", "ū": "U", "ṛ": "f", "ṝ": "F", "ḷ": "x", "ḹ": "X",
    # Dravidian-script long e/o carry an explicit macron; Sanskrit has only
    # the (always-long) e/o, so both fold to the same SLP1 letter.
    "ē": "e", "ō": "o",
    # sibilants + special nasals
    "ś": "S", "ṣ": "z", "ṅ": "N", "ñ": "Y", "ṇ": "R",
    # retroflex/other single-diacritic consonants
    "ṭ": "w", "ḍ": "q",
    # marks: anusvara (ṃ/ṁ) and visarga (ḥ)
    "ṃ": "M", "ṁ": "M", "ḥ": "H",
    # plain ASCII vowels + consonants (identity where SLP1 agrees)
    "a": "a", "i": "i", "u": "u", "e": "e", "o": "o",
    "k": "k", "g": "g", "c": "c", "j": "j", "t": "t", "d": "d", "n": "n",
    "p": "p", "b": "b", "m": "m", "y": "y", "r": "r", "l": "l", "v": "v",
    "w": "v", "s": "s", "h": "h",
    # avagraha variants -> SLP1 avagraha (silent, ignored in weighting)
    "'": "'", "ʼ": "'", "’": "'", "ऽ": "'",
}
#: match longest romanisation keys first
_ROMAN_KEYS: list[str] = sorted(_ROMAN_TO_SLP1, key=len, reverse=True)

#: Devanagari -> SLP1 building blocks (implicit-'a' expansion done in to_slp1).
_DEVA_INDEP_VOWEL = {
    "अ": "a", "आ": "A", "इ": "i", "ई": "I", "उ": "u", "ऊ": "U",
    "ऋ": "f", "ॠ": "F", "ऌ": "x", "ॡ": "X",
    "ए": "e", "ऐ": "E", "ओ": "o", "औ": "O",
}
_DEVA_MATRA = {
    "ा": "A", "ि": "i", "ी": "I", "ु": "u", "ू": "U", "ृ": "f", "ॄ": "F",
    "ॢ": "x", "े": "e", "ै": "E", "ो": "o", "ौ": "O",
}
_DEVA_CONSONANT = {
    "क": "k", "ख": "K", "ग": "g", "घ": "G", "ङ": "N",
    "च": "c", "छ": "C", "ज": "j", "झ": "J", "ञ": "Y",
    "ट": "w", "ठ": "W", "ड": "q", "ढ": "Q", "ण": "R",
    "त": "t", "थ": "T", "द": "d", "ध": "D", "न": "n",
    "प": "p", "फ": "P", "ब": "b", "भ": "B", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "S", "ष": "z", "स": "s", "ह": "h",
}
_DEVA_VIRAMA = "्"
_DEVA_ANUSVARA = {"ं": "M", "ँ": "M"}  # anusvara, candrabindu
_DEVA_VISARGA = "ः"


def _has_devanagari(text: str) -> bool:
    #: only actual aksharas count -- NOT the danda ।/॥ (U+0964/5), which lives
    #: in the Devanagari block but also terminates romanised verses.
    return any(
        ch in _DEVA_CONSONANT or ch in _DEVA_INDEP_VOWEL or ch in _DEVA_MATRA
        for ch in text
    )


def _devanagari_to_slp1(text: str) -> str:
    out: list[str] = []
    pending_a = False  # a bare consonant carries an implicit short 'a'
    for ch in text:
        if ch in _DEVA_CONSONANT:
            if pending_a:
                out.append("a")
            out.append(_DEVA_CONSONANT[ch])
            pending_a = True
        elif ch == _DEVA_VIRAMA:
            pending_a = False  # virama suppresses the implicit 'a'
        elif ch in _DEVA_MATRA:
            out.append(_DEVA_MATRA[ch])
            pending_a = False
        else:
            if pending_a:  # flush the implicit 'a' before non-consonant marks
                out.append("a")
                pending_a = False
            if ch in _DEVA_INDEP_VOWEL:
                out.append(_DEVA_INDEP_VOWEL[ch])
            elif ch in _DEVA_ANUSVARA:
                out.append("M")
            elif ch == _DEVA_VISARGA:
                out.append("H")
            elif ch == "ऽ":
                out.append("'")
            elif ch in "।॥":
                out.append("\n")
            elif ch == "\n":
                # a literal line break is a pada boundary and must survive:
                # folding it to a space welds all four padas into one line.
                out.append("\n")
            elif ch.isspace():
                out.append(" ")
            # digits, punctuation, unknown marks -> dropped
    if pending_a:
        out.append("a")
    return "".join(out)


_SLP1_SET = set(SLP1_ALPHABET)


def _is_slp1(text: str) -> bool:
    """True when *text* is already SLP1 (all letters valid, at least one
    uppercase SLP1 letter present as a positive signal). Pure-lowercase ASCII
    is left to the roman path, which maps it identically anyway."""
    core = [c for c in text if not (c.isspace() or c in "|।॥")]
    return bool(core) and all(c in _SLP1_SET for c in core) and any(
        c.isupper() for c in core
    )


def _roman_to_slp1(text: str) -> str:
    text = text.lower()
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "।॥|":
            out.append("\n")
            i += 1
            continue
        if ch.isspace():
            # newlines are pada boundaries (see _devanagari_to_slp1) -- keep
            # them; every other whitespace run collapses to a single space.
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        for key in _ROMAN_KEYS:
            if text.startswith(key, i):
                out.append(_ROMAN_TO_SLP1[key])
                i += len(key)
                break
        else:
            # unrecognised (digit, stray punctuation) -> word boundary
            out.append(" ")
            i += 1
    return "".join(out)


def to_slp1(text: str) -> str:
    """Normalise IAST, loose romanisation, or Devanagari input to SLP1.

    Dandas (।/॥/|) become newlines so a one-line verse splits into padas;
    unrecognised characters (digits, punctuation) become spaces. Text that is
    already valid SLP1 passes through unchanged (every SLP1 letter is its own
    key in the roman table). This is a pragmatic prosody-oriented shim, not a
    full reversible transliterator: it preserves syllable weight, not every
    phonemic distinction (loose ``ch`` -> ``C`` is intentional).
    """
    text = unicodedata.normalize("NFC", text)
    if _has_devanagari(text):
        slp1 = _devanagari_to_slp1(text)
    elif _is_slp1(text):
        # already SLP1 -- do NOT lowercase (SLP1 uses case distinctively);
        # only fold dandas to newlines.
        slp1 = "".join("\n" if ch in "|।॥" else ch for ch in text)
    else:
        slp1 = _roman_to_slp1(text)
    # collapse runs of spaces (but keep newlines as pada separators)
    lines = [" ".join(ln.split()) for ln in slp1.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def syllabify(word: str) -> list[str]:
    """Split an SLP1 word into syllables.

    Convention used (deliberately the "heavy coda" convention, chosen because
    it makes :func:`syllable_weights` a trivial per-syllable scan): the
    leading consonant cluster of the word is the onset of syllable 0; every
    other syllable has *no* onset of its own -- instead, ALL consonants (and
    any anusvara/visarga/avagraha) between one vowel and the next are folded
    into the PRECEDING syllable as its coda. The final syllable's coda runs
    to the end of the word. Concretely::

        syllable[0]   = <leading consonants> + vowel[0] + <coda up to vowel[1]>
        syllable[i>0] = vowel[i] + <coda up to vowel[i+1], or to end of word>

    This differs from the "maximal onset" convention taught for onset-heavy
    syllabification of e.g. English, but it is the natural one for Sanskrit
    chandas: a syllable's weight only depends on what immediately follows
    its own vowel, so keeping that material *in* the syllable string makes
    weight-scoring a pure per-syllable operation. Non-phonemic/whitespace
    input is not filtered; callers should pass a single clean word.

    Returns [] for a word with no vowels (e.g. empty string).
    """
    vowel_positions = [i for i, ch in enumerate(word) if ch in VOWELS]
    if not vowel_positions:
        return [word] if word else []

    syllables: list[str] = []
    for idx, vpos in enumerate(vowel_positions):
        end = vowel_positions[idx + 1] if idx + 1 < len(vowel_positions) else len(word)
        start = 0 if idx == 0 else vpos
        syllables.append(word[start:end])
    return syllables


def syllable_weights(word: str) -> str:
    """Return an L/G string, one char per syllable, per standard chandas rules.

    A syllable is Guru (G) if any of:
      - its vowel is long/diphthong (member of LONG_VOWELS), or
      - it is immediately followed by anusvara (M) or visarga (H), or
      - it is followed by 2+ consonants before the next vowel (a "heavy
        cluster" closes the syllable with a consonant, making it heavy by
        position -- this is the sthAna/prayatna-independent positional rule,
        traditionally called guru by sannihita/samyoga).
    Otherwise it is Laghu (L).

    Avagraha (') is silent (marks an elided vowel, not a phoneme) and is
    ignored when counting the following consonant cluster.
    """
    weights = []
    for syl in syllabify(word):
        vpos = next((i for i, ch in enumerate(syl) if ch in VOWELS), None)
        if vpos is None:
            # No vowel at all (stray consonant cluster) -- not a real
            # syllable; skip rather than guess a weight for it.
            continue
        vowel = syl[vpos]
        coda = syl[vpos + 1:].replace(AVAGRAHA, "")

        long_vowel = vowel in LONG_VOWELS
        nasal_or_visarga = bool(coda) and coda[0] in NASAL_VISARGA
        consonant_cluster = sum(1 for ch in coda if ch in CONSONANTS) >= 2

        weights.append("G" if (long_vowel or nasal_or_visarga or consonant_cluster) else "L")
    return "".join(weights)


# ---------------------------------------------------------------------------
# 2. Sandhi engine
# ---------------------------------------------------------------------------

class SandhiEngine:
    """Rule table over sandhi-rules-full.csv: join padas, propose splits.

    Each CSV row means: applying sandhi where a pada ending in ``first_slp1``
    meets a pada starting with ``second_slp1`` yields ``result_slp1`` at the
    junction (the rest of each pada is untouched). ``result_slp1`` sometimes
    contains a literal space -- that is not a bug, it reflects real Sanskrit
    orthographic convention: pure vowel-sandhi (ac-sandhi) is written as one
    continuous word, but visarga/consonant-boundary sandhi (hal-sandhi,
    hal+ac / ac+hal boundary) conventionally keeps the two source words
    visually separate even though their edge sounds changed (e.g. "rAmaH" +
    "gacCati" -> "rAmo gacCati", still two written words). join() therefore
    reproduces that space verbatim when the matched rule's result contains
    one; it is not stripped.

    One row in the data (first_slp1='s', second_slp1='') encodes visarga at
    a pause/utterance-end ("avasAne"), i.e. it does not describe a junction
    between two padas at all. Both join() and split() only operate on true
    two-pada junctions, so rows with an empty second_slp1 are excluded from
    the matching table built here.
    """

    def __init__(self, csv_path: str | Path | None = None):
        path = Path(csv_path) if csv_path else _REPO_ROOT / "sandhi-rules-full.csv"
        with open(path, encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))

        # Junction rules only (see docstring): second_slp1 must be non-empty.
        self.rules: list[dict] = [r for r in all_rows if r["second_slp1"]]

        # Index for split(): bucket rules by the first character of the
        # SPACE-STRIPPED result. result_slp1 embeds a space as a word-boundary
        # marker (e.g. i+a -> "y a" for "ity aham"), but split()'s input is
        # continuous space-free text, so matching must use the stripped form
        # ("ya"); the boundary is recovered from first_slp1/second_slp1, not the
        # space. Without this, the ~1427 spaced rules never match (jm8.5).
        for r in self.rules:
            r["_res_ns"] = r["result_slp1"].replace(" ", "")
        self._by_result_first_char: dict[str, list[dict]] = {}
        for r in self.rules:
            key = r["_res_ns"][0] if r["_res_ns"] else ""
            self._by_result_first_char.setdefault(key, []).append(r)
        # Within each bucket, try longer (more specific) results first --
        # this makes split()'s bounded search surface higher-confidence
        # candidates before it runs out of its node/result budget.
        for bucket in self._by_result_first_char.values():
            bucket.sort(key=lambda r: -len(r["_res_ns"]))

    def join(self, first: str, second: str) -> list[tuple[str, str]]:
        """Apply sandhi at the junction of two padas.

        Finds every rule whose ``first_slp1`` is a suffix of ``first`` and
        whose ``second_slp1`` is a prefix of ``second``, keeps only the
        longest such match(es) (longest-match wins; ``first_slp1`` is 1 or 2
        chars in the data, ``second_slp1`` is 0 or 1 -- both are maximized
        together), and returns one (joined_string, category) pair per
        surviving rule. Some junctions are genuinely optionally-sandhied in
        the grammar (e.g. n+l has two attested outcomes), so more than one
        candidate can come back with equal-length matches; both are real.

        Falls back to plain concatenation tagged "no-sandhi" if no rule matches.
        """
        best_len = -1
        candidates: list[dict] = []
        for rule in self.rules:
            f, s = rule["first_slp1"], rule["second_slp1"]
            if first.endswith(f) and second.startswith(s):
                total = len(f) + len(s)
                if total > best_len:
                    best_len, candidates = total, [rule]
                elif total == best_len:
                    candidates.append(rule)

        if not candidates:
            return [(first + second, "no-sandhi")]

        results: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for rule in candidates:
            f, s = rule["first_slp1"], rule["second_slp1"]
            joined = first[: len(first) - len(f)] + rule["result_slp1"] + second[len(s):]
            key = (joined, rule["category"])
            if key not in seen:
                seen.add(key)
                results.append(key)
        return results

    def split(self, text: str, max_results: int = 20) -> list[list[str]]:
        """Propose segmentations of a sandhied SLP1 string (no spaces) into padas.

        Algorithm (bounded recursive lattice search):

        Walk ``text`` left to right maintaining ``(pos, building)`` where
        ``building`` is the prefix of the *current, not-yet-closed* pada
        assembled so far. At every position two kinds of move are tried:

          1. **Match a sandhi rule.** For every rule whose ``result_slp1``
             equals ``text[pos : pos+len(result_slp1)]``, close the current
             pada as ``building + rule.first_slp1``, open a new pada seeded
             with ``rule.second_slp1``, and recurse from
             ``pos + len(result_slp1)``. This is the literal inverse of
             join(): the rule's ``result_slp1`` is exactly what a real join
             would have written at this spot, so finding it in the text is
             evidence a sandhi happened here.
          2. **Skip.** Consume ``text[pos]`` into ``building`` without
             closing a pada, and recurse from ``pos + 1``. This lets a rule
             match at a *later* position win instead of forcing a split at
             every spot that merely looks like a rule's output (e.g. a bare
             "A" is the result of a common vowel-sandhi rule but also just
             an ordinary long A occurring inside an unsandhied word).

        When ``pos`` reaches the end of ``text``, the accumulated ``building``
        closes the final pada and the completed pada list is a candidate
        split. Recursion always advances ``pos`` (by 1 in the skip move, by
        >=1 -- len(result_slp1) -- in the match move) so it terminates; a
        hard node-expansion budget additionally guards against the branching
        factor (one branch per matching rule, plus the skip branch, at every
        position) blowing up on longer inputs.

        This is a *proposer*, not a certifier (see module docstring): a pada
        is only checked for being non-empty, nothing more -- no dictionary or
        morphological validity check is applied. Candidates are ranked with
        splits that found more sandhi boundaries first (a single-pada
        "found nothing" result, if present, always sorts last), then by
        fewer plain "skip" characters (i.e. splits explained more by actual
        rule matches). Returns at most ``max_results`` candidates.
        """
        NODE_BUDGET = 200_000
        n = len(text)
        found: list[tuple[list[str], int]] = []  # (padas, skip_char_count)
        nodes = [0]

        def rec(pos: int, building: str, padas: list[str], skips: int) -> None:
            if nodes[0] > NODE_BUDGET or len(found) >= max_results * 8:
                return
            nodes[0] += 1

            if pos == n:
                final_padas = padas + [building] if building else padas
                if final_padas:
                    found.append((final_padas, skips))
                return

            ch = text[pos]
            # Skip branch FIRST: depth-first order then reaches low-boundary
            # (mostly-skipped) segmentations before descending into the
            # combinatorial tree of rule-match branches. With match branches
            # first, any input with a spurious early rule match (e.g. the
            # "pf|..." matches at the start of "prApnuyurmAnavAH") filled the
            # found-list cap with over-split garbage and the correct minimal
            # split was never generated at all.
            rec(pos + 1, building + ch, padas, skips + 1)

            for rule in self._by_result_first_char.get(ch, ()):
                res = rule["_res_ns"]  # space-stripped; input text is continuous
                if text[pos:pos + len(res)] != res:
                    continue
                new_pada = building + rule["first_slp1"]
                if not new_pada:
                    continue
                rec(pos + len(res), rule["second_slp1"], padas + [new_pada], skips)

        rec(0, "", [], 0)

        # Rank: FEWER boundaries (fewer padas) first — minimal splits are the
        # linguistically likely ones, and the downstream Analyzer rescores the
        # whole pool with lexicon coverage anyway, so this ordering only
        # decides which candidates survive the max_results cut. Ties broken by
        # fewer skip chars (more of the split explained by actual rule
        # matches).
        found.sort(key=lambda item: (len(item[0]), item[1]))

        deduped: list[list[str]] = []
        seen_tuples: set[tuple[str, ...]] = set()
        for padas, _skips in found:
            key = tuple(padas)
            if key not in seen_tuples:
                seen_tuples.add(key)
                deduped.append(padas)
            if len(deduped) >= max_results:
                break

        if not deduped:
            deduped = [[text]]
        return deduped


# ---------------------------------------------------------------------------
# 3. Dhatu lookup  (owned by sanskrit_analyzer)
# ---------------------------------------------------------------------------
#
# The Dhatupatha index and the it-marker stripping moved to
# sanskrit_analyzer.dhatu.dhatupatha, which is now the single owner of that
# data — three projects were carrying partial copies. Re-exported here so
# every existing `rules.DhatuKosha()` / `rules.strip_anubandhas()` call site
# in datagen, infer, evals and demo keeps working unchanged.

from sanskrit_analyzer.dhatu.dhatupatha import (  # noqa: E402
    DhatuKosha,
    strip_anubandhas,
)

__all__ = [*globals().get("__all__", []), "DhatuKosha", "strip_anubandhas"]


# ---------------------------------------------------------------------------
# 4. Meter (chandas) identification
# ---------------------------------------------------------------------------

#: Pingala's eight trisyllabic ganas plus the two single-syllable markers.
#: Every fixed pattern in meters-full.csv is a concatenation of these, and the
#: CSV's own ``ganas`` column spells out which -- so the two columns can be
#: cross-checked against each other (see tests/test_chandas_golden.py).
GANAS: dict[str, str] = {
    "ma": "GGG", "na": "LLL", "bha": "GLL", "ya": "LGG",
    "ja": "LGL", "ra": "GLG", "sa": "LLG", "ta": "GGL",
    "ga": "G", "la": "L",
}

#: Fallback pada splitter for a verse typed on ONE line with no dandas: two or
#: more spaces are treated as a pada break. (Single ``|`` dandas never reach
#: here -- to_slp1 has already turned them into newlines.)
_PADA_SPLIT_RE = re.compile(r"\s{2,}")

#: Moraic (*jati*) meters, as mora counts per pada. A guru is 2 matras, a
#: laghu 1. Unlike a vrtta these fix no syllable pattern at all -- only each
#: pada's mora total -- so they cannot live in meters-full.csv, which stores
#: L/G strings. Arya is the one that matters: the whole Samkhyakarika is in it.
#: Counts and the matching algorithm follow the MIT-licensed sanskrit/chandas
#: package <https://github.com/sanskrit/chandas>.
JATIS: dict[str, tuple[int, int, int, int]] = {
    "āryā": (12, 18, 12, 15),
}


class ChandasEngine:
    """Identify Sanskrit verse meters from meters-full.csv (145 named vrttas)
    plus the two rule-defined families that no fixed table can hold: anustubh
    and upajati.

    CSV ``pattern`` syntax:
      ``|``  yati (caesura) marker -- not a syllable, stripped.
      ``/``  pada boundary -- an *ardhasamavrtta* row gives a different L/G
             string per pada (e.g. akhyaniki, pushpitagra). These are split
             into :pydata:`_pada_patterns`; concatenating them, as this class
             used to, produced a string containing a literal ``/`` that could
             never match anything.

    Prosodic rules this class implements beyond raw string matching:

    * **Pada-final anceps.** The last syllable of a pada is *free*: a short
      syllable there scans as guru. 142 of the 145 CSV patterns end in G, so
      without this rule any verse whose pada ends in a short vowel -- e.g.
      Bhagavad Gita 8.28 pada a, ``...tapaHsu caiva`` -- fails to match.
    * **Anustubh pathya/vipula.** See :meth:`_anustubh`.
    * **Upajati.** See :meth:`_upajati`.
    """

    #: Syllables 5-7 of an ODD anustubh pada (a, c). ``LGG`` is the canonical
    #: *pathya*; the four other legal shapes are the *vipulas*, each named for
    #: the gana it forms. Any other shape (sa/ja/ta) is avoided by poets and
    #: is treated here as "not anustubh".
    _ODD_5_7 = {
        "LGG": "pathyā", "LLL": "na-vipulā", "GLL": "bha-vipulā",
        "GGG": "ma-vipulā", "GLG": "ra-vipulā",
    }
    #: Syllables 5-7 of an EVEN anustubh pada (b, d) -- obligatory, no variants.
    _EVEN_5_7 = "LGL"

    def __init__(self, csv_path: str | Path | None = None):
        path = Path(csv_path) if csv_path else _REPO_ROOT / "meters-full.csv"
        with open(path, encoding="utf-8") as f:
            self.meters: list[dict] = list(csv.DictReader(f))
        for m in self.meters:
            pats = [p for p in m["pattern"].replace("|", "").split("/") if p]
            m["_pada_patterns"] = pats
            m["_samavrtta"] = len(pats) == 1
            m["_clean_pattern"] = pats[0]  # back-compat: first pada's pattern

        # Upajati families: two samavrtta meters whose patterns differ ONLY in
        # the first syllable (indravajra/upendravajra; vamsastha/indravamsa).
        # A verse that mixes them pada-by-pada is an upajati -- by definition
        # not a single fixed pattern, so it cannot be a CSV row. Keyed by the
        # shared tail (pattern minus its first syllable).
        families: dict[str, list[str]] = {}
        for m in self.meters:
            if m["_samavrtta"]:
                families.setdefault(m["_clean_pattern"][1:], []).append(m["name_iast"])
        self.upajati_families: dict[str, list[str]] = {
            tail: sorted(names) for tail, names in families.items() if len(names) > 1
        }

    # --- matching primitives ---------------------------------------------

    @staticmethod
    def _matches(weights: str, pattern: str) -> bool:
        """Exact metrical match, honouring pada-final anceps."""
        return bool(weights) and len(weights) == len(pattern) and \
            weights[:-1] == pattern[:-1]

    @staticmethod
    def _distance(weights: str, pattern: str) -> int:
        """Length-penalized Hamming distance, with the pada-final syllable
        treated as anceps when the lengths agree (so a correct verse ending in
        a short syllable scores 0, not 1). When the lengths differ the final
        position is a genuine mismatch and is counted normally."""
        if not weights or not pattern:
            return max(len(weights), len(pattern))
        common = min(len(weights), len(pattern))
        stop = common - 1 if len(weights) == len(pattern) else common
        mismatches = sum(1 for a, b in zip(weights[:stop], pattern[:stop]) if a != b)
        return abs(len(weights) - len(pattern)) + mismatches

    def identify(self, verse_line: str) -> list[dict]:
        """Rank all known fixed-pattern meters by distance to a single pada.

        Input is transliterated to SLP1 first, so IAST or Devanagari padas
        work. For an ardhasamavrtta row (several pada patterns) the row scores
        its best-fitting pada. Exact matches (distance 0, anceps-tolerant)
        sort first. Returns [{name_iast, class, pattern, distance, exact}, ...].

        NOTE: this is a *pada-level* ranker over fixed patterns. It cannot
        recognise the rule-defined families (anustubh, upajati), which need
        the whole verse -- use :meth:`scan`, which checks those first. A
        non-zero best distance means "no exact match", not an identification.
        """
        weights = syllable_weights(to_slp1(verse_line).replace("\n", " "))
        ranked = []
        for m in self.meters:
            d = min(self._distance(weights, p) for p in m["_pada_patterns"])
            ranked.append({
                "name_iast": m["name_iast"],
                "class": m["class"],
                "pattern": m["pattern"],
                "distance": d,
                "exact": d == 0,
            })
        ranked.sort(key=lambda r: (r["distance"], r["name_iast"]))
        return ranked

    # --- rule-defined families -------------------------------------------

    @classmethod
    def _anustubh(cls, padas_w: list[str]) -> str | None:
        """Recognise the anustubh (sloka) family from per-pada L/G strings.

        Anustubh is defined by *rules*, not a fixed pattern: 4 padas (or a
        2-pada half) of 8 syllables. Syllables 1 and 8 are free; the metre
        lives at syllables 5-7:

        * even padas (b, d) are always ``L G L`` -- no licence;
        * odd padas (a, c) are ``L G G`` in the canonical *pathya*, or one of
          the four *vipula* shapes (na ``LLL``, bha ``GLL``, ma ``GGG``,
          ra ``GLG``).

        Returns a label naming the variant, or None if the rules don't hold.
        The previous implementation demanded 5th-laghu + 6th-guru in *every*
        pada, which is the pathya rule only -- it rejected every vipula verse,
        including Ramayana 1.1.1 (``tapaHsvADyAyanirataM``, na-vipula).
        """
        if len(padas_w) not in (2, 4) or any(len(p) != 8 for p in padas_w):
            return None
        variants: list[str] = []
        for i, p in enumerate(padas_w):
            shape = p[4:7]
            if i % 2:  # pada b / d
                if shape != cls._EVEN_5_7:
                    return None
            else:      # pada a / c
                label = cls._ODD_5_7.get(shape)
                if label is None:
                    return None
                variants.append(label)
        vipulas = list(dict.fromkeys(v for v in variants if v != "pathyā"))
        return f"anuṣṭubh ({', '.join(vipulas)})" if vipulas else "anuṣṭubh (pathyā)"

    def _upajati(self, padas_w: list[str]) -> list[str] | None:
        """Recognise an upajati: a verse whose padas mix two meters differing
        only in their first syllable. The classical case is indravajra
        (``GGLGGLLGLGG``) mixed with upendravajra (``LGLGGLLGLGG``), which
        accounts for 49 of the Bhagavad Gita's 55 non-anustubh verses and for
        the opening of the Kumarasambhava. Returns the family's member names,
        or None. A verse whose padas all start with the *same* weight is a
        pure vrtta, not an upajati, and is deliberately rejected here."""
        if len(padas_w) < 2 or len({len(p) for p in padas_w}) != 1:
            return None
        if len({p[0] for p in padas_w}) < 2:
            return None  # unmixed -> let the plain vrtta match name it
        for tail, names in self.upajati_families.items():
            if all(self._matches(p[1:], tail) for p in padas_w):
                return names
        return None

    @staticmethod
    def _jati(scan: str) -> str | None:
        """Recognise a moraic (*jati*) meter from the verse's flat L/G scan.

        A jati fixes no syllable pattern -- only the *mora* total of each pada
        -- so no amount of L/G string matching can find one. Guru = 2 matras,
        laghu = 1: walk the scan accumulating a running total and record every
        prefix sum. The verse is a given jati iff each pada's cumulative mora
        boundary appears among those prefix sums, i.e. the syllables can be cut
        at exactly the right places. Padas b and d may come up one matra short
        (a guru straddling the boundary), so both paths are tried.

        Algorithm from the MIT-licensed sanskrit/chandas package
        <https://github.com/sanskrit/chandas>, reimplemented here.

        Deliberately the LAST thing :meth:`scan` tries: a mora rule is far
        looser than a fixed pattern, so it would shadow real vrtta matches.
        """
        totals: set[int] = set()
        running = 0
        for weight in scan:
            running += 2 if weight == "G" else 1
            totals.add(running)
        for name, (a, b, c, d) in JATIS.items():
            b, c, d = a + b, a + b + c, a + b + c + d
            if a not in totals:
                continue
            if b in totals and c in totals and (d in totals or d - 1 in totals):
                return name
            if b - 1 in totals and c - 1 in totals and (d - 1 in totals or d - 2 in totals):
                return name
        return None

    def _vrtta(self, padas_w: list[str], *, samavrtta: bool) -> dict | None:
        """Name the meter when EVERY pada matches one CSV row. Ardhasamavrtta
        rows (several pada patterns) are matched cyclically, so a 4-pada verse
        checks pattern[0], pattern[1], pattern[0], pattern[1]."""
        if not padas_w:
            return None  # all() is vacuously true on an empty verse, which
            # would hand back whichever row happens to come first in the CSV
        for m in self.meters:
            if m["_samavrtta"] is not samavrtta:
                continue
            pats = m["_pada_patterns"]
            if all(self._matches(w, pats[i % len(pats)])
                   for i, w in enumerate(padas_w)):
                return m
        return None

    # --- whole-verse entry point -----------------------------------------

    def scan(self, verse: str, transliterate: bool = True) -> dict:
        """Identify the meter of a whole verse.

        The verse is transliterated to SLP1 (IAST / loose-roman / Devanagari
        all accepted; dandas and line breaks split padas), then tried in this
        order, most-constrained first:

        1. **anustubh** -- 8-syllable padas obeying the pathya/vipula rules;
        2. **samavrtta** -- every pada matches the same fixed-pattern row of
           meters-full.csv, pada-final anceps allowed;
        3. **upajati** -- the padas mix two meters differing only in their
           first syllable;
        4. **ardhasamavrtta** -- the padas alternate between the two patterns
           of one multi-pada row;
        5. **jati** -- no fixed pattern at all, only per-pada mora totals
           (arya). Tried last because a mora rule is much looser than a
           syllable pattern and would otherwise shadow real vrtta matches.

        Upajati is deliberately tried *before* ardhasamavrtta: an indravajra /
        upendravajra alternation also fits the ardhasama row ``akhyaniki``, but
        the conventional reading of e.g. Kumarasambhava 1.1 is upajati (the
        ardhasama name is reserved for works that keep the alternation
        systematically). ``meter_detail`` names the alternative.

        Only a whole-verse agreement counts: the old implementation named the
        verse after pada 0's best match alone, which called a verse identified
        on the strength of a quarter of its evidence.

        Returns per-pada weights/counts/best_meter plus:
          ``meter_name``   canonical name, or None when nothing matched;
          ``meter_detail`` the variant ("pathyā", "na-vipulā", "vṛtta", ...);
          ``verse_meter``  the human-readable one-liner;
          ``anustubh`` / ``verse_meter_guess``  kept for backward compat.
        """
        if transliterate:
            verse = to_slp1(verse)
        lines = [ln.strip() for ln in verse.strip().splitlines() if ln.strip()]
        if len(lines) <= 1:
            raw = lines[0] if lines else verse.strip()
            parts = [p.strip() for p in _PADA_SPLIT_RE.split(raw) if p.strip()]
            lines = parts if len(parts) > 1 else ([raw] if raw else [])

        padas = []
        for line in lines:
            weights = syllable_weights(to_slp1(line))
            padas.append({
                "text": line,
                "weights": weights,
                "syllable_count": len(weights),
                "best_meter": (self.identify(line) or [None])[0],
            })

        pada_weights = [p["weights"] for p in padas]

        # Anustubh check: flatten all syllables and re-chunk into 8-syl padas
        # (a sloka is often typed as 2 lines of 16, not 4 lines of 8).
        flat = "".join(pada_weights)
        anustubh = None
        if flat and len(flat) % 8 == 0 and len(flat) // 8 in (2, 4):
            chunks = [flat[i:i + 8] for i in range(0, len(flat), 8)]
            anustubh = self._anustubh(chunks)

        meter_name: str | None = None
        meter_detail: str | None = None
        if anustubh:
            meter_name = "anuṣṭubh"
            meter_detail = anustubh[anustubh.index("(") + 1:-1]
            verse_meter = f"{anustubh}  [{len(flat)} syllables]"
        elif (row := self._vrtta(pada_weights, samavrtta=True)) is not None:
            meter_name = row["name_iast"]
            meter_detail = "vṛtta"
            verse_meter = f"{meter_name} (vṛtta, exact)"
        elif (family := self._upajati(pada_weights)) is not None:
            meter_name = "upajāti"
            meter_detail = "/".join(family) + " mix"
            verse_meter = f"upajāti ({meter_detail})"
        elif (row := self._vrtta(pada_weights, samavrtta=False)) is not None:
            meter_name = row["name_iast"]
            meter_detail = "ardhasamavṛtta"
            verse_meter = f"{meter_name} (ardhasamavṛtta, exact)"
        elif (jati := self._jati(flat)) is not None:
            meter_name = jati
            meter_detail = "jāti"
            verse_meter = f"{jati} (jāti, {sum(JATIS[jati])} mātrās)"
        else:
            verse_meter = "unknown (no exact vṛtta match; not anuṣṭubh or a jāti)"

        return {
            "pada_count": len(padas),
            "padas": padas,
            "total_syllables": len(flat),
            "anustubh": anustubh,
            "meter_name": meter_name,
            "meter_detail": meter_detail,
            "verse_meter": verse_meter,
            "verse_meter_guess": padas[0]["best_meter"] if padas else None,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("slm/rules.py self-test — symbolic Sanskrit rules engine")
    print("=" * 70)

    print("\n[Loading engines...]")
    sandhi = SandhiEngine()
    dhatu = DhatuKosha()
    chandas = ChandasEngine()
    print(f"  SandhiEngine: {len(sandhi.rules)} junction rules loaded")
    print(f"  DhatuKosha:   {len(dhatu.entries)} dhatu entries "
          f"({sum(e['curated'] for e in dhatu.entries)} curated), "
          f"{len(dhatu.all_roots())} unique resolved roots")
    print(f"  ChandasEngine: {len(chandas.meters)} named meters")

    print("\n[1] Sandhi join")
    for a, b in [("rAma", "asti"), ("tat", "hitam")]:
        for joined, category in sandhi.join(a, b):
            print(f"    {a!r} + {b!r} -> {joined!r}   ({category})")

    print("\n[2] Sandhi split (proposer, not certifier)")
    for text in ["rAmo'sti", "tadDitam"]:
        candidates = sandhi.split(text)
        print(f"    split({text!r}) -> {len(candidates)} candidate(s), top 5:")
        for c in candidates[:5]:
            print(f"      {c}")

    print("\n[3] Dhatu lookup")
    for root in ["gam", "BU"]:
        entries = dhatu.lookup(root)
        print(f"    lookup({root!r}): {len(entries)} entrie(s)")
        for e in entries[:3]:
            print(f"      {e['dhatu_slp1']!r:12} gana={e['gana']:>2} ({e['gana_name']:8}) "
                  f"artha={e['artha_slp1']!r}  curated={e['curated']}")

    print("\n[4] Syllable weights")
    for word in ["rAmAya", "narendra", "BagavadgIta"]:
        print(f"    syllabify({word!r}) = {syllabify(word)}")
        print(f"    syllable_weights({word!r}) = {syllable_weights(word)!r}")

    print("\n[5] Meter identification (single pada)")
    # NOTE: this used to synthesise a pada by pasting "kA"/"ka" together to
    # match a pattern read out of the CSV, then check that the pattern matched
    # itself -- a tautology that could not fail. Real padas from real texts
    # only; the graded set lives in data/golden_meters.json and is asserted by
    # tests/test_chandas_golden.py.
    for label, pada in [
        ("Meghaduta 1.1a", "kaścit kāntāvirahaguruṇā svādhikārapramattaḥ"),
        ("Bhagavad Gita 8.28a", "vedeṣu yajñeṣu tapaḥsu caiva"),
    ]:
        weights = syllable_weights(to_slp1(pada))
        print(f"    {label}: {weights} ({len(weights)} syllables)")
        for guess in chandas.identify(pada)[:3]:
            print(f"      distance={guess['distance']}  {guess['name_iast']:18} "
                  f"({guess['class']})  pattern={guess['pattern']!r}")

    print("\n[6] Verse scan (whole verse, all four padas must agree)")
    verses = {
        "Ramayana 1.1.1 (anustubh, na-vipula)":
            "tapaḥsvādhyāyanirataṃ\ntapasvī vāgvidāṃ varam\n"
            "nāradaṃ paripapraccha\nvālmīkir munipuṅgavam",
        "Bhagavad Gita 8.28 (indravajra; pada a ends short)":
            "vedeṣu yajñeṣu tapaḥsu caiva\ndāneṣu yatpuṇyaphalaṃ pradiṣṭam\n"
            "atyeti tatsarvamidaṃ viditvā\nyogī paraṃ sthānamupaiti cādyam",
        "Kumarasambhava 1.1 (upajati: mixed indra/upendra)":
            "astyuttarasyāṃ diśi devatātmā\nhimālayo nāma nagādhirājaḥ\n"
            "pūrvāparau toyanidhī vagāhya\nsthitaḥ pṛthivyā iva mānadaṇḍaḥ",
        "Samkhyakarika 2 (arya: moraic, no fixed pattern)":
            "dṛṣṭavadānuśravikaḥ sa hyaviśuddhikṣayātiśayayuktaḥ\n"
            "tadviparītaḥ śreyānvyaktāvyaktajñavijñānāt",
    }
    for label, verse in verses.items():
        result = chandas.scan(verse)
        print(f"    {label}")
        for p in result["padas"]:
            print(f"      [{p['syllable_count']:2}] {p['weights']:22} {p['text']!r}")
        print(f"      => {result['verse_meter']}")

    print("\n" + "=" * 70)
    print("Self-test complete.")
    print("=" * 70)

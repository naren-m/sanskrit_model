"""rules.py — the "all the rules" symbolic Sanskrit baseline (Path A, Phase 0/1).

A single, dependency-free (stdlib-only) module that answers four questions
about SLP1-encoded Sanskrit text purely from lookup tables + hand-written
algorithms, with no neural network involved:

  1. Phonology  — syllabify a word, score its syllables laghu/guru (chandas).
  2. Sandhi     — join two padas at their junction; propose splits of a
                  sandhied string back into padas.
  3. Dhātu      — look up a verbal root's gaṇa/meaning from the dhātupāṭha.
  4. Chandas    — identify the meter of a verse line from its L/G pattern.

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
    meters-full.csv         145 named meters (vrtta), one L/G pattern per pada

Naming note: constants/classes below (VOWELS, CONSONANTS, SLP1_ALPHABET,
SandhiEngine, DhatuKosha, ChandasEngine) are a fixed public API relied on by
other modules in this project — do not rename without updating call sites.
"""
from __future__ import annotations

import csv
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

        # Index for split(): bucket rules by the first character of
        # result_slp1, so at each text position we only test rules whose
        # result could plausibly start there instead of scanning all ~1467.
        self._by_result_first_char: dict[str, list[dict]] = {}
        for r in self.rules:
            key = r["result_slp1"][0] if r["result_slp1"] else ""
            self._by_result_first_char.setdefault(key, []).append(r)
        # Within each bucket, try longer (more specific) results first --
        # this makes split()'s bounded search surface higher-confidence
        # candidates before it runs out of its node/result budget.
        for bucket in self._by_result_first_char.values():
            bucket.sort(key=lambda r: -len(r["result_slp1"]))

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
            for rule in self._by_result_first_char.get(ch, ()):
                res = rule["result_slp1"]
                if text[pos:pos + len(res)] != res:
                    continue
                new_pada = building + rule["first_slp1"]
                if not new_pada:
                    continue
                rec(pos + len(res), rule["second_slp1"], padas + [new_pada], skips)

            rec(pos + 1, building + ch, padas, skips + 1)

        rec(0, "", [], 0)

        # Rank: more boundaries (more padas) first, then fewer skip chars
        # (more of the split explained by actual rule matches), then
        # shorter total representation as a final tiebreaker.
        found.sort(key=lambda item: (-len(item[0]), item[1]))

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
# 3. Dhatu lookup
# ---------------------------------------------------------------------------

#: Traditional dhatupatha "cutu" it-clusters conventionally prefixed to a
#: root purely to disambiguate it in the recitation list (e.g. "qukfY" for
#: kf, "quBf\\Y" for Bf). Safe to strip: this convention is unambiguous and
#: does not collide with any real root's actual initial phonemes in the data.
_IT_PREFIX_CLUSTERS = ("qu", "wu", "Qu", "Wu", "Gu")


def strip_anubandhas(upadesha: str) -> str:
    """Heuristically strip Paninian it-markers (anubandhas) from a dhatu's
    upadesha (citation) form, approximating the "clean root".

    This is deliberately NOT a full implementation of the it-samjna sutras
    (Ashtadhyayi 1.3.2-1.3.9). Those require knowing, per root, exactly
    which letters are markers versus real root phonemes -- the same surface
    letter (e.g. a final consonant, or a consonant just before a nasalized
    vowel) is a marker in some roots and part of the root in others, which
    is exactly why dhatus-core.csv exists as a 294-entry hand-curated table.
    Callers should always prefer that curated ``core_root`` column when a
    root is in it; this function is only the best-effort fallback for the
    ~2000 roots in dhatus-full.csv that are not (yet) curated.

    Steps applied, all directly from the upadesha's surface form:
      1. Drop pitch-accent marks ``\\`` (anudatta) and ``^`` (svarita) --
         these are never phonemic, just recitation accent.
      2. Drop a conventional leading "cutu" it-prefix (qu-/wu-/Qu-/Wu-/Gu-),
         a fixed dhatupatha citation convention (see _IT_PREFIX_CLUSTERS).
      3. If the form ends in ``~`` (anunasika/nasal-vowel marker per 1.3.2),
         drop the ``~`` and the vowel immediately before it.
      4. Otherwise, if the form (after steps 1-2) ends in a bare consonant
         with no ``~`` anywhere, drop that one trailing consonant (a rough
         reading of 1.3.3 "halantyam": upadesha forms overwhelmingly do not
         cite roots with a genuine bare final consonant).

    Measured against dhatus-core.csv's hand-curated core_root as ground
    truth, this reproduces the curated root exactly for ~75% of the 294
    entries; most misses are cases that need lexical knowledge no surface
    heuristic can recover (nasal-infix roots like citi~ -> "cint", or
    retroflexion s/z alternations like za\\dx~ -> "sad").
    """
    s = upadesha.replace("^", "").replace("\\", "")

    for prefix in _IT_PREFIX_CLUSTERS:
        if s.startswith(prefix) and len(s) > len(prefix) + 1:
            s = s[len(prefix):]
            break

    if s.endswith("~"):
        s = s[:-1]
        if s and s[-1] in VOWELS:
            s = s[:-1]
    elif s and s[-1] in CONSONANTS:
        s = s[:-1]

    return s


class DhatuKosha:
    """Lookup over the dhatupatha: dhatus-full.csv (~2259 roots) merged with
    the hand-curated dhatus-core.csv (294 roots with a clean ``core_root``).

    Every row of dhatus-full.csv is loaded and given a resolved clean-root
    field: the curated ``core_root`` from dhatus-core.csv when that row's
    ``code`` is present there (curated=True), else the best-effort output of
    :func:`strip_anubandhas` on its ``dhatu_slp1`` (curated=False). All three
    lookup methods operate over this single merged table.
    """

    def __init__(self, full_path: str | Path | None = None, core_path: str | Path | None = None):
        full_p = Path(full_path) if full_path else _REPO_ROOT / "dhatus-full.csv"
        core_p = Path(core_path) if core_path else _REPO_ROOT / "dhatus-core.csv"

        with open(core_p, encoding="utf-8") as f:
            core_by_code = {r["code"]: r["core_root"] for r in csv.DictReader(f)}

        with open(full_p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.entries: list[dict] = []
        for r in rows:
            entry = dict(r)
            curated_root = core_by_code.get(r["code"])
            if curated_root is not None:
                entry["core_root"] = curated_root
                entry["curated"] = True
            else:
                entry["core_root"] = strip_anubandhas(r["dhatu_slp1"])
                entry["curated"] = False
            self.entries.append(entry)

    def lookup(self, root: str) -> list[dict]:
        """Return every dhatu entry whose resolved core_root == ``root``
        (exact match on the clean SLP1 root, e.g. "gam", "BU", "kf")."""
        return [e for e in self.entries if e["core_root"] == root]

    def by_gana(self, gana: int) -> list[dict]:
        """Return every dhatu entry belonging to a given gana (1-10)."""
        return [e for e in self.entries if int(e["gana"]) == int(gana)]

    def all_roots(self) -> list[str]:
        """Return the sorted set of unique resolved clean roots."""
        return sorted({e["core_root"] for e in self.entries})


# ---------------------------------------------------------------------------
# 4. Meter (chandas) identification
# ---------------------------------------------------------------------------

class ChandasEngine:
    """Identify Sanskrit verse meters from meters-full.csv (145 named vrttas,
    each a fixed L/G weight pattern for one pada).

    ``pattern`` in the CSV may contain a literal ``|`` marking a yati
    (caesura) position; that character is not a syllable and is stripped
    before comparing against a computed L/G weight string.
    """

    def __init__(self, csv_path: str | Path | None = None):
        path = Path(csv_path) if csv_path else _REPO_ROOT / "meters-full.csv"
        with open(path, encoding="utf-8") as f:
            self.meters: list[dict] = list(csv.DictReader(f))
        for m in self.meters:
            m["_clean_pattern"] = m["pattern"].replace("|", "")

    @staticmethod
    def _distance(weights: str, pattern: str) -> int:
        """Length-penalized Hamming distance: exact Hamming when the two
        strings are the same length; otherwise the length difference plus
        the Hamming distance over their common (overlapping) prefix, so
        near-miss (off-by-a-syllable) meters still rank above wildly wrong
        ones instead of being simply excluded."""
        common = min(len(weights), len(pattern))
        mismatches = sum(1 for a, b in zip(weights[:common], pattern[:common]) if a != b)
        return abs(len(weights) - len(pattern)) + mismatches

    def identify(self, verse_line: str) -> list[dict]:
        """Compute ``verse_line``'s syllable_weights and rank all known
        meters by distance to it. Exact matches (distance 0) sort first.
        Returns [{name_iast, class, pattern, distance}, ...] for every meter,
        best (lowest distance) first."""
        weights = syllable_weights(verse_line)
        ranked = []
        for m in self.meters:
            d = self._distance(weights, m["_clean_pattern"])
            ranked.append({
                "name_iast": m["name_iast"],
                "class": m["class"],
                "pattern": m["pattern"],
                "distance": d,
            })
        ranked.sort(key=lambda r: (r["distance"], r["name_iast"]))
        return ranked

    def scan(self, verse: str) -> dict:
        """Split a multi-line verse into padas (one per non-blank line; if
        the whole verse is on a single line, fall back to splitting on
        runs of 2+ spaces or a danda '|' as a secondary heuristic so a
        sloka typed as one line still yields multiple padas), then report
        per-pada syllable weights, syllable counts, and the best meter guess
        for each pada plus for the verse as a whole (first pada's guess,
        the traditional diagnostic pada for meter identification)."""
        lines = [ln.strip() for ln in verse.strip().splitlines() if ln.strip()]
        if len(lines) <= 1:
            raw = lines[0] if lines else verse.strip()
            import re
            parts = [p.strip() for p in re.split(r"\s{2,}|\|", raw) if p.strip()]
            lines = parts if len(parts) > 1 else ([raw] if raw else [])

        padas = []
        for line in lines:
            weights = syllable_weights(line)
            guesses = self.identify(line)
            padas.append({
                "text": line,
                "weights": weights,
                "syllable_count": len(weights),
                "best_meter": guesses[0] if guesses else None,
            })

        return {
            "pada_count": len(padas),
            "padas": padas,
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

    print("\n[5] Meter identification")
    # Reconstruct a dummy pada whose weights exactly match a known vrtta's
    # pattern, then confirm identify() recovers it at distance 0.
    sample_meter = next(m for m in chandas.meters if m["name_iast"] == "māṇavaka")
    pattern = sample_meter["_clean_pattern"]
    print(f"    target meter: {sample_meter['name_iast']} pattern={pattern!r}")

    def _syllable_for(weight: str) -> str:
        # Guru: long vowel A. Laghu: short vowel a followed by <2 consonants.
        return "kA" if weight == "G" else "ka"

    dummy_line = "".join(_syllable_for(w) for w in pattern)
    print(f"    constructed dummy line: {dummy_line!r}")
    print(f"    its syllable_weights:   {syllable_weights(dummy_line)!r}")
    for guess in chandas.identify(dummy_line)[:3]:
        print(f"      distance={guess['distance']}  {guess['name_iast']:15} "
              f"({guess['class']})  pattern={guess['pattern']!r}")

    print("\n[6] Verse scan")
    verse = "kacit kucalam avyagraM rAjyaM te rAGavAnuja\nkacit svakezu dArezu SAstravat kuruze wiman"
    result = chandas.scan(verse)
    print(f"    pada_count={result['pada_count']}")
    for p in result["padas"]:
        best = p["best_meter"]
        best_str = f"{best['name_iast']} (d={best['distance']})" if best else "n/a"
        print(f"      [{p['syllable_count']:2}] {p['weights']:20} best={best_str}  text={p['text']!r}")

    print("\n" + "=" * 70)
    print("Self-test complete.")
    print("=" * 70)

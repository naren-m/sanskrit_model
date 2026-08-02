"""Cross-validate ChandasEngine against the third-party sanskrit/chandas.

`sanskrit/chandas <https://github.com/sanskrit/chandas>`_ (MIT) is an
independent, purely deterministic meter classifier from the sanskrit-programmers
group. It ships its own 83-meter table, so running it over the golden set is a
genuine second opinion: it shares no data and no code with this repo.

It is NOT a dependency — it has no setup.py and cannot be pip/uv installed, and
its ``__init__.py`` still uses a Python 2 implicit relative import. So this
module is opt-in::

    git clone --depth 1 https://github.com/sanskrit/chandas /tmp/chandas
    SANSKRIT_CHANDAS_PATH=/tmp/chandas uv run pytest tests/test_chandas_crossvalidate.py

Without the env var every test here skips, so CI and the default
``uv run pytest`` stay hermetic.

What the comparison found (all reproduced by the tests below):

* **18 of 20 golden verses agree**, once naming conventions are mapped
  (they say ``śloka`` for anuṣṭubh, and spell the upajāti family into the name).
* **Both engines return "no match" for BG 11.32** — independent confirmation
  that the ``irregular`` classification in the golden set is right, and that
  the popular "upajāti" label for that verse is not metrically supportable.
* **Two places where this engine is stricter**, both asserted below as
  deliberate divergences rather than silently tolerated:
  :func:`test_divergence_padanta_anceps` and
  :func:`test_divergence_sloka_wildcard`.
"""
import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from slm import rules

_HERE = Path(__file__).resolve().parent
GOLDEN = json.loads((_HERE / "data" / "golden_meters.json").read_text(encoding="utf-8"))
_CHANDAS_PATH = os.environ.get("SANSKRIT_CHANDAS_PATH")

pytestmark = pytest.mark.skipif(
    not _CHANDAS_PATH,
    reason="set SANSKRIT_CHANDAS_PATH to a clone of github.com/sanskrit/chandas",
)


def _load_classifier():
    """Import their Classifier without executing their Python 2 __init__.

    Substituting an empty package module means ``from classify import
    Classifier`` never runs, while the submodules' own relative imports still
    resolve against ``__path__``. Nothing in the user's checkout is modified.
    """
    root = Path(_CHANDAS_PATH).resolve()
    pkg = types.ModuleType("chandas")
    pkg.__path__ = [str(root / "chandas")]
    sys.modules["chandas"] = pkg
    classifier_cls = importlib.import_module("chandas.classify").Classifier
    return classifier_cls.from_json_file(str(root / "data" / "data.json"))


@pytest.fixture(scope="module")
def theirs():
    return _load_classifier()


@pytest.fixture(scope="module")
def ours():
    return rules.ChandasEngine()


def _their_name(classifier, text):
    result = classifier.classify(rules.to_slp1(text))
    return result.name if result else None


#: Their names for the same meters. They use the synonym 'śloka' for anuṣṭubh
#: and spell the family into the upajāti name; neither is a disagreement.
_ALIASES = {
    "śloka": "anuṣṭubh",
    "upajāti (indravajrā, upendravajrā)": "upajāti",
    "upajāti (indravaṃśā, vaṃśastha)": "upajāti",
}

#: Verses where the two engines genuinely disagree, with the reason. Each has
#: its own focused test below; listing them here keeps the sweep honest instead
#: of loosening its assertion.
_KNOWN_DIVERGENCES = {"bg-8.28", "samkhyakarika-1"}


def _verse(vid):
    for section in ("verses", "irregular"):
        for v in GOLDEN[section]:
            if v["id"] == vid:
                return v
    raise KeyError(vid)


def test_engines_agree_on_the_golden_set(theirs, ours):
    """Modulo naming, the two independent engines agree on every golden verse
    except the two documented divergences."""
    disagreements = []
    for v in GOLDEN["verses"]:
        if v["id"] in _KNOWN_DIVERGENCES:
            continue
        their_raw = _their_name(theirs, v["text"])
        their_name = _ALIASES.get(their_raw, their_raw)
        our_name = ours.scan(v["text"])["meter_name"]
        if their_name != our_name:
            disagreements.append(
                f"{v['id']}: ours={our_name!r} theirs={their_raw!r}")
    assert not disagreements, (
        "undocumented disagreement with sanskrit/chandas:\n  "
        + "\n  ".join(disagreements))


def test_both_engines_reject_the_irregular_verse(theirs, ours):
    """BG 11.32 is popularly called upajāti. Two engines built from unrelated
    tables both decline to name it — which is why the golden set files it under
    'irregular' rather than grading against the popular label."""
    for v in GOLDEN["irregular"]:
        assert _their_name(theirs, v["text"]) is None, v["id"]
        assert ours.scan(v["text"])["meter_name"] is None, v["id"]


def test_divergence_padanta_anceps(theirs, ours):
    """BG 8.28: we say indravajrā, they say upajāti.

    Their ``Vrtta.regex`` applies anceps only to pādas b and d, leaving a and c
    to require the exact final weight. BG 8.28's pāda a ends short, so it fails
    their indravajrā entry and falls through to their upajāti entry, whose
    pattern has a wildcard at both ends.

    We apply anceps at every pāda end (``sarvatra pādānte``), which is what the
    attestation supports: BG 8.28 is one of only three verses the Gita's own
    metrical index lists as *pure* indravajrā, and calling it upajāti asserts a
    mixture that its four identical pāda openings do not contain."""
    v = _verse("bg-8.28")
    assert ours.scan(v["text"])["meter_name"] == "indravajrā"
    assert _their_name(theirs, v["text"]) == "upajāti (indravajrā, upendravajrā)"


def test_divergence_sloka_wildcard(theirs, ours):
    """Samkhyakarika 1: we say āryā, they say śloka.

    Their śloka entry is the all-wildcard pattern ``. . . . . . . .``, which
    matches *any* 32-syllable verse; this āryā happens to have 32 syllables and
    is caught before the jāti test ever runs. Our anuṣṭubh check enforces the
    5-7 pathyā/vipulā rules, so pāda d (L G G in an even pāda) is rejected and
    the verse correctly reaches the mora test.

    Their own engine gets Samkhyakarika *2* right — it has 35 syllables, so the
    wildcard never fires. The bug only shows on the collision."""
    v = _verse("samkhyakarika-1")
    assert ours.scan(v["text"])["meter_name"] == "āryā"
    assert _their_name(theirs, v["text"]) == "śloka"
    # ...and the non-colliding arya agrees in both engines
    v2 = _verse("samkhyakarika-2")
    assert ours.scan(v2["text"])["meter_name"] == "āryā"
    assert _their_name(theirs, v2["text"]) == "āryā"

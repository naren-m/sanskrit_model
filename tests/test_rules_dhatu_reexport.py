"""The Dhatupatha rules now live in sanskrit_analyzer; slm.rules re-exports
them so existing call sites (datagen, infer, evals, demo) keep working."""

from slm import rules


def test_strip_anubandhas_is_the_analyzer_implementation():
    from sanskrit_analyzer.dhatu.dhatupatha import strip_anubandhas

    assert rules.strip_anubandhas is strip_anubandhas


def test_dhatu_kosha_is_the_analyzer_implementation():
    from sanskrit_analyzer.dhatu.dhatupatha import DhatuKosha

    assert rules.DhatuKosha is DhatuKosha


def test_kosha_loads_without_local_csvs():
    """The CSVs are gone from this repo; the index must still build."""
    assert len(rules.DhatuKosha().entries) == 2259


def test_upstream_fixes_are_visible_here():
    assert rules.strip_anubandhas("o~hA\\k") == "hA"
    assert rules.strip_anubandhas("GuRa~") == "GuR"

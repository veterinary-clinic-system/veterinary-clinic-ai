"""(b) Rule-based Vietnamese keyword extraction: RED / GREEN / empty-string cases."""

from app.services.nlp_triage import rule_based_extract


def test_red_keyword_sample_is_classified_as_severity_4():
    result = rule_based_extract("Chó nhà tôi bị ngừng thở, bất tỉnh và chảy máu nhiều")
    assert result.severity == 4
    assert "ngừng thở" in result.matched_keywords
    assert result.disease_groups  # at least one disease group surfaced
    assert 0.0 < result.confidence <= 1.0


def test_green_keyword_sample_is_classified_as_severity_1():
    result = rule_based_extract("Bé mèo nhà em bị ngứa da và đi khập khiễng mấy hôm nay")
    assert result.severity == 1
    assert "ngứa da" in result.matched_keywords
    assert "đi khập khiễng" in result.matched_keywords


def test_empty_string_defaults_to_blue_with_no_keywords():
    result = rule_based_extract("")
    assert result.severity == 0
    assert result.matched_keywords == []
    assert result.disease_groups == []


def test_diacritics_free_input_still_matches():
    """Vietnamese users often type without diacritics on mobile keyboards."""
    result = rule_based_extract("cho nha em bi kho tho nang va co giat lien tuc")
    assert result.severity == 3

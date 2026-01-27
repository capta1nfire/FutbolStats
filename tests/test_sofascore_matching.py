"""Unit tests for Sofascore refs matching: normalization, alias lookup, similarity."""

import pytest

from app.etl.sofascore_provider import (
    normalize_team_name,
    calculate_team_similarity,
    _parse_threshold_overrides,
)
from app.etl.sofascore_aliases import build_alias_index, names_are_aliases


# ---------------------------------------------------------------------------
# normalize_team_name (v2)
# ---------------------------------------------------------------------------

class TestNormalizeTeamName:
    """Tests for normalize_team_name (v2: preserves real/united/city)."""

    def test_preserves_real(self):
        assert normalize_team_name("Real Madrid") == "real madrid"
        assert normalize_team_name("Real Sociedad") == "real sociedad"

    def test_preserves_united_city(self):
        assert normalize_team_name("Manchester United FC") == "manchester united"
        assert normalize_team_name("Manchester City FC") == "manchester city"

    def test_diacritics(self):
        assert normalize_team_name("Atlético Madrid") == "atletico madrid"
        assert normalize_team_name("Bodø/Glimt") == "bodo glimt"

    def test_punctuation_to_space(self):
        assert normalize_team_name("Bodo/Glimt") == "bodo glimt"
        assert normalize_team_name("Paris Saint-Germain") == "paris saint germain"

    def test_strips_org_suffixes(self):
        assert normalize_team_name("FC Barcelona") == "barcelona"
        assert normalize_team_name("Liverpool FC") == "liverpool"
        assert normalize_team_name("AFC Bournemouth") == "bournemouth"

    def test_no_collisions(self):
        n = normalize_team_name
        assert n("Real Madrid") != n("Atlético Madrid")
        assert n("Manchester City") != n("Manchester United")
        assert n("Real Sociedad") != n("Real Madrid")
        assert n("Leicester City") != n("Stoke City")

    def test_empty_and_whitespace(self):
        assert normalize_team_name("") == ""
        assert normalize_team_name("   ") == ""

    def test_saudi_hyphens(self):
        assert normalize_team_name("Al-Hilal") == "al hilal"
        assert normalize_team_name("Al Hilal") == "al hilal"

    def test_turkish_diacritics(self):
        assert normalize_team_name("Beşiktaş") == "besiktas"
        assert normalize_team_name("Fenerbahçe") == "fenerbahce"


# ---------------------------------------------------------------------------
# build_alias_index + names_are_aliases
# ---------------------------------------------------------------------------

class TestAliasIndex:
    """Tests for cross-provider alias lookup."""

    @pytest.fixture(scope="class")
    def index(self):
        return build_alias_index()

    def test_psg_alias(self, index):
        assert names_are_aliases("Paris Saint Germain", "Paris SG", index)
        assert names_are_aliases("PSG", "Paris Saint-Germain", index)

    def test_man_city_alias(self, index):
        assert names_are_aliases("Manchester City", "Man City", index)

    def test_man_united_alias(self, index):
        assert names_are_aliases("Manchester United", "Man United", index)

    def test_nottm_forest(self, index):
        assert names_are_aliases("Nottingham Forest", "Nott'm Forest", index)

    def test_no_false_positive_madrid(self, index):
        assert not names_are_aliases("Real Madrid", "Real Sociedad", index)

    def test_no_false_positive_manchester(self, index):
        assert not names_are_aliases("Manchester City", "Manchester United", index)

    def test_exact_match_no_index_needed(self, index):
        assert names_are_aliases("Liverpool", "Liverpool", index)

    def test_saudi_alias(self, index):
        assert names_are_aliases("Al-Hilal Saudi FC", "Al Hilal", index)
        assert names_are_aliases("Al-Nassr", "Al Nassr", index)

    def test_turkish_alias(self, index):
        assert names_are_aliases("Beşiktaş", "Besiktas", index)
        assert names_are_aliases("Fenerbahçe", "Fenerbahce", index)

    def test_argentina_alias(self, index):
        assert names_are_aliases("Estudiantes L.P.", "Estudiantes de La Plata", index)
        assert names_are_aliases("Talleres Cordoba", "Talleres", index)

    def test_brazil_alias(self, index):
        assert names_are_aliases("Sao Paulo", "São Paulo", index)
        assert names_are_aliases("Atletico-MG", "Atletico Mineiro", index)

    def test_portugal_alias(self, index):
        assert names_are_aliases("FC Porto", "Porto", index)
        assert names_are_aliases("SC Braga", "Sporting Braga", index)

    def test_near_miss_round1_aliases(self, index):
        """Aliases from P1 near-miss log analysis (round 1)."""
        assert names_are_aliases("1899 Hoffenheim", "TSG Hoffenheim", index)
        assert names_are_aliases("Werder Bremen", "SV Werder Bremen", index)
        assert names_are_aliases("Talleres Cordoba", "CA Talleres", index)
        assert names_are_aliases("Gimnasia M.", "Gimnasia y Esgrima Mendoza", index)
        assert names_are_aliases("Ajax", "AFC Ajax", index)
        assert names_are_aliases("FC Copenhagen", "FC København", index)
        assert names_are_aliases("Atletico Paranaense", "Athletico", index)
        assert names_are_aliases("Al-Hazm", "Al-Hazem", index)
        assert names_are_aliases("Belgrano Cordoba", "Club Atlético Belgrano", index)
        assert names_are_aliases("Ferencvarosi TC", "Ferencváros TC", index)
        assert names_are_aliases("Olympiakos Piraeus", "Olympiacos FC", index)
        assert names_are_aliases("Velez Sarsfield", "Vélez Sarsfield", index)
        assert names_are_aliases("Jaguares", "Jaguares de Córdoba", index)
        assert names_are_aliases("La Equidad", "Internacional de Bogotá", index)


# ---------------------------------------------------------------------------
# calculate_team_similarity with alias
# ---------------------------------------------------------------------------

class TestTeamSimilarityWithAlias:
    """Tests for calculate_team_similarity with alias_index."""

    @pytest.fixture(scope="class")
    def index(self):
        return build_alias_index()

    def test_exact_match(self, index):
        assert calculate_team_similarity("Liverpool", "Liverpool", index) == 1.0

    def test_alias_match_score(self, index):
        score = calculate_team_similarity("Man City", "Manchester City", index)
        assert score == 0.95

    def test_substring_match(self, index):
        # "tottenham" contains nothing special, substring should work
        score = calculate_team_similarity("Tottenham", "Tottenham Hotspur")
        assert score == 0.85

    def test_no_alias_falls_to_jaccard(self, index):
        score = calculate_team_similarity("Some Team FC", "Another Team FC", index)
        assert score < 0.85


# ---------------------------------------------------------------------------
# _parse_threshold_overrides
# ---------------------------------------------------------------------------

class TestParseThresholdOverrides:
    def test_empty(self):
        assert _parse_threshold_overrides("") == {}

    def test_single(self):
        assert _parse_threshold_overrides("128:0.70") == {128: 0.70}

    def test_multiple(self):
        result = _parse_threshold_overrides("128:0.70,307:0.65")
        assert result == {128: 0.70, 307: 0.65}

    def test_whitespace(self):
        result = _parse_threshold_overrides(" 128 : 0.70 , 307 : 0.65 ")
        assert result == {128: 0.70, 307: 0.65}

    def test_invalid_skipped(self):
        result = _parse_threshold_overrides("128:0.70,bad,307:0.65")
        assert result == {128: 0.70, 307: 0.65}

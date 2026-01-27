"""Unit tests for shared name normalization module (P2 consolidation)."""

import pytest
from datetime import datetime

from app.etl.name_normalization import normalize_team_name
from app.etl.sofascore_aliases import build_alias_index, names_are_aliases


# ---------------------------------------------------------------------------
# normalize_team_name — shared module (moved from sofascore_provider)
# ---------------------------------------------------------------------------

class TestNormalizeTeamNameShared:
    """Verify shared normalize function behaves identically to v2."""

    def test_preserves_real(self):
        assert normalize_team_name("Real Madrid") == "real madrid"
        assert normalize_team_name("Real Sociedad") == "real sociedad"

    def test_preserves_united_city(self):
        assert normalize_team_name("Manchester United FC") == "manchester united"
        assert normalize_team_name("Manchester City FC") == "manchester city"

    def test_no_collisions_b1_fix(self):
        """Bug B1 fix: 'real' is NOT stripped — no more collisions."""
        n = normalize_team_name
        assert n("Real Madrid") != n("Atlético Madrid")
        assert n("Manchester City") != n("Manchester United")
        assert n("Real Sociedad") != n("Real Madrid")

    def test_nordic_chars(self):
        assert normalize_team_name("Bodø/Glimt") == "bodo glimt"

    def test_diacritics(self):
        assert normalize_team_name("Atlético Madrid") == "atletico madrid"
        assert normalize_team_name("Beşiktaş") == "besiktas"
        assert normalize_team_name("Fenerbahçe") == "fenerbahce"

    def test_strips_org_suffixes(self):
        assert normalize_team_name("FC Barcelona") == "barcelona"
        assert normalize_team_name("Liverpool FC") == "liverpool"
        assert normalize_team_name("AFC Bournemouth") == "bournemouth"

    def test_empty_and_whitespace(self):
        assert normalize_team_name("") == ""
        assert normalize_team_name("   ") == ""

    def test_saudi_hyphens(self):
        assert normalize_team_name("Al-Hilal") == "al hilal"
        assert normalize_team_name("Al Hilal") == "al hilal"


# ---------------------------------------------------------------------------
# compute_match_score with alias_index (P2.4)
# ---------------------------------------------------------------------------

class TestComputeMatchScoreWithAlias:
    """Verify legacy scorer benefits from alias index."""

    @pytest.fixture(scope="class")
    def index(self):
        return build_alias_index()

    def test_alias_boosts_score(self, index):
        """La Equidad vs Internacional de Bogotá should score higher with alias."""
        from app.etl.match_external_refs import compute_match_score

        api = {
            "kickoff_utc": datetime(2026, 1, 28, 20, 0),
            "home_team": "Jaguares",
            "away_team": "La Equidad",
        }
        ext = {
            "kickoff_utc": datetime(2026, 1, 28, 20, 0),
            "home_team": "Jaguares de Córdoba",
            "away_team": "Internacional de Bogotá",
        }

        score_without = compute_match_score(api, ext)
        score_with = compute_match_score(api, ext, alias_index=index)
        assert score_with > score_without

    def test_no_alias_backward_compatible(self):
        """Without alias_index, behavior unchanged."""
        from app.etl.match_external_refs import compute_match_score

        api = {
            "kickoff_utc": datetime(2026, 1, 28, 20, 0),
            "home_team": "Liverpool",
            "away_team": "Arsenal",
        }
        ext = {
            "kickoff_utc": datetime(2026, 1, 28, 20, 0),
            "home_team": "Liverpool",
            "away_team": "Arsenal",
        }
        score = compute_match_score(api, ext)
        assert score >= 0.6  # kickoff(0.4) + home(0.2) + away(0.2)

    def test_real_madrid_no_collision_in_scorer(self):
        """Real Madrid and Real Sociedad must NOT match as same team after B1 fix."""
        from app.etl.match_external_refs import compute_match_score

        api = {
            "kickoff_utc": datetime(2026, 1, 28, 20, 0),
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
        }
        ext = {
            "kickoff_utc": datetime(2026, 1, 28, 20, 0),
            "home_team": "Real Sociedad",
            "away_team": "Barcelona",
        }
        score = compute_match_score(api, ext)
        # Home teams differ ("real madrid" != "real sociedad") — should NOT get 0.2
        # With B1 bug they'd both normalize to "madrid"/"sociedad" → collision
        assert score < 0.8  # kickoff(0.4) + away(0.2) + partial home at best

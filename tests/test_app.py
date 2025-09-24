# tests/test_app.py
import json
import types
import numpy as np
import pandas as pd
import pytest

# Import the module under test as "app"
# Adjust the import path to match your project layout
import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "beta_3.py"  # change if your file name is different
spec = importlib.util.spec_from_file_location("app", str(MODULE_PATH))
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)


# ---------------------------
# Fixtures (tiny canonical data)
# ---------------------------

@pytest.fixture
def teams_df():
    # minimal 3-team table with distinct strengths
    return pd.DataFrame({
        "team_id": [1, 2, 3],
        "name": ["Alpha", "Bravo", "Charlie"],
        "short": ["ALP", "BRA", "CHA"],
        "str_home": [1300, 1200, 1100],
        "str_away": [1280, 1180, 1080],
    })

@pytest.fixture
def fixtures_df():
    # GW 6..10; include blank, single, and double fixtures
    return pd.DataFrame([
        # GW6: team 1 home vs 2
        {"event": 6, "home_id": 1, "away_id": 2, "finished": False, "kickoff_time": "2025-09-01T12:00:00Z"},
        # GW7: team 1 away @3
        {"event": 7, "home_id": 3, "away_id": 1, "finished": False, "kickoff_time": "2025-09-08T12:00:00Z"},
        # GW8: double for team 2 (home vs 3, away @1)
        {"event": 8, "home_id": 2, "away_id": 3, "finished": False, "kickoff_time": "2025-09-15T12:00:00Z"},
        {"event": 8, "home_id": 1, "away_id": 2, "finished": False, "kickoff_time": "2025-09-15T18:00:00Z"},
        # GW9: no matches for team 3 (blank)
        {"event": 9, "home_id": 1, "away_id": 2, "finished": False, "kickoff_time": "2025-09-22T12:00:00Z"},
        # GW10: team 3 home vs 2
        {"event": 10, "home_id": 3, "away_id": 2, "finished": False, "kickoff_time": "2025-09-29T12:00:00Z"},
    ]).astype({"event": int})

@pytest.fixture
def ratings(teams_df):
    # Use your fixed mapping helper to guarantee consistent 1..5 buckets
    # but we can also set explicit values to prove logic
    return {
        1: {"home": 5, "away": 4},  # strong team
        2: {"home": 3, "away": 3},  # medium
        3: {"home": 2, "away": 2},  # weaker
    }


# ---------------------------
# Unit tests: strength mapping
# ---------------------------

def test_strength_to_fixed_cutpoints_basic():
    s = pd.Series([1000, 1040, 1101, 1241, 1400])  # around the default cuts
    cuts = (1040, 1100, 1240, 1340)
    out = app.strength_to_fixed_cutpoints(s, cuts)
    # Expected bins (<=1040)->1, 1041-1100->2, 1101-1240->3, 1241-1340->4, >1340->5
    assert list(out) == [1, 1, 3, 4, 5]


def test_default_ratings_fixed_uses_team_columns(teams_df, monkeypatch):
    # Ensure it reads "str_home/str_away" and returns 1..5 ints per venue
    r = app.default_ratings_fixed(teams_df)
    assert set(r.keys()) == {1, 2, 3}
    for v in r.values():
        assert set(v.keys()) == {"home", "away"}
        assert 1 <= v["home"] <= 5
        assert 1 <= v["away"] <= 5


# ---------------------------
# Unit tests: difficulty math
# ---------------------------

@pytest.mark.parametrize(
    "team_is_home, team_h, team_a, opp_h, opp_a, method, w_team, w_opp, expected_range",
    [
        (True, 5, 4, 3, 3, "Opponent only", 0.0, 1.0, (1, 5)),
        (False, 5, 4, 3, 3, "Team only", 1.0, 0.0, (1, 5)),
        (True, 5, 4, 2, 2, "Team + Opponent", 0.25, 0.75, (1, 5)),
        (False, 5, 4, 5, 5, "Team + Opponent", 0.5, 0.5, (1, 5)),
    ],
)
def test_compute_fixture_difficulty_bounds(team_is_home, team_h, team_a, opp_h, opp_a, method, w_team, w_opp, expected_range):
    d = app.compute_fixture_difficulty(team_is_home, team_h, team_a, opp_h, opp_a, method, w_team, w_opp)
    lo, hi = expected_range
    assert lo <= d <= hi
    # Team-only must invert team strength: stronger team -> easier (lower diff)
    if method == "Team only":
        d_strong = app.compute_fixture_difficulty(team_is_home, 5, 5, opp_h, opp_a, method, 1.0, 0.0)
        d_weak   = app.compute_fixture_difficulty(team_is_home, 1, 1, opp_h, opp_a, method, 1.0, 0.0)
        assert d_strong < d_weak


# ---------------------------
# Unit tests: ticker builder
# ---------------------------

def test_build_ticker_shapes(teams_df, fixtures_df, ratings):
    disp, val = app.build_ticker(
        teams=teams_df,
        fixtures=fixtures_df,
        ratings=ratings,
        gw_start=6,
        gw_len=5,  # 6..10
        visible_team_ids=[1, 2, 3],
        method="Team + Opponent",
        w_team=0.25,
        w_opp=0.75,
    )
    # Column set: Team, 6..10, Total
    expected_cols = ["Team"] + [str(g) for g in range(6, 11)] + ["Total"]
    assert list(disp.columns) == expected_cols
    assert list(val.columns) == expected_cols
    # Same row order and count
    assert len(disp) == len(val) == 3
    # GW with no matches for a team should be '—' and NaN
    # (team 3 has no GW9 in the fixture list)
    row3_disp = disp[disp["Team"] == "CHA"].iloc[0]
    row3_val  = val[val["Team"].isna()].iloc[0]  # 'Team' in val is NaN by design
    assert row3_disp["9"] == "—"
    # GW with no matches for a team should be '—' and NaN
    # (team 3 has no GW9 in the fixture list)
    idx_cha = disp.index[disp["Team"] == "CHA"][0]
    assert disp.at[idx_cha, "9"] == "—"
    assert pd.isna(val.at[idx_cha, "9"])

    # Double GW cells should average diffs in val and join labels in disp
    idx_bra = disp.index[disp["Team"] == "BRA"][0]
    assert "/" in disp.at[idx_bra, "8"]
    assert 1 <= float(val.at[idx_bra, "8"]) <= 5  # averaged difficulty stays in range

    # Double GW cells should average diffs in val and join labels in disp
    row2_disp = disp[disp["Team"] == "BRA"].iloc[0]
    assert "/" in row2_disp["8"]


def test_build_ticker_sorting_by_total(teams_df, fixtures_df, ratings):
    disp, _ = app.build_ticker(
        teams=teams_df,
        fixtures=fixtures_df,
        ratings=ratings,
        gw_start=6,
        gw_len=5,
        visible_team_ids=[1, 2, 3],
        method="Opponent only",  # simpler to reason about ranking
        w_team=0.0,
        w_opp=1.0,
    )
    # Should be sorted easiest (lowest Total) first
    totals = list(disp["Total"])
    assert totals == sorted(totals)


# ---------------------------
# Unit tests: Styler
# ---------------------------

def test_style_fpl_like_returns_styler(teams_df, fixtures_df, ratings):
    disp, val = app.build_ticker(
        teams=teams_df,
        fixtures=fixtures_df,
        ratings=ratings,
        gw_start=6,
        gw_len=2,
        visible_team_ids=[1, 2, 3],
        method="Team + Opponent",
        w_team=0.25,
        w_opp=0.75,
    )
    styler = app.style_fpl_like(disp, val)
    assert hasattr(styler, "to_html")
    html = styler.to_html()
    # Check that known color tokens appear
    for hex_color in app.FPL_FDR_COLORS.values():
        if hex_color in html:
            break
    else:
        pytest.fail("Expected at least one FDR color to appear in styled HTML.")


# ---------------------------
# Integration-ish: load_fpl_data mocking
# ---------------------------

def test_load_fpl_data_mocks_requests(monkeypatch):
    # Ensure no cached result short-circuits the monkeypatch
    try:
        app.st.cache_data.clear()
    except Exception:
        pass

    base = "https://fantasy.premierleague.com/api/"
    calls = []

    class _Resp:
        def __init__(self, payload): self._p = payload
        def json(self): return self._p

    def fake_get(url):
        calls.append(url)
        if url == base + "bootstrap-static/":
            return _Resp({
                "teams": [
                    {"id": 1, "name": "Alpha", "short_name": "ALP",
                     "strength_overall_home": 1300, "strength_overall_away": 1280},
                    {"id": 2, "name": "Bravo", "short_name": "BRA",
                     "strength_overall_home": 1200, "strength_overall_away": 1180},
                ]
            })
        if url == base + "fixtures/":
            return _Resp([
                {"event": 6, "team_h": 1, "team_a": 2, "finished": False,
                 "kickoff_time": "2025-09-01T12:00:00Z"},
                {"event": None, "team_h": 2, "team_a": 1, "finished": False,
                 "kickoff_time": "2025-09-02T12:00:00Z"},
            ])
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(app.requests, "get", fake_get)

    # Call the undecorated function as an extra guard (ok if already cleared)
    fn = getattr(app.load_fpl_data, "__wrapped__", app.load_fpl_data)
    teams_df, fx_df = fn()

    # Assert our mock was actually called
    assert any("bootstrap-static" in u for u in calls)
    assert any("fixtures" in u for u in calls)

    # Schema checks
    assert {"team_id", "short", "str_home", "str_away"}.issubset(teams_df.columns)
    assert {"event", "home_id", "away_id"}.issubset(fx_df.columns)
    assert fx_df["event"].min() == 6  # None filtered out
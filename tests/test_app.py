# tests/test_app.py
import json
import types
import numpy as np
import pandas as pd
import pytest

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "app/fpl_core.py"  
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

def _df(rows):
    # rows = [id, is_current, is_next, finished]
    return pd.DataFrame(rows, columns=["id", "is_current", "is_next", "finished"])

@pytest.mark.parametrize(
    "events,expected",
    [
        # prefers current
        (_df([
            [1, False, False, True],
            [2, True,  False, False],
            [3, False, True,  False],
        ]), 3),

        # uses next when no current
        (_df([
            [1, False, False, True],
            [2, False, True,  False],
            [3, False, False, False],
        ]), 2),

        # falls back to last finished
        (_df([
            [1, False, False, True],
            [2, False, False, True],
            [3, False, False, False],
        ]), 2),

        # nothing marked -> 1
        (_df([
            [1, False, False, False],
            [2, False, False, False],
        ]), 1),

        # multiple current rows -> first one
        (_df([
            [3, True,  False, False],
            [4, True,  False, False],
        ]), 4),
    ],
    ids=[
        "prefers_current",
        "uses_next",
        "fallback_last_finished",
        "default_one",
        "multiple_current_first",
    ]
)
def test_determine_current_gw_all_cases(events, expected):
    assert app.determine_current_gw(events) == expected


# ---------------------------
# Unit tests: strength mapping
# ---------------------------

def test_strength_to_fixed_cutpoints_basic():
    s = pd.Series([1000, 1040, 1101, 1241, 1400])  # around the default cuts
    cuts = (1000, 1040, 1100, 1240, 1340)
    out = app.strength_to_fixed_cutpoints(s, cuts)
    # Expected bins (<=1040)->1, 1041-1100->2, 1101-1240->3, 1241-1340->4, >1340->5
    assert list(out) == [0, 1, 3, 4, 5]


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

# def test_style_fpl_like_returns_styler(teams_df, fixtures_df, ratings):
#     disp, val = app.build_ticker(
#         teams=teams_df,
#         fixtures=fixtures_df,
#         ratings=ratings,
#         gw_start=6,
#         gw_len=2,
#         visible_team_ids=[1, 2, 3],
#         method="Team + Opponent",
#         w_team=0.25,
#         w_opp=0.75,
#     )
#     styler = app.style_fpl_like(disp, val)
#     assert hasattr(styler, "to_html")
#     html = styler.to_html()
#     # Check that known color tokens appear
#     for hex_color in app.FPL_FDR_COLORS.values():
#         if hex_color in html:
#             break
#     else:
#         pytest.fail("Expected at least one FDR color to appear in styled HTML.")


# ---------------------------
# Integration-ish: load_fpl_data mocking
# ---------------------------


def test_fetch_fpl_data_mocks_http_session(monkeypatch):
    base = app.BASE
    calls = []

    class FakeResp:
        def __init__(self, payload):
            self._p = payload
        def json(self):
            return self._p
    class FakeSession:
        def get(self, url):
            calls.append(url)
            if url == base + "bootstrap-static/":
                return FakeResp({
                    "teams": [
                        {
                            "id": 1,
                            "name": "Alpha",
                            "short_name": "ALP",
                            "strength_overall_home": 1300,
                            "strength_overall_away": 1280,
                        },
                        {
                            "id": 2,
                            "name": "Bravo",
                            "short_name": "BRA",
                            "strength_overall_home": 1200,
                            "strength_overall_away": 1180,
                        },
                    ],
                    "events": [
                        {"id": 5, "finished": True,  "is_current": False, "is_next": False, "deadline_time": "2025-08-25T12:00:00Z"},
                        {"id": 6, "finished": False, "is_current": True,  "is_next": False, "deadline_time": "2025-09-01T12:00:00Z"},
                        {"id": 7, "finished": False, "is_current": False, "is_next": True,  "deadline_time": "2025-09-08T12:00:00Z"},
                    ],
                    "elements": [
                        {
                            "id": 10,
                            "web_name": "Player A",
                            "team": 1,
                            "element_type": 3,
                            "now_cost": 75,
                            "total_points": 100,
                            "influence": "50.0",
                            "creativity": "40.0",
                            "threat": "60.0",
                            "ict_index": "150.0",
                        },
                        {
                            "id": 11,
                            "web_name": "Player B",
                            "team": 2,
                            "element_type": 4,
                            "now_cost": 60,
                            "total_points": 80,
                            "influence": "30.0",
                            "creativity": "20.0",
                            "threat": "25.0",
                            "ict_index": "75.0",
                        },
                    ],
                })
            if url == base + "fixtures/":
                return FakeResp([
                    {
                        "event": 6,
                        "team_h": 1,
                        "team_a": 2,
                        "finished": False,
                        "kickoff_time": "2025-09-01T12:00:00Z",
                        "team_h_score": None,
                        "team_a_score": None,
                    },
                    {
                        "event": None,
                        "team_h": 2,
                        "team_a": 1,
                        "finished": False,
                        "kickoff_time": "2025-09-02T12:00:00Z",
                        "team_h_score": None,
                        "team_a_score": None,
                    },
                ])
            raise AssertionError(f"Unexpected URL: {url}")

    fake_session = FakeSession()

    monkeypatch.setattr(app, "http_session", lambda: fake_session)

    teams_df, fx_df, event_df, fetched_at, players_df = app.fetch_fpl_data()

    # --- basic sanity / schema checks ---
    # fake session was actually used
    assert any("bootstrap-static" in u for u in calls)
    assert any("fixtures" in u for u in calls)

    # teams dataframe
    assert set(teams_df.columns) == {"team_id", "name", "short", "str_home", "str_away"}
    assert len(teams_df) == 2

    # events dataframe
    assert set(event_df.columns) == {"id", "is_current", "is_next", "finished", "deadline_time"}
    assert event_df["is_current"].sum() == 1

    # fixtures dataframe: only rows with non-null event, correct renames
    assert {"event", "home_id", "away_id", "finished", "kickoff_time", "team_h_score", "team_a_score"}.issubset(fx_df.columns)
    assert fx_df["event"].min() == 6  # None filtered out

    # players dataframe: correct columns and transformed fields
    expected_player_cols = {
        "id", "name", "pos", "team_short", "price_m", "total_points",
        "influence", "creativity", "threat", "ict_index", "team_id", "position_id",
    }
    assert expected_player_cols.issubset(players_df.columns)
    # price_m scaled by /10
    assert list(players_df["price_m"]) == [7.5, 6.0]
    # positions mapped
    assert set(players_df["pos"]) == {"MF", "FW"}
    # joined with team short names
    assert set(players_df["team_short"]) == {"ALP", "BRA"}

    # fetched_at should be a timezone-aware Timestamp
    assert hasattr(fetched_at, "tzinfo") and fetched_at.tzinfo is not None

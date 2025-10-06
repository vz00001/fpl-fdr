# fpl_core.py
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import requests

# ---------------------------
# Fetch (no Streamlit here)
# ---------------------------

def fetch_fpl_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Fetch teams, fixtures, events from FPL; return dataframes + fetched_at (UTC)."""
    base = "https://fantasy.premierleague.com/api/"

    static = requests.get(base + "bootstrap-static/").json()
    teams_df = pd.DataFrame(static["teams"])[
        ["id", "name", "short_name", "strength_overall_home", "strength_overall_away"]
    ].rename(columns={
        "id": "team_id",
        "short_name": "short",
        "strength_overall_home": "str_home",
        "strength_overall_away": "str_away",
    })

    event_df = pd.DataFrame(static["events"])[
        ["id", "is_current", "is_next", "finished", "deadline_time"]
    ]

    fixtures = requests.get(base + "fixtures/").json()
    fx_df = pd.DataFrame(fixtures)
    fx_df = fx_df.loc[fx_df["event"].notna(), [
        "event", "team_h", "team_a", "finished", "kickoff_time", "team_h_score", "team_a_score"
    ]].rename(columns={"team_h": "home_id", "team_a": "away_id"})
    fx_df["event"] = fx_df["event"].astype(int)

    fetched_at = pd.Timestamp.now(tz="UTC")
    return teams_df, fx_df, event_df, fetched_at


# ---------------------------
# Ratings helpers
# ---------------------------

def strength_to_fixed_cutpoints(series: pd.Series, cuts: Tuple[int, int, int, int]) -> pd.Series:
    c1, c2, c3, c4 = cuts
    bins = [-np.inf, c1, c2, c3, c4, np.inf]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(int)

def default_ratings_fixed(teams: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    cuts = (1040, 1100, 1240, 1340)
    # NOTE: mirrors your current mapping (home <- str_away, away <- str_home)
    home = strength_to_fixed_cutpoints(teams["str_away"], cuts=cuts)
    away = strength_to_fixed_cutpoints(teams["str_home"], cuts=cuts)
    return {int(tid): {"home": int(h), "away": int(a)}
            for tid, h, a in zip(teams["team_id"], home, away)}

def table_ratings_fixed(table: pd.DataFrame, teams: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    cuts = (0, 5, 10, 15)
    # NOTE: mirrors your current mapping (home <- str_away, away <- str_home)
    home = strength_to_fixed_cutpoints(table["Pos"], cuts=cuts)
    away = strength_to_fixed_cutpoints(table["Pos"], cuts=cuts)
    short2id = {r["short"]: int(r["team_id"]) for _, r in teams.iterrows()}
    return {short2id[team_short]: {"home": 6 - int(h) + 1, "away": 6 - int(a)}
            for team_short, h, a in zip(table["Team"], home, away)}

def determine_current_gw(event_df: pd.DataFrame) -> int:
    current = event_df.loc[event_df["is_current"] == True, "id"]
    if not current.empty:
        return int(current.iloc[0])
    nxt = event_df.loc[event_df["is_next"] == True, "id"]
    if not nxt.empty:
        return int(nxt.iloc[0])
    finished = event_df.loc[event_df["finished"] == True, "id"]
    return int(finished.max()) if not finished.empty else 1


# ---------------------------
# PL table builder
# ---------------------------

def build_pl_table(teams_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Build a league table using finished fixtures."""
    table = teams_df[["team_id", "short"]].rename(columns={"short": "Team"}).set_index("team_id")
    for col in ("P","W","D","L","GF","GA","GD","Pts"):
        table[col] = 0

    finished_fixtures = fixtures_df[fixtures_df["finished"] == True]
    for _, match in finished_fixtures.iterrows():
        home_id, away_id = match["home_id"], match["away_id"]
        home_goals = match.get("team_h_score", 0)
        away_goals = match.get("team_a_score", 0)

        table.at[home_id, "P"] += 1
        table.at[away_id, "P"] += 1

        table.at[home_id, "GF"] += home_goals
        table.at[home_id, "GA"] += away_goals
        table.at[away_id, "GF"] += away_goals
        table.at[away_id, "GA"] += home_goals

        if home_goals > away_goals:
            table.at[home_id, "W"] += 1; table.at[away_id, "L"] += 1
            table.at[home_id, "Pts"] += 3
        elif home_goals < away_goals:
            table.at[away_id, "W"] += 1; table.at[home_id, "L"] += 1
            table.at[away_id, "Pts"] += 3
        else:
            table.at[home_id, "D"] += 1; table.at[away_id, "D"] += 1
            table.at[home_id, "Pts"] += 1; table.at[away_id, "Pts"] += 1

    table["GD"] = table["GF"] - table["GA"]
    table = table.sort_values(
        by=["Pts", "GD", "GF", "Team"],
        ascending=[False, False, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    table.insert(0, "Pos", range(1, len(table) + 1))
    return table.reset_index()


# ---------------------------
# Ticker maths
# ---------------------------

def compute_fixture_difficulty(
    team_is_home: bool,
    team_rating_home: int,
    team_rating_away: int,
    opp_rating_home: int,
    opp_rating_away: int,
    method: str,
    w_team: float,
    w_opp: float,
) -> float:
    team_context = team_rating_home if team_is_home else team_rating_away
    opp_context = opp_rating_away if team_is_home else opp_rating_home
    if method == "Opponent only":
        diff = opp_context
    elif method == "Team only":
        diff = 6 - team_context
    else:
        s = max(1e-6, (w_team + w_opp))
        w_t, w_o = w_team / s, w_opp / s
        diff = (w_o * opp_context) + (w_t * (6 - team_context))
    return float(np.clip(diff, 1.0, 5.0))

def build_ticker(
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    ratings: Dict[int, Dict[str, int]],
    gw_start: int,
    gw_len: int,
    visible_team_ids: List[int],
    method: str,
    w_team: float,
    w_opp: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gw_cols = list(range(gw_start, gw_start + gw_len))
    id2short = dict(zip(teams["team_id"], teams["short"]))
    rows, rows_vals = [], []

    for tid in visible_team_ids:
        if tid not in id2short:
            continue
        display_cells = {"Team": id2short[tid], "_tid": tid}
        value_cells   = {"Team": np.nan,         "_tid": tid}
        total = 0.0

        for gw in gw_cols:
            games = fixtures[(fixtures["event"] == gw) & ((fixtures["home_id"] == tid) | (fixtures["away_id"] == tid))]
            if games.empty:
                display_cells[str(gw)] = "—"; value_cells[str(gw)] = np.nan
            else:
                labels, diffs = [], []
                for _, g in games.iterrows():
                    team_home = (g["home_id"] == tid)
                    opp_id = int(g["away_id"] if team_home else g["home_id"])
                    tag = id2short.get(opp_id, "???")
                    label = tag if team_home else tag.lower()
                    d = compute_fixture_difficulty(
                        team_is_home=team_home,
                        team_rating_home=ratings[tid]["home"],
                        team_rating_away=ratings[tid]["away"],
                        opp_rating_home=ratings[opp_id]["home"],
                        opp_rating_away=ratings[opp_id]["away"],
                        method=method, w_team=w_team, w_opp=w_opp,
                    )
                    diffs.append(float(np.clip(d, 1.0, 5.0))); labels.append(label)
                display_cells[str(gw)] = " / ".join(labels)
                value_cells[str(gw)] = float(np.mean(diffs))
                total += sum(diffs)

        display_cells["Total"] = round(total, 2)
        value_cells["Total"]   = float(total)
        rows.append(display_cells); rows_vals.append(value_cells)

    disp_df = pd.DataFrame(rows); val_df = pd.DataFrame(rows_vals)
    if disp_df.empty:
        return disp_df, val_df

    ordered_cols = ["Team"] + [str(g) for g in gw_cols] + ["Total", "_tid"]
    disp_df = disp_df.reindex(columns=ordered_cols)
    val_df  = val_df.reindex(columns=ordered_cols)

    disp_df = disp_df.sort_values("Total", ascending=True, kind="mergesort")
    val_df = val_df.set_index("_tid").loc[disp_df["_tid"]].reset_index()
    disp_df = disp_df.drop(columns=["_tid"]).reset_index(drop=True)
    val_df  = val_df.drop(columns=["_tid"]).reset_index(drop=True)
    return disp_df, val_df


# ---------------------------
# tiny utils you reuse in app
# ---------------------------

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

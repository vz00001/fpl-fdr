import json
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="FPL FDR (Custom Weights)", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    base_url = "https://fantasy.premierleague.com/api/"
    static = requests.get(base_url + "bootstrap-static/").json()
    teams_df = pd.DataFrame(static["teams"])[
        ["id", "name", "short_name", "strength_overall_home", "strength_overall_away"]
    ].rename(columns={
        "id": "team_id",
        "short_name": "short",
        "strength_overall_home": "str_home",
        "strength_overall_away": "str_away"
    })

    fixtures = requests.get(base_url + "fixtures/").json()
    fx_df = pd.DataFrame(fixtures)
    fx_df = fx_df.loc[fx_df["event"].notna(), [
        "event", "team_h", "team_a", "finished", "kickoff_time"
    ]].rename(columns={"team_h": "home_id", "team_a": "away_id"})
    fx_df["event"] = fx_df["event"].astype(int)
    return teams_df, fx_df

def strength_to_rating(series: pd.Series) -> pd.Series:
    ranked = series.rank(method="first")
    buckets = pd.qcut(ranked, 5, labels=[1,2,3,4,5]).astype(int)
    return buckets

def default_ratings(teams: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    home = strength_to_rating(teams["str_home"])
    away = strength_to_rating(teams["str_away"])
    ratings = {}
    # for tid, h, a in zip(teams["team_id"], home, away):
    for tid, h, a in zip(teams["short"], home, away):    
        ratings[tid] = {"home": int(h), "away": int(a)}
    return ratings

# -----------
def compute_fixture_difficulty(
    team_is_home: bool,
    team_rating_home: int,
    team_rating_away: int,
    opp_rating_home: int,
    opp_rating_away: int,
    method: str,
    w_team: float,
    w_opp: float
) -> float:

    team_context = team_rating_home if team_is_home else team_rating_away
    opp_context = opp_rating_away if team_is_home else opp_rating_home

    if method == "Opponent only":
        fdr = opp_context
    elif method == "Team only":
        fdr = 6 - team_context
    else:
        s = max(1e6, (w_team + w_opp))
        w_t, w_o = w_team / s, w_opp / s
        diff = (w_o * opp_context) + (w_t * (6 - team_context))
    # clip to [1, 5] just in case
    return float(np.clip(diff, 1.0, 5.0))        
 
def build_team_row(
    tid: int,
    gw_cols: List[int],
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    ratings: Dict[int, Dict[str, int]],
    method: str,
    w_team: float,
    w_opp: float,
) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Build the ticker row for ONE team across the selected gameweeks.

    Returns:
        display_cells: dict with text labels (opponents like 'MCI (H)')
        value_cells:   dict with numeric difficulty values for styling

    Example (for team 'ARS', 2 GWs):
    display_cells = {
        "Team": "ARS",
        "1": "CHE (H)",
        "2": "—",
        "Total": 3.5
    }
    value_cells = {
        "Team": NaN,
        "1": 3.5,
        "2": NaN,
        "Total": 3.5
    }
    """

    # Build a lookup from team_id → short name, e.g. {1:"ARS",2:"CHE"}
    id2short = dict(zip(teams["team_id"], teams["short"]))

    # Initialize row for this team
    team_short = id2short[tid]
    display_cells = {"Team": team_short}  # what users see
    value_cells = {"Team": np.nan}        # numeric values for coloring
    total = 0.0                           # running sum of fixture difficulties

    # Loop through each GW column
    for gw in gw_cols:
        # Find all fixtures in this GW where this team is either home or away
        games = fixtures[
            (fixtures["event"] == gw) &
            ((fixtures["home_id"] == tid) | (fixtures["away_id"] == tid))
        ]

        if games.empty:
            # No fixture = blank GW
            display_cells[str(gw)] = "—"
            value_cells[str(gw)] = np.nan
        else:
            labels = []  # opponent labels
            diffs = []   # difficulty values

            for _, g in games.iterrows():
                # Are we the home team?
                team_is_home = (g["home_id"] == tid)
                # Find opponent ID
                opp_id = int(g["away_id"] if team_is_home else g["home_id"])
                # Make a label like "CHE (H)" or "LIV (A)"
                label = f"{id2short[opp_id]}" + (" (H)" if team_is_home else " (A)")

                # Compute difficulty using ratings + weights
                d = compute_fixture_difficulty(
                    team_is_home=team_is_home,
                    team_rating_home=ratings[tid]["home"],
                    team_rating_away=ratings[tid]["away"],
                    opp_rating_home=ratings[opp_id]["home"],
                    opp_rating_away=ratings[opp_id]["away"],
                    method=method,
                    w_team=w_team,
                    w_opp=w_opp,
                )

                labels.append(label)
                diffs.append(d)

            # If multiple games (double GW), join labels and average difficulty
            display_cells[str(gw)] = " / ".join(labels)
            value_cells[str(gw)] = float(np.mean(diffs))  # average for styling color
            total += sum(diffs)                           # sum counts for Total

    # Add the total at the end
    display_cells["Total"] = round(total, 2)
    value_cells["Total"] = total

    return display_cells, value_cells

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
    """
    Build the full ticker for all selected teams.

    Returns:
        disp_df : DataFrame with labels (strings like 'CHE (H)')
        val_df  : DataFrame with numeric difficulties (for coloring)

    Example output (2 teams × 2 GWs):

    disp_df =
      Team    1        2   Total
    0  ARS  CHE (H)    —    3.5
    1  CHE  ARS (A)  LIV (H)  7.0

    val_df =
      Team    1     2   Total
    0  ARS  3.5   NaN    3.5
    1  CHE  3.5   3.5    7.0
    """

    # The GW columns we want to show, e.g. [6,7,8,9,10,11]
    gw_cols = list(range(gw_start, gw_start + gw_len))

    rows, rows_vals = [], []

    # Loop over each team we want to display
    for tid in visible_team_ids:
        display_cells, value_cells = build_team_row(
            tid, gw_cols, teams, fixtures, ratings, method, w_team, w_opp
        )
        rows.append(display_cells)
        rows_vals.append(value_cells)

    # Convert to DataFrames
    disp_df = pd.DataFrame(rows)
    val_df = pd.DataFrame(rows_vals)

    # Sort teams by Total difficulty (easiest first)
    disp_df = disp_df.sort_values("Total", ascending=True, kind="mergesort").reset_index(drop=True)
    val_df = val_df.loc[disp_df.index].reset_index(drop=True)

    return disp_df, val_df


teams_df, fixtures_df = load_data()
print(fixtures_df)
print(teams_df)
print(default_ratings(teams_df))
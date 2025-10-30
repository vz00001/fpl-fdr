# fpl_core.py
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
# fpl_core.py
import requests
from requests.adapters import HTTPAdapter, Retry
import streamlit as st

@st.cache_resource
def http_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",)
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "FPL-FDR/1.0 (+streamlit)"})
    return s

# ---------------------------
# Fetching FPL data
# ---------------------------
BASE = "https://fantasy.premierleague.com/api/"
def fetch_fpl_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    """Fetch teams, fixtures, events, players from FPL; return dataframes + fetched_at (UTC)."""

    static = http_session().get(BASE + "bootstrap-static/").json()
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

    players_df = pd.DataFrame(static["elements"])[
        ["id", "web_name", "team", "element_type", "now_cost", "total_points", "influence", "creativity", "threat", "ict_index"]
    ].rename(columns={
        "web_name": "name",
        "team": "team_id",
        "element_type": "position_id",
        "now_cost": "price_m"
    })          
    players_df["price_m"] = players_df["price_m"] / 10.0
    players_df["influence"] = players_df["influence"].astype(float)
    players_df["creativity"] = players_df["creativity"].astype(float)
    players_df["threat"] = players_df["threat"].astype(float)
    players_df["ict_index"] = players_df["ict_index"].astype(float)
    players_df["pos"] = players_df["position_id"].map({1: "GK", 2: "DF", 3: "MF", 4: "FW"})
    players_df = players_df.merge(teams_df[["team_id", "short"]], on="team_id", how="left").rename(columns={"short": "team_short"})
    players_df = players_df[["id", "name", "pos", "team_short", "price_m", "total_points", "influence", "creativity", "threat", "ict_index", "team_id", "position_id"]]

    fixtures = http_session().get(BASE + "fixtures/").json()
    fx_df = pd.DataFrame(fixtures)
    fx_df = fx_df.loc[fx_df["event"].notna(), [
        "event", "team_h", "team_a", "finished", "kickoff_time", "team_h_score", "team_a_score"
    ]].rename(columns={"team_h": "home_id", "team_a": "away_id"})
    fx_df["event"] = fx_df["event"].astype(int)

    fetched_at = pd.Timestamp.now(tz="UTC")

    return teams_df, fx_df, event_df, fetched_at, players_df

@st.cache_data(ttl=3600)
def _fetch_player_history_json(player_id: int) -> dict:
    r = http_session().get().get(f"{BASE}element-summary/{player_id}/", timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def fetch_player_history(player_id: int) -> pd.DataFrame:
    data = _fetch_player_history_json(player_id)
    hist = pd.DataFrame(data.get("history", []))
    if hist.empty:
        return pd.DataFrame(columns=["round","kickoff_time","total_points","influence","creativity","threat","ict_index","minutes"])
    keep = ["round","kickoff_time","total_points","influence","creativity","threat","ict_index","minutes"]
    hist = hist[keep].copy()
    hist["kickoff_time"] = pd.to_datetime(hist["kickoff_time"], utc=True, errors="coerce")
    for c in ["total_points","influence","creativity","threat","ict_index","minutes"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0.0)
    return hist.sort_values("round", ascending=False, kind="mergesort").reset_index(drop=True)


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

def table_ratings_fixed(table: pd.DataFrame, team: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    cuts = (0, 5, 10, 15)
    # NOTE: mirrors your current mapping (home <- str_away, away <- str_home)
    home = strength_to_fixed_cutpoints(table["Pos"], cuts=cuts)
    away = strength_to_fixed_cutpoints(table["Pos"], cuts=cuts)
    # return {int(tid): {"home": int(h), "away": int(a)}
    #         for tid, h, a in zip(table["team_id"], home, away)}
    short2id = {r["short"]: int(r["team_id"]) for _, r in team.iterrows()}
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

        display_cells["Total"] = int(total) if float(total).is_integer() else round(total, 2)
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

# ---------------------------
# ICT index helper  
# ---------------------------


def make_position_filter(selected_pos, players_df: pd.DataFrame) -> pd.Series:
    selected_pos_upper = (selected_pos or "All").upper()
    if selected_pos_upper == "ALL":
        return pd.Series([True] * len(players_df), index=players_df.index)
    else:
        return pd.Series(players_df['pos'].str.upper().eq(selected_pos_upper), index=players_df.index)

def make_price_filter(max_price, players_df: pd.DataFrame) -> pd.Series:
    return pd.Series(players_df['price_m'].le(max_price), index=players_df.index)

def combine_filters(selected_pos, max_price, players_df: pd.DataFrame) -> pd.DataFrame:
    pos_mask = make_position_filter(selected_pos, players_df)
    price_mask = make_price_filter(max_price, players_df)
    final_mask = pos_mask & price_mask
    filtered_players = players_df.loc[final_mask].sort_values(by='ict_index', ascending=False).reset_index(drop=True)

    return filtered_players

def aggregate_last_n(history_df: pd.DataFrame, n: int, *, exclude_zero_min=True) -> Dict[str, float]:
    if history_df.empty:
        return {"total_points":0.0, "influence":0.0, "creativity":0.0, "threat":0.0, "ict_index":0.0}
    df = history_df
    if exclude_zero_min and "minutes" in df.columns:
        df = df[df["minutes"] > 0]
    n = int(max(1, min(n, len(df))))
    recent = df.head(n)
    return {
        "total_points": float(recent["total_points"].sum()),
        "influence": float(recent["influence"].sum()),
        "creativity": float(recent["creativity"].sum()),
        "threat": float(recent["threat"].sum()),
        "ict_index": float(recent["ict_index"].sum()),
    }

@st.cache_data(ttl=3600)
def rolling_ict_for_player(player_id: int, n: int, exclude_zero_min: bool = True) -> Dict[str, float]:
    df = fetch_player_history(player_id)
    if exclude_zero_min and "minutes" in df.columns:
        df = df[df["minutes"] > 0]
    if df.empty:
        return {"total_points":0.0,"influence":0.0,"creativity":0.0,"threat":0.0,"ict_index":0.0}
    n = int(max(1, min(n, len(df))))
    recent = df.head(n)
    return {
        "total_points": float(recent["total_points"].sum()),
        "influence": float(recent["influence"].sum()),
        "creativity": float(recent["creativity"].sum()),
        "threat": float(recent["threat"].sum()),
        "ict_index": float(recent["ict_index"].sum()),
    }

from concurrent.futures import ThreadPoolExecutor, as_completed

def apply_rolling(players_df_slice: pd.DataFrame, n: int) -> pd.DataFrame:
    ids = list(players_df_slice["id"])

    rows: List[Dict[str, float]] = []
    # keep workers modest to be polite to the API
    max_workers = min(8, max(1, len(ids)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(rolling_ict_for_player, pid, n): pid for pid in ids}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                agg = fut.result()
            except Exception:
                # fail-safe: return zeros for this player if anything goes wrong
                agg = {"total_points":0.0,"influence":0.0,"creativity":0.0,"threat":0.0,"ict_index":0.0}
            agg["id"] = pid
            rows.append(agg)

    agg_df = pd.DataFrame(rows)
    for c in ["total_points","influence","creativity","threat","ict_index"]:
        agg_df[c] = pd.to_numeric(agg_df[c], errors="coerce").fillna(0.0)

    base = players_df_slice[["id","name","pos","team_short","price_m"]].copy()
    joined = base.merge(agg_df, on="id", how="left")
    joined[["total_points","influence","creativity","threat","ict_index"]] = \
        joined[["total_points","influence","creativity","threat","ict_index"]].fillna(0.0)

    return joined.sort_values("ict_index", ascending=False, kind="mergesort").reset_index(drop=True)












# fpl-fdr/quick-test.py
teams_df, fixtures_df, event_df, fetched_at, players_df = fetch_fpl_data()
# print(teams_df)
# print(fixtures_df)
# print(players_df.head(30))
print(apply_rolling(combine_filters("MF", 7.5, players_df), 5).head(10))
# print(make_position_filter("FW", players_df).head(30))
# print(make_price_filter(6.5, players_df).head(30))
# print(players_df["team_short"].unique())
# print(players_df["pos"].value_counts())
# print(players_df.describe())
# table_df = build_pl_table(teams_df, fixtures_df)
# print(table_df)
# print(table_ratings_fixed(table_df, teams_df))

# _all = teams_df.sort_values("name")
# disp_df, val_df = build_ticker(
#     teams=teams_df, 
#     fixtures=fixtures_df, 
#     ratings=table_ratings_fixed(table_df, teams_df),
#     gw_start=int(7), 
#     gw_len=int(5),
#     visible_team_ids=list(map(int, _all["team_id"])),
#     method="Team + Opponent", 
#     w_team=0.25, 
#     w_opp=0.75,
# )

# print(disp_df)
# print(val_df)   
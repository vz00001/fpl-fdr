import json
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from pandas.io.formats.style import Styler 

st.set_page_config(page_title="FPL FDR (Custom Weights)", layout="wide")

# ---------------------------
# Data loading (FPL official)
# ---------------------------

@st.cache_data(ttl=3600)
def load_fpl_data():
    """Fetch teams and fixtures from FPL API."""
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

    fixtures = requests.get(base + "fixtures/").json()
    fx_df = pd.DataFrame(fixtures)
    # keep only scheduled fixtures with a gameweek (event)
    fx_df = fx_df.loc[fx_df["event"].notna(), [
        "event", "team_h", "team_a", "finished", "kickoff_time"
    ]].rename(columns={"team_h": "home_id", "team_a": "away_id"})
    fx_df["event"] = fx_df["event"].astype(int)
    return teams_df, fx_df


def strength_to_fixed_cutpoints(series: pd.Series, cuts: Tuple[int, int, int, int]) -> pd.Series:
    """
    Map strengths to 1..5 using explicit thresholds on the raw numbers.
    `cuts` are the four boundaries separating bands 1|2|3|4|5.
    Example: cuts=(1100, 1150, 1250, 1330)
      <=1100 -> 1
      1101-1150 -> 2
      1151-1250 -> 3
      1251-1330 -> 4
      >1330 -> 5
    """
    c1, c2, c3, c4 = cuts
    bins = [-np.inf, c1, c2, c3, c4, np.inf]
    labels = [1, 2, 3, 4, 5]
    # qcut/cut return Categorical; convert to int
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(int)

def default_ratings_fixed(teams: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    # Tune these once to mirror the table you want
    cuts = (1040, 1100, 1240, 1340)


    home = strength_to_fixed_cutpoints(teams["str_away"], cuts=cuts)
    away = strength_to_fixed_cutpoints(teams["str_home"], cuts=cuts)

    return {
        int(tid): {"home": int(h), "away": int(a)}
        for tid, h, a in zip(teams["team_id"], home, away)
    }


# ---------------------------
# FDR maths
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
    """
    Return difficulty on the classic 1..5 scale (5=hard, 1=easy).
    'method' ∈ {"Opponent only", "Team only", "Team + Opponent"}.
    We treat "team rating" as self-strength (higher means stronger),
    so difficulty gets easier as your team rating rises.

    Opponent context:
      - If you're home, opponent uses their AWAY rating (tend to be weaker).
      - If you're away, opponent uses their HOME rating.
    Team context:
      - If you're home, your HOME rating applies; else your AWAY rating.

    Weighted combination:
      Opponent-only: difficulty = opp_context
      Team-only:     difficulty = 6 - team_context
      Combo:         diff = w_opp*opp_context + w_team*(6 - team_context)
    """
    team_context = team_rating_home if team_is_home else team_rating_away
    opp_context = opp_rating_away if team_is_home else opp_rating_home

    if method == "Opponent only":
        diff = opp_context
    elif method == "Team only":
        diff = 6 - team_context
    else:  # "Team + Opponent"
        # ensure normalized
        s = max(1e-6, (w_team + w_opp))
        w_t, w_o = w_team / s, w_opp / s
        diff = (w_o * opp_context) + (w_t * (6 - team_context))

    # clip to [1, 5] just in case
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
    """
    Build (1) a display DF with opponent labels and (2) a numeric DF for styling.
    Alignment is made robust by using a hidden row key (`_tid`) and reindexing val_df
    to match disp_df after sorting.
    """
    gw_cols = list(range(gw_start, gw_start + gw_len))
    id2short = dict(zip(teams["team_id"], teams["short"]))

    rows: List[Dict[str, object]] = []
    rows_vals: List[Dict[str, object]] = []

    for tid in visible_team_ids:
        if tid not in id2short:
            # skip if this team id isn't in `teams` (defensive)
            continue

        team_short = id2short[tid]
        display_cells = {"Team": team_short, "_tid": tid}  # <-- hidden key
        value_cells   = {"Team": np.nan,    "_tid": tid}   # <-- hidden key

        total = 0.0

        for gw in gw_cols:
            games = fixtures[
                (fixtures["event"] == gw) &
                ((fixtures["home_id"] == tid) | (fixtures["away_id"] == tid))
            ]

            if games.empty:
                display_cells[str(gw)] = "—"
                value_cells[str(gw)] = np.nan
            else:
                labels = []
                diffs = []
                for _, g in games.iterrows():
                    team_home = (g["home_id"] == tid)
                    opp_id = int(g["away_id"] if team_home else g["home_id"])
                    # label style: uppercase for home, lowercase for away (like your current UI)
                    tag = id2short.get(opp_id, "???")
                    label = tag if team_home else tag.lower()

                    d = compute_fixture_difficulty(
                        team_is_home=team_home,
                        team_rating_home=ratings[tid]["home"],
                        team_rating_away=ratings[tid]["away"],
                        opp_rating_home=ratings[opp_id]["home"],
                        opp_rating_away=ratings[opp_id]["away"],
                        method=method,
                        w_team=w_team,
                        w_opp=w_opp,
                    )
                    diffs.append(float(np.clip(d, 1.0, 5.0)))  # clamp early
                    labels.append(label)

                display_cells[str(gw)] = " / ".join(labels)
                value_cells[str(gw)] = float(np.mean(diffs))  # avg for cell color
                total += sum(diffs)                           # sum for Total

        # cleaner Total for display; exact total kept in values
        display_cells["Total"] = int(total) if float(total).is_integer() else round(total, 2)
        value_cells["Total"]   = float(total)

        rows.append(display_cells)
        rows_vals.append(value_cells)

    # ---------------------------
    # Build frames & ALIGN SAFELY
    # ---------------------------
    disp_df = pd.DataFrame(rows)
    val_df  = pd.DataFrame(rows_vals)

    if disp_df.empty:
        # nothing to show
        return disp_df, val_df

    # Keep column order: Team | GWs... | Total (plus hidden _tid temporarily)
    ordered_cols = ["Team"] + [str(g) for g in gw_cols] + ["Total", "_tid"]
    disp_df = disp_df.reindex(columns=ordered_cols)
    val_df  = val_df.reindex(columns=ordered_cols)

    # sort display by Total (easiest first)
    disp_df = disp_df.sort_values("Total", ascending=True, kind="mergesort")

    # **Critical step**: reindex val_df by _tid to match disp_df’s order
    val_df = val_df.set_index("_tid").loc[disp_df["_tid"]].reset_index()

    # drop hidden key in both frames and reset row indices
    disp_df = disp_df.drop(columns=["_tid"]).reset_index(drop=True)
    val_df  = val_df.drop(columns=["_tid"]).reset_index(drop=True)

    return disp_df, val_df


# ---------------------------
# UI helpers
# ---------------------------


FPL_FDR_COLORS = {
    1: "#34a853",  # very easy: dark teal green
    2: "#01FC7A",  # easy: mint/cyan
    3: "#E7E7E7",  # neutral: light grey
    4: "#E60023",  # hard: hot pink/red
    5: "#80072d",  # very hard: deep magenta/red
}

def _round_half_up(x: float) -> int:
    # 2.5 -> 3, 3.5 -> 4 (unlike Python round which does bankers rounding)
    return int(np.floor(x + 0.5))

def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def style_fpl_like(disp_df: pd.DataFrame, val_df: pd.DataFrame) -> Styler:
    """
    Style the ticker pills with FPL-like discrete colors.
    - Uses half-up rounding and clamps levels to 1..5 so colors are stable.
    - Team/Total: neutral, bold.
    """
    css = pd.DataFrame("", index=disp_df.index, columns=disp_df.columns)

    for i in disp_df.index:
        for col in disp_df.columns:
            if col in ("Team", "Total"):
                css.at[i, col] = "font-weight:700; background-color:#ffffff; color:#000000; text-align:left;"
                continue

            v = val_df.at[i, col] if col in val_df.columns else np.nan
            if pd.isna(v):
                css.at[i, col] = "background-color:#F2F2F2; color:#000000; text-align:center;"
            else:
                level = _round_half_up(float(v))
                level = _clamp(level, 1, 5)
                bg = FPL_FDR_COLORS[level]
                fg = "#000000" if level <= 3 else "#FFFFFF"
                css.at[i, col] = f"background-color:{bg}; color:{fg}; text-align:center;"

    styler = (
        disp_df.style
        .apply(lambda _: css, axis=None)
        .set_table_attributes('style="border-collapse:separate;border-spacing:6px 8px;width:100%;"')
        .set_table_styles([
            {"selector": "td, th", "props": [("border", "0")]},
            {"selector": "thead th.col_heading", "props": [("font-weight", "700")]}
        ], overwrite=False)
        .set_properties(subset=[c for c in disp_df.columns if c not in ("Team", "Total")],
                        **{"border-radius": "12px", "padding": "6px 10px", "font-weight": "600"})
        .set_properties(subset=["Team", "Total"], **{"padding": "6px 6px", "font-weight": "700"})
    )
    return styler



# ---------------------------
# App
# ---------------------------

st.title("FPL VZ MINHEE")

with st.spinner("Loading FPL data..."):
    teams_df, fixtures_df = load_fpl_data()

# Session state for ratings (persist while the app runs)
if "ratings" not in st.session_state:
    st.session_state["ratings"] = default_ratings_fixed(teams_df)  # or default_ratings_fixed

# ---------- Sidebar: Tuning ----------
with st.sidebar:
    st.header("Tuning")

    # Gameweek range
    min_gw = int(fixtures_df["event"].min())
    max_gw = int(fixtures_df["event"].max())
    gw_start = st.number_input("First Gameweek", min_value=min_gw, max_value=max_gw, value=min(6, min_gw), step=1)
    gw_len = st.number_input("Number of gameweeks", min_value=1, max_value=max_gw - gw_start + 1, value=5, step=1)

    rating_method = st.selectbox(
        "Rating Method",
        ["Team + Opponent", "Opponent only", "Team only"],
        index=0
    )

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        w_team = st.slider("Weight: Team", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    with col_w2:
        w_opp = st.slider("Weight: Opponent", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

    st.caption("Weights only apply to **Team + Opponent**. They’ll be ignored for the other methods.")

    # Save / Load ratings preset
    st.subheader("Presets")
    preset = {
        "ratings": st.session_state["ratings"],
        "settings": {
            "gw_start": gw_start, "gw_len": gw_len,
            "rating_method": rating_method, "w_team": w_team, "w_opp": w_opp
        }
    }
    st.download_button(
        "Download current preset",
        data=json.dumps(preset, indent=2),
        file_name="fdr_preset.json",
        mime="application/json",
        use_container_width=True
    )
    uploaded = st.file_uploader("Upload preset (.json)", type=["json"])
    if uploaded:
        try:
            obj = json.load(uploaded)
            st.session_state["ratings"] = obj.get("ratings", st.session_state["ratings"])
            s = obj.get("settings", {})
            gw_start = int(s.get("gw_start", gw_start))
            gw_len = int(s.get("gw_len", gw_len))
            rating_method = s.get("rating_method", rating_method)
            w_team = float(s.get("w_team", w_team))
            w_opp = float(s.get("w_opp", w_opp))
            st.success("Preset loaded.")
        except Exception as e:
            st.error(f"Invalid preset: {e}")

# ---------- Ratings editor ----------
with st.expander("Ratings (1 easy → 5 hard) — edit per team & venue", expanded=False):
    st.write("Set how tough each **team** is to face at **home** or **away**.")
    left, right = st.columns(2)
    split = math.ceil(len(teams_df) / 2)
    for col, subdf in zip((left, right), (teams_df.iloc[:split], teams_df.iloc[split:])):
        with col:
            for _, row in subdf.sort_values("name").iterrows():
                tid, name = int(row["team_id"]), row["name"]
                cols = st.columns([2, 1, 1])
                with cols[0]:
                    st.write(f"**{name}**")
                with cols[1]:
                    st.session_state["ratings"][tid]["home"] = st.number_input(
                        f"Home {name}", key=f"r{tid}h", min_value=1, max_value=5,
                        value=int(st.session_state["ratings"][tid]["home"]), step=1, label_visibility="collapsed"
                    )
                with cols[2]:
                    st.session_state["ratings"][tid]["away"] = st.number_input(
                        f"Away {name}", key=f"r{tid}a", min_value=1, max_value=5,
                        value=int(st.session_state["ratings"][tid]["away"]), step=1, label_visibility="collapsed"
                    )

# ---------- Team visibility ----------
with st.expander("Team Visibility", expanded=False):
    all_teams = teams_df.sort_values("name")
    team_options = [f'{r["name"]} ({r["short"]})' for _, r in all_teams.iterrows()]
    default_sel = team_options  # all selected by default
    current = st.multiselect("Show teams in ticker:", team_options, default=default_sel)
    # map back to ids
    visible_ids = []
    for opt in current:
        short = opt.split("(")[-1].strip(")")
        short = short[:-1] if short.endswith(")") else short
        # find id with that short
        row = all_teams[all_teams["short"] == short].iloc[0]
        visible_ids.append(int(row["team_id"]))

# ---------- Build & display ticker ----------
gw_cols = list(range(int(gw_start), int(gw_start) + int(gw_len)))

disp_df, val_df = build_ticker(
    teams=teams_df,
    fixtures=fixtures_df,
    ratings=st.session_state["ratings"],
    gw_start=int(gw_start),
    gw_len=int(gw_len),
    visible_team_ids=visible_ids if current else list(teams_df["team_id"]),
    method=rating_method,
    w_team=w_team,
    w_opp=w_opp,
)

st.subheader("Fixture Ticker")
st.caption("Green = easier fixtures (lower difficulty). Red = tougher fixtures (higher difficulty).")

styled = style_fpl_like(disp_df, val_df).hide(axis="index")
st.write(styled)


import json
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

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


def strength_to_rating(series: pd.Series) -> pd.Series:
    """
    Convert 0-100ish 'strength' to a 1-5 rating (5 = tougher opponent).
    We use quantile bins to mimic "official-like" buckets.
    """
    # rank first to avoid duplicate bin edge issues
    ranked = series.rank(method="first")
    buckets = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return buckets


def default_ratings(teams: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    """Create default 1..5 ratings per team (home/away) from strengths."""
    home = strength_to_rating(teams["str_home"])
    away = strength_to_rating(teams["str_away"])
    ratings = {}
    for tid, h, a in zip(teams["team_id"], home, away):
        ratings[int(tid)] = {"home": int(h), "away": int(a)}
    return ratings


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
    Handles blanks and double GWs (averages difficulty within a GW, sums in Total).
    """
    gw_cols = list(range(gw_start, gw_start + gw_len))
    id2short = dict(zip(teams["team_id"], teams["short"]))
    rows = []
    rows_vals = []

    for tid in visible_team_ids:
        team_short = id2short[tid]
        display_cells = {"Team": team_short}
        value_cells = {"Team": np.nan}

        total = 0.0

        for gw in gw_cols:
            games = fixtures[(fixtures["event"] == gw) & (
                (fixtures["home_id"] == tid) | (fixtures["away_id"] == tid)
            )]

            if games.empty:
                display_cells[str(gw)] = "—"
                value_cells[str(gw)] = np.nan
            else:
                labels = []
                diffs = []
                for _, g in games.iterrows():
                    team_home = (g["home_id"] == tid)
                    opp_id = int(g["away_id"] if team_home else g["home_id"])
                    label = f"{id2short[opp_id]}"
                    label = label if team_home else label.lower()

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
                    labels.append(label)
                    diffs.append(d)

                # If double GW: join labels with "/" and average difficulty for the cell
                display_cells[str(gw)] = " / ".join(labels)
                cell_val = float(np.mean(diffs))
                value_cells[str(gw)] = cell_val
                total += sum(diffs)

        display_cells["Total"] = round(total, 2)
        value_cells["Total"] = total
        rows.append(display_cells)
        rows_vals.append(value_cells)

    disp_df = pd.DataFrame(rows)
    val_df = pd.DataFrame(rows_vals)
    disp_df = disp_df.sort_values("Total", ascending=True, kind="mergesort").reset_index(drop=True)
    val_df = val_df.loc[disp_df.index].reset_index(drop=True)
    return disp_df, val_df


# ---------------------------
# UI helpers
# ---------------------------

def color_for_value(v: float) -> str:
    """
    Map 1..5 difficulty -> green..red.
    1 = very easy (green), 5 = very hard (red).
    """
    if math.isnan(v):
        return "#f2f2f2"
    # linear gradient green (1) -> yellow (3) -> red (5)
    # we'll interpolate manually between stops
    def lerp(a, b, t): return int(a + (b - a) * t)
    # choose stop
    if v <= 3:
        # green -> yellow
        t = (v - 1) / 2.0
        r = lerp(46, 255, t)   # 0x2e -> 0xff
        g = lerp(204, 235, t)  # 0xcc -> 0xeb
        b = lerp(113, 59, t)   # 0x71 -> 0x3b
    else:
        # yellow -> red
        t = (v - 3) / 2.0
        r = lerp(255, 231, t)  # 0xff -> 0xe7
        g = lerp(235, 76, t)   # 0xeb -> 0x4c
        b = lerp(59, 60, t)    # 0x3b -> 0x3c
    return f"#{r:02x}{g:02x}{b:02x}"


def style_by_values(display_df: pd.DataFrame, values_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Build a Styler with background/text colors based on values_df.
    - Team column: bold text, neutral background
    - Blank cells (NaN): light gray
    - Difficulty 1..5 -> green..red via color_for_value()
    """
    # Build a same-shaped DataFrame of CSS strings
    css = pd.DataFrame("", index=display_df.index, columns=display_df.columns)

    for i in display_df.index:
        for col in display_df.columns:
            if col == "Team":
                css.at[i, col] = "font-weight: 600; background-color: #ffffff;"
                continue

            v = values_df.at[i, col] if (col in values_df.columns) else np.nan
            if pd.isna(v):
                css.at[i, col] = "background-color: #f2f2f2; color: #000000;"
            else:
                bg = color_for_value(float(v))
                fg = "#000000" if v < 4.4 else "#ffffff"
                css.at[i, col] = f"background-color: {bg}; color: {fg};"

    # Apply the CSS DataFrame in one shot (no applymap)
    styler = display_df.style.apply(lambda _: css, axis=None)
    return styler



# ---------------------------
# App
# ---------------------------

st.title("FPL VZ MINHEE")

with st.spinner("Loading FPL data..."):
    teams_df, fixtures_df = load_fpl_data()

# Session state for ratings (persist while the app runs)
if "ratings" not in st.session_state:
    st.session_state["ratings"] = default_ratings(teams_df)

# ---------- Sidebar: Tuning ----------
with st.sidebar:
    st.header("Tuning")

    # Gameweek range
    min_gw = int(fixtures_df["event"].min())
    max_gw = int(fixtures_df["event"].max())
    gw_start = st.number_input("First Gameweek", min_value=min_gw, max_value=max_gw, value=min(6, min_gw), step=1)
    gw_len = st.number_input("Number of gameweeks", min_value=1, max_value=max_gw - gw_start + 1, value=6, step=1)

    rating_method = st.selectbox(
        "Rating Method",
        ["Team + Opponent", "Opponent only", "Team only"],
        index=0
    )

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        w_team = st.slider("Weight: Team", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    with col_w2:
        w_opp = st.slider("Weight: Opponent", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

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

styled = style_by_values(disp_df, val_df)

# IMPORTANT: render the Styler with st.write or st.table so colors show up
st.write(styled)              # ✅ colors preserved
# st.table(styled)            # also works; table is static

# If you still want a scrollable, interactive grid (but w/o colors), you can also show:
# st.dataframe(disp_df, width="stretch", hide_index=True)


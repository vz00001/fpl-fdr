# app.py
import json, math
import numpy as np
import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler  # for type hints only

import fpl_core as core  # <- our logic module
from importlib import reload
core = reload(core)


st.set_page_config(page_title="Fixture Difficulty Rating", layout="wide")
st.markdown("""
<style>
.fdr-tt { position: relative; display:inline-block; }
.fdr-tt::after {
  content: attr(data-tt);
  white-space: pre-line;            /* \n becomes line-breaks */
  visibility: hidden;
  opacity: 0;
  position: absolute;
  left: 50%;
  transform: translate(-50%, -110%);
  background: #1e1e2f;
  color: #fff;
  padding: 8px 10px;
  border-radius: 8px;
  box-shadow: 0 8px 22px rgba(0,0,0,.25);
  width: 240px;
  z-index: 5;
  transition: opacity .15s ease-in-out;
  pointer-events: none;
  font-size: 12px; line-height: 1.25;
}
.fdr-tt:hover::after { visibility: visible; opacity: 1; }
table.dataframe td, table.dataframe th { position: relative; }
</style>
""", unsafe_allow_html=True)


# ------------- cached wrappers around core.fetch -------------
# @st.cache_data(ttl=3600)
def load_fpl_data():
    return core.fetch_fpl_data()

# ------------- UI-only helpers (styling etc.) -------------
FPL_FDR_COLORS = {
    1: "#34a853", 2: "#01FC7A", 3: "#E7E7E7", 4: "#E60023", 5: "#80072d",
}

def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))

def style_fpl_like(disp_df: pd.DataFrame, val_df: pd.DataFrame) -> Styler:
    """
    Style the ticker with FPL-like colors and add per-cell hover popups
    using a span + data-tt attribute (no pandas.set_tooltips dependency).
    """
    # We inject HTML into a copy so we can render with escape=False
    html_df = disp_df.copy()
    css = pd.DataFrame("", index=disp_df.index, columns=disp_df.columns)

    for i in disp_df.index:
        team_name = disp_df.at[i, "Team"]
        for col in disp_df.columns:
            if col in ("Team", "Total"):
                css.at[i, col] = "font-weight:700; background-color:#ffffff; color:#000; text-align:left;"
                continue

            v = val_df.at[i, col] if col in val_df.columns else np.nan
            label = disp_df.at[i, col]

            if pd.isna(v):
                css.at[i, col] = "background-color:#F2F2F2; color:#000; text-align:center;"
                # keep an em dash or label; optional tooltip
                html_df.at[i, col] = f'<span class="fdr-tt" data-tt="No fixture scheduled">{label}</span>'
            else:
                level = int(max(1, min(5, _round_half_up(float(v)))))
                bg = FPL_FDR_COLORS[level]
                fg = "#000000" if (2 <= level <= 4) else "#FFFFFF"
                css.at[i, col] = f"background-color:{bg}; color:{fg}; text-align:center;"

                # Build a multiline tooltip with plain text (use \n)
                tt = (
                    f"Opponent(s): {label}\n"
                    f"Raw difficulty: {float(v):.2f}\n"
                    f"Rounded: {level}"
                )
                safe_tt = tt.replace('"', "'")  # avoid breaking HTML attribute
                html_df.at[i, col] = f'<span class="fdr-tt" data-tt="{safe_tt}">{label}</span>'

    styler = (
        html_df.style
        .apply(lambda _: css, axis=None)
        .set_table_attributes('style="border-collapse:separate;border-spacing:6px 8px;width:100%;"')
        .set_table_styles([
            {"selector": "td, th", "props": [("border", "0")]},
            {"selector": "thead th.col_heading", "props": [("font-weight", "700")]}
        ], overwrite=False)
        .set_properties(subset=[c for c in disp_df.columns if c not in ("Team","Total")],
                        **{"border-radius":"12px","padding":"6px 10px","font-weight":"600"})
        .set_properties(subset=["Team","Total"], **{"padding":"6px 6px","font-weight":"1900"})
    )
    return styler


# --------------------------- App ---------------------------
st.title("FPL - Fixture Difficulty Rating")

with st.spinner("Loading FPL data..."):
    teams_df, fixtures_df, event_df, fetched_at, players_df = load_fpl_data()
    table_df = core.build_pl_table(teams_df, fixtures_df)

# Session state for ratings
if "ratings" not in st.session_state:
    st.session_state["ratings"] = core.table_ratings_fixed(table_df, teams_df)

# ---------- Sidebar: Tuning ----------
with st.sidebar:
    st.header("Tuning")
    current_gw = core.determine_current_gw(event_df)
    min_gw = int(fixtures_df["event"].min())
    max_gw = int(fixtures_df["event"].max())
    gw_start_default = core.clamp(int(current_gw), int(min_gw), int(max_gw))
    gw_start = st.number_input("First Gameweek", min_value=min_gw, max_value=max_gw, value=gw_start_default, step=1)
    gw_len = st.number_input("Number of gameweeks", min_value=1, max_value=max_gw - gw_start + 1, value=5, step=1)

    rating_method = st.selectbox("Rating Method", ["Team + Opponent","Opponent only","Team only"], index=0)
    col_w1, col_w2 = st.columns(2)
    with col_w1: w_team = st.slider("Weight: Team", min_value=0.0, max_value=1.0, value=0.25, step=0.25)
    with col_w2: w_opp  = st.slider("Weight: Opponent", min_value=0.0, max_value=1.0, value=0.75, step=0.25)
    st.caption("Works best when both weights add to 1.0.")

    st.subheader("Presets")
    preset = {"ratings": st.session_state["ratings"],
              "settings": {"gw_start": gw_start, "gw_len": gw_len,
                           "rating_method": rating_method, "w_team": w_team, "w_opp": w_opp}}
    st.download_button("Download current preset", data=json.dumps(preset, indent=2),
                       file_name="fdr_preset.json", mime="application/json", use_container_width=True)
    uploaded = st.file_uploader("Upload preset (.json)", type=["json"])
    if uploaded:
        try:
            obj = json.load(uploaded)
            st.session_state["ratings"] = obj.get("ratings", st.session_state["ratings"])
            s = obj.get("settings", {})
            gw_start = int(s.get("gw_start", gw_start))
            gw_len = int(s.get("gw_len", gw_len))
            rating_method = s.get("rating_method", rating_method)
            w_team = float(s.get("w_team", w_team)); w_opp = float(s.get("w_opp", w_opp))
            st.success("Preset loaded.")
        except Exception as e:
            st.error(f"Invalid preset: {e}")
    # --- Sidebar: Team visibility ---
    st.subheader("Teams")

    _all = teams_df.sort_values("name")
    id_options = list(map(int, _all["team_id"]))

    def _fmt_team(tid: int) -> str:
        r = _all.loc[_all["team_id"] == tid].iloc[0]
        return f'{r["name"]} ({r["short"]})'

    visible_ids = st.multiselect(
        "Show teams in ticker:",
        id_options,
        default=id_options,           # all selected by default
        format_func=_fmt_team,
    )
    # safety: if user deselects everything, fallback to all
    if not visible_ids:
        visible_ids = id_options

# ---------- Build & display ticker ----------
disp_df, val_df = core.build_ticker(
    teams=teams_df, 
    fixtures=fixtures_df, 
    ratings=st.session_state["ratings"],
    gw_start=int(gw_start), 
    gw_len=int(gw_len),
    visible_team_ids=visible_ids,
    method=rating_method, 
    w_team=w_team, 
    w_opp=w_opp,
)

st.subheader("Fixture Ticker")
st.caption("Green = easier fixtures. Red = tougher fixtures.")
styled = style_fpl_like(disp_df, val_df).hide(axis="index")
# Render the full HTML manually so Streamlit doesn't escape tooltip markup
st.markdown(styled.to_html(escape=False), unsafe_allow_html=True)


# ---------- Display league table ----------
st.divider()
st.subheader("Premier League Table")

display_cols = ["Pos","Team","P","W","D","L","GF","GA","GD","Pts"]
table_disp = table_df[display_cols].copy()

st.markdown("""
<style>
thead tr th { position: sticky; top: 0; background: white; z-index: 1; }
td, th { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

def _pos_band(pos: int) -> str:
    if pos <= 4:      return "#34a853"
    if pos == 5:      return "#0048FF"
    if pos >= 18:     return "#E60023"
    return "#ffffff"

def _style_pos_only(df: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for i, pos in enumerate(df["Pos"]):
        styles.at[i, "Pos"] = f"background-color:{_pos_band(int(pos))}; font-weight:700; text-align:center;"
    return styles

styler = (
    table_disp.style
    .hide(axis="index")
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "left")]},
        {"selector": "td", "props": [("padding", "6px 8px")]}
    ])
    .set_properties(subset=["Pts"], **{"font-weight": "700"})
    .apply(_style_pos_only, axis=None)
)
st.write(styler, unsafe_allow_html=True)

# ---------- Ratings editor ----------
st.divider()
with st.expander("Ratings (1 easy → 5 hard) — edit per team & venue", expanded=False):
    st.write("Set how tough each team is to face at home or away.")
    left, right = st.columns(2)
    split = math.ceil(len(teams_df) / 2)
    for col, subdf in zip((left, right), (teams_df.iloc[:split], teams_df.iloc[split:])):
        with col:
            for _, row in subdf.sort_values("name").iterrows():
                tid, name = int(row["team_id"]), row["name"]
                cols = st.columns([2, 1, 1])
                with cols[0]: st.write(f"**{name}**")
                with cols[1]:
                    st.session_state["ratings"][tid]["home"] = st.number_input(
                        f"Home {name}", key=f"r{tid}h", min_value=1, max_value=5,
                        value=int(st.session_state["ratings"][tid]["home"]), step=1, label_visibility="collapsed")
                with cols[2]:
                    st.session_state["ratings"][tid]["away"] = st.number_input(
                        f"Away {name}", key=f"r{tid}a", min_value=1, max_value=5,
                        value=int(st.session_state["ratings"][tid]["away"]), step=1, label_visibility="collapsed")

# --- Footer / meta ---
tz = "Asia/Ho_Chi_Minh"
bits = []
if fetched_at is not None:
    try:
        local = pd.to_datetime(fetched_at, utc=True).tz_convert(tz)
        bits.append(f"Last updated: {local.strftime('%Y-%m-%d %H:%M %Z')} My Tho time")
    except Exception:
        pass
src = "Source: FPL fixtures (finished matches only). ^0.5.4"
tail = " • ".join(bits) + (" • " if bits else "")
st.caption(f"{tail}{src}")

# ---------- ICTindex: Filters + Table ----------

st.subheader("ICT Index")

# --- Sidebar controls
with st.sidebar:
    st.header("ICT Filters")

    # 1) By Position
    pos_options = ["All", "GK", "DF", "MF", "FW"]
    selected_pos = st.segmented_control("By Position", pos_options, default="All")

    # 2) Max Price slider
    min_price = float(players_df["price_m"].min())
    max_price = float(players_df["price_m"].max())
    sel_price = st.slider(
        "Max Price (£m)",
        min_value=round(min_price, 1),
        max_value=round(max_price, 1),
        value=round(max_price, 1),
        step=0.1,
    )

# Normalize your “All” to the value your function expects
pos_arg = "ALL" if selected_pos == "All" else selected_pos

# --- Apply filters
filtered = core.combine_filters(pos_arg, sel_price, players_df)

# --- Optional: pagination
page_size = st.selectbox("Rows per page", [10, 20, 40], index=0, key="ict_page_size")
total_rows = len(filtered)
total_pages = max(1, (total_rows + page_size - 1) // page_size)

# Keep page index in session so it doesn't jump around when filtering
pg_key = "ict_page_index"
if pg_key not in st.session_state:
    st.session_state[pg_key] = 1
# If filters change and current page is now out of range, clamp it
st.session_state[pg_key] = max(1, min(st.session_state[pg_key], total_pages))

col_prev, col_page, col_next = st.columns([1,2,1])
with col_prev:
    if st.button("◀ Previous", use_container_width=True, disabled=(st.session_state[pg_key] <= 1)):
        st.session_state[pg_key] -= 1
with col_page:
    st.write(f"Page {st.session_state[pg_key]} / {total_pages}")
with col_next:
    if st.button("Next ▶", use_container_width=True, disabled=(st.session_state[pg_key] >= total_pages)):
        st.session_state[pg_key] += 1

start = (st.session_state[pg_key] - 1) * page_size
end = start + page_size
page_df = filtered.iloc[start:end].copy()

# --- Shape for display (don’t expose internal ids unless you want to)
display_cols = [
    "name", "pos", "team_short",
    "price_m", "total_points",
    "ict_index", "influence", "creativity", "threat",
]
page_df = page_df[display_cols]

# --- Pretty column names + number formats
page_df = page_df.rename(columns={
    "name": "Player", "pos": "Pos", "team_short": "T",
    "price_m": "Val", "total_points": "Pts",
    "ict_index": "ICT", "influence": "Inf", "creativity": "Crt", "threat": "Thr",
})

# Streamlit’s dataframe formatting
st.dataframe(
    page_df.style
        .format({
            "Val": "{:.1f}",
            "ICT": "{:.1f}",
            "Inf": "{:.1f}",
            "Crt": "{:.1f}",
            "Thr": "{:.1f}",
        })
        .hide(axis="index"),
    use_container_width=True,
)

st.caption(f"Showing {len(page_df)} of {total_rows} matching players.")

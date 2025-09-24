"""
Tiny test harness for quick, no-framework checks while building.
- Time function calls
- Pretty-print small outputs
- Lightweight assertions
- Optional JSON caching for API responses
Usage:
  python test_harness.py load_fpl_data
  python test_harness.py compute_fixture_difficulty --args '{"team_is_home": true, "team_rating_home":4, "team_rating_away":3, "opp_rating_home":5, "opp_rating_away":4, "method":"Team + Opponent", "w_team":0.5, "w_opp":0.5}'
"""
import argparse
import inspect
import io
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Tuple

# -------------------------
# (1) your app functions
# -------------------------
# You can import from your module instead:
from app import load_data, strength_to_rating, default_ratings
import requests
import pandas as pd
import numpy as np

def load_fpl_data():
    """Fetch teams and fixtures from FPL API; return (teams_df, fixtures_df)."""
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
    fx_df = fx_df.loc[fx_df["event"].notna(), ["event", "team_h", "team_a", "kickoff_time"]]
    fx_df = fx_df.rename(columns={"team_h": "home_id", "team_a": "away_id"})
    fx_df["event"] = fx_df["event"].astype(int)
    return teams_df, fx_df

def compute_fixture_difficulty(
    team_is_home: bool,
    team_rating_home: int,
    team_rating_away: int,
    opp_rating_home: int,
    opp_rating_away: int,
    method: str = "Team + Opponent",
    w_team: float = 0.5,
    w_opp: float = 0.5,
) -> float:
    """Simple 1..5 difficulty (5=hard)."""
    team_ctx = team_rating_home if team_is_home else team_rating_away
    opp_ctx = opp_rating_away if team_is_home else opp_rating_home
    if method == "Opponent only":
        d = opp_ctx
    elif method == "Team only":
        d = 6 - team_ctx
    else:
        s = max(1e-6, w_team + w_opp)
        d = (w_opp / s) * opp_ctx + (w_team / s) * (6 - team_ctx)
    return float(np.clip(d, 1.0, 5.0))

teams_df, fixtures_df = load_data()

def strength_to_rating_wrapper():
    return strength_to_rating(teams_df["str_home"])

def default_ratings_wrapper():
    return default_ratings(teams_df)

# Map names -> callables you want to test
REGISTRY: Dict[str, Callable[..., Any]] = {
    "load_fpl_data": load_fpl_data,
    "compute_fixture_difficulty": compute_fixture_difficulty,
    "load_data": load_data,
    "strength_to_rating": strength_to_rating_wrapper,
    "default_ratings": default_ratings_wrapper
}

# -------------------------
# (2) helpers
# -------------------------
def tictoc(fn: Callable, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    dt = (time.time() - t0) * 1000
    return out, dt

def pretty(obj: Any, max_rows: int = 20) -> str:
    try:
        import pandas as pd  # noqa
        if hasattr(obj, "head"):
            buf = io.StringIO()
            obj.head(max_rows).to_string(buf)
            return buf.getvalue()
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}({len(obj)} items) -> {repr(obj[:3])}..."
    try:
        return json.dumps(obj, indent=2)[:800]
    except Exception:
        return repr(obj)[:800]

def ensure_cache_dir():
    p = ".cache"
    os.makedirs(p, exist_ok=True)
    return p

def cache_json(path: str, loader: Callable[[], Any]) -> Any:
    """Load from json cache if exists, otherwise compute & save."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    data = loader()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data

# -------------------------
# (3) quick checks (asserts)
# -------------------------
def quick_check(name: str, result: Any):
    """Lightweight validation per function."""
    if name == "load_fpl_data":
        assert isinstance(result, tuple) and len(result) == 2, "Should return (teams_df, fixtures_df)"
        teams, fixtures = result
        for col in ["team_id", "name", "short", "str_home", "str_away"]:
            assert col in teams.columns, f"Missing teams column: {col}"
        for col in ["event", "home_id", "away_id"]:
            assert col in fixtures.columns, f"Missing fixtures column: {col}"
        assert len(teams) > 0, "No teams returned"
        print(f"[OK] teams: {teams.shape} fixtures: {fixtures.shape}")
    elif name == "compute_fixture_difficulty":
        v = result
        assert isinstance(v, float), "Should return float"
        assert 1.0 <= v <= 5.0, "Difficulty must be within 1..5"
        print(f"[OK] difficulty={v:.2f}")
    else:
        # Default: just show type/size
        print(f"[INFO] Output type: {type(result)}")

# -------------------------
# (4) CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Tiny function test harness")
    parser.add_argument("fn", choices=sorted(REGISTRY.keys()), help="Function name to call")
    parser.add_argument("--args", type=str, default="{}", help="JSON of keyword args for the function")
    parser.add_argument("--use-cache", action="store_true", help="Cache example API payloads where relevant")
    args = parser.parse_args()

    fn = REGISTRY[args.fn]
    kwargs = json.loads(args.args)

    # Optional: monkey-patch requests.get for load_fpl_data if caching requested
    if args.use_cache and args.fn == "load_fpl_data":
        ensure_cache_dir()
        real_get = requests.get

        def cached_get(url, *a, **k):
            key = url.split("/api/")[-1].replace("/", "_")
            path = os.path.join(".cache", f"{key}.json")
            return type("Resp", (), {
                "json": lambda: cache_json(path, lambda: real_get(url, *a, **k).json())
            })()

        requests.get = cached_get  # type: ignore

    # Show signature hint
    sig = inspect.signature(fn)
    print(f"▶ Running {args.fn}{sig} with kwargs={kwargs}")

    out, ms = tictoc(fn, **kwargs)

    # Pretty preview
    if isinstance(out, tuple) and len(out) == 2:
        print("\n[Preview] item 1:\n", pretty(out[0]))
        print("\n[Preview] item 2:\n", pretty(out[1]))
    else:
        print("\n[Preview]:\n", pretty(out))

    # Quick assertions
    quick_check(args.fn, out)

    print(f"\n⏱  {args.fn} completed in {ms:.1f} ms")

if __name__ == "__main__":
    sys.exit(main())

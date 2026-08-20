"""
build_game_table.py

Builds the one row per game table

Outputs:
    data/processed/games.parquet     played regular season games (has target)
    data/processed/upcoming.parquet  scheduled games not yet played

Run:
    python src/build_game_table.py

Why not nfl_data_py: as of v0.3.3 it pins pandas<2.0 and numpy<2.0, which will
not build on modern Python. We read the same nflverse source file directly.
"""

from pathlib import Path
import sys

import pandas as pd

# nflverse maintains this file and updates through the season
SCHEDULE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# 2002 is the first season with the current 32-team / 8-division alignment,
# which keeps scheduling structure consistent
FIRST_SEASON = 2002

OUT_DIR = Path("data/processed")

# Historical -> current abbreviations. Verified against the full file: these are
# the only three that ever change. nflverse uses LA (not LAR) and JAX (not JAC).
TEAM_FIXES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Columns worth carrying forward. Vegas lines and rest days are already here.
KEEP = [
    "game_id", "season", "week", "gameday", "game_type",
    "home_team", "away_team", "home_score", "away_score", "result",
    "home_rest", "away_rest", "div_game", "location",
    "roof", "surface", "temp", "wind",
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
]


def load_schedules() -> pd.DataFrame:
    print(f"downloading {SCHEDULE_URL}")
    df = pd.read_csv(SCHEDULE_URL, low_memory=False)
    print(f"  {len(df):,} rows, seasons {df.season.min()}-{df.season.max()}")

    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        # Upstream schema changed. Fail loudly rather than silently dropping features.
        sys.exit(f"ERROR: expected columns missing upstream: {missing}")

    return df[KEEP].copy()


def standardize_teams(df: pd.DataFrame) -> pd.DataFrame:
    df["home_team"] = df.home_team.replace(TEAM_FIXES)
    df["away_team"] = df.away_team.replace(TEAM_FIXES)

    n = df.home_team.nunique()
    # If a relocation/rename happens after this was written, this catches it.
    assert n == 32, f"expected 32 teams after remapping, saw {n}"
    return df


def build(df: pd.DataFrame):
    df = df[(df.season >= FIRST_SEASON) & (df.game_type == "REG")].copy()

    upcoming = df[df.home_score.isna()].copy()
    played = df[df.home_score.notna()].copy()

    # `result` is home_score - away_score. Ties are ~0.2% of games; a binary
    # classifier has nowhere to put them, so drop and record the count.
    n_ties = int((played.result == 0).sum())
    played = played[played.result != 0].copy()
    played["home_win"] = (played.result > 0).astype(int)

    played["gameday"] = pd.to_datetime(played.gameday)
    played = played.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    return played, upcoming, n_ties


def sanity_checks(played: pd.DataFrame):
    """Cheap assertions that would have caught every join bug I've seen."""
    assert played.game_id.is_unique, "duplicate game_id"
    assert played.home_win.notna().all(), "null target"
    assert (played.home_team != played.away_team).all(), "team playing itself"

    # A team should appear at most once per week
    long = pd.concat([
        played[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
        played[["season", "week", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    dupes = long.groupby(["season", "week", "team"]).size()
    assert (dupes == 1).all(), f"team appears twice in a week:\n{dupes[dupes > 1]}"

    rate = played.home_win.mean()
    assert 0.50 < rate < 0.62, f"home win rate {rate:.3f} is implausible"

    print("  sanity checks passed")


def report(played: pd.DataFrame, upcoming: pd.DataFrame, n_ties: int):
    print(f"\n{'='*58}\nGAME TABLE\n{'='*58}")
    print(f"played games   : {len(played):,}  ({played.season.min()}-{played.season.max()})")
    print(f"upcoming games : {len(upcoming):,}")
    print(f"ties dropped   : {n_ties}")

    print("\n--- BASELINES (this is what you have to beat) ---")
    print(f"always pick home : {100*played.home_win.mean():.2f}%")

    s = played.dropna(subset=["spread_line"])
    vegas = ((s.spread_line > 0) == (s.home_win == 1)).mean()
    print(f"always pick favorite : {100*vegas:.2f}%  (n={len(s):,}, "
          f"{100*len(s)/len(played):.0f}% of games have a line)")

    # Sign-convention check. Home win% must rise monotonically with spread_line.
    # If this ever inverts, the upstream convention flipped and every downstream
    # feature built on it is silently backwards.
    print("\n--- spread_line sanity (home win% must increase) ---")
    bins = [-30, -10, -7, -3, 0, 3, 7, 10, 30]
    grp = s.groupby(pd.cut(s.spread_line, bins), observed=True).home_win.agg(["mean", "size"])
    prev = -1.0
    for interval, row in grp.iterrows():
        print(f"  {str(interval):>12}  home win% {100*row['mean']:5.1f}  n={int(row['size']):5d}")
        assert row["mean"] > prev, "spread_line convention is not monotonic -- investigate"
        prev = row["mean"]
    print("  convention confirmed: positive spread_line = home favored")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = standardize_teams(load_schedules())
    played, upcoming, n_ties = build(df)

    sanity_checks(played)
    report(played, upcoming, n_ties)

    played.to_parquet(OUT_DIR / "games.parquet", index=False)
    upcoming.to_parquet(OUT_DIR / "upcoming.parquet", index=False)
    print(f"\nwrote {OUT_DIR/'games.parquet'} and {OUT_DIR/'upcoming.parquet'}")


if __name__ == "__main__":
    main()
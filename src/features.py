from pathlib import Path

import numpy as np
import pandas as pd

TEAM_GAMES = Path("data/processed/team_games.parquet")
GAMES = Path("data/processed/games.parquet")
OUT = Path("data/processed/features.parquet")

WINDOW = 5          # rolling window length, in games
MIN_PERIODS = 3     # games required before a feature is produced

# Windows reset at each season start. Rosters, coaches and schemes turn over
# heavily in the offseason, so last January's form is weak evidence about September.
RESET_EACH_SEASON = True

STATS = ["off_epa", "def_epa", "off_success", "def_success"]


# ----------------------------------------------------------------------------
# rolling features
# ----------------------------------------------------------------------------

def add_rolling(tg: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game, the mean of that team's previous WINDOW games.

    Sort matters: rolling operates on row order, so the frame must be ordered by
    team and then chronologically before this runs.
    """
    tg = tg.sort_values(["team", "season", "week"]).reset_index(drop=True)
    keys = ["team", "season"] if RESET_EACH_SEASON else ["team"]

    for stat in STATS:
        # shift(1) first, then roll: the window can only ever see earlier rows.
        # Doing it inside groupby keeps one team's games from bleeding into
        # another's at the boundary between teams.
        tg[f"{stat}_r{WINDOW}"] = (
            tg.groupby(keys)[stat]
              .transform(lambda s: s.shift(1)
                                   .rolling(WINDOW, min_periods=MIN_PERIODS)
                                   .mean())
        )

    return tg


# ----------------------------------------------------------------------------
# the test that matters
# ----------------------------------------------------------------------------

def test_no_leakage(tg: pd.DataFrame, team="BAL", season=2024, week=10) -> None:
    """
    Recompute one feature by hand and compare.

    Pull the team's games from strictly earlier weeks, average the last WINDOW
    of them, and assert it equals what add_rolling produced. If shift(1) were
    missing or off by one, this fails immediately -- which is exactly how the
    bug presents.
    """
    rows = tg[(tg.team == team) & (tg.season == season)]

    hist = rows[rows.week < week].sort_values("week")
    target = rows[rows.week == week]
    assert len(target) == 1, f"expected exactly one {team} game in week {week}"

    for stat in STATS:
        expected = hist[stat].tail(WINDOW).mean()
        actual = target[f"{stat}_r{WINDOW}"].iloc[0]
        assert np.isclose(expected, actual), (
            f"LEAKAGE: {stat} for {team} {season} wk{week} -- "
            f"hand-computed {expected:.6f}, feature says {actual:.6f}"
        )

    # Second check: the current game's own value must not appear in its feature.
    # Constructed so it can only pass if the current row was excluded.
    own = target["off_epa"].iloc[0]
    feat = target[f"off_epa_r{WINDOW}"].iloc[0]
    without = hist["off_epa"].tail(WINDOW).mean()
    with_current = pd.concat([hist["off_epa"].tail(WINDOW - 1),
                              pd.Series([own])]).mean()
    assert np.isclose(feat, without) and not np.isclose(feat, with_current), \
        "feature value is consistent with including the current game"

    weeks_used = hist.week.tail(WINDOW).tolist()
    print(f"  leakage test passed ({team} {season} wk{week} "
          f"built from weeks {weeks_used})")


# ----------------------------------------------------------------------------
# join to games
# ----------------------------------------------------------------------------

def join_to_games(tg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    # Attach each team's features to the game as home_* and away_*
    cols = [f"{s}_r{WINDOW}" for s in STATS]
    side = tg[["game_id", "team"] + cols]

    out = games.merge(
        side.rename(columns={c: f"home_{c}" for c in cols}),
        left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="inner"
    ).drop(columns="team")

    out = out.merge(
        side.rename(columns={c: f"away_{c}" for c in cols}),
        left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="inner"
    ).drop(columns="team")

    # Matchup differences. A logistic model can form these itself, but stating
    # them explicitly makes coefficients readable: positive favors the home team.
    out["epa_edge_off"] = out[f"home_off_epa_r{WINDOW}"] - out[f"away_off_epa_r{WINDOW}"]
    out["epa_edge_def"] = out[f"away_def_epa_r{WINDOW}"] - out[f"home_def_epa_r{WINDOW}"]
    out["rest_edge"] = out.home_rest - out.away_rest

    return out


def main() -> None:
    tg = pd.read_parquet(TEAM_GAMES)
    games = pd.read_parquet(GAMES)

    tg = add_rolling(tg)
    test_no_leakage(tg)

    feat = join_to_games(tg, games)

    before = len(feat)
    feat = feat.dropna(subset=[c for c in feat.columns if c.endswith(f"_r{WINDOW}")])
    dropped = before - len(feat)

    print(f"\n{'='*58}\nFEATURES\n{'='*58}")
    print(f"games with features : {len(feat):,}")
    print(f"dropped (warm-up)   : {dropped:,} "
          f"({100*dropped/before:.1f}% -- first {MIN_PERIODS} games of each season)")
    print(f"seasons             : {feat.season.min()}-{feat.season.max()}")
    print(f"window              : {WINDOW} games, min {MIN_PERIODS}, "
          f"reset each season: {RESET_EACH_SEASON}")

    print("\n--- does the edge feature actually separate outcomes? ---")
    q = pd.qcut(feat.epa_edge_off, 5, labels=["worst", "low", "mid", "high", "best"])
    print(feat.groupby(q, observed=True).home_win.agg(["mean", "size"])
              .rename(columns={"mean": "home_win_rate", "size": "n"})
              .round(3).to_string())

    feat.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
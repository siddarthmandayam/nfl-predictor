from pathlib import Path

import pandas as pd

URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_{season}.parquet")

SEASONS = list(range(2022, 2025))   # start small, can backfill once this is trusted

RAW = Path("data/raw")
OUT = Path("data/processed/team_games.parquet")

# Use only scrimmage plays since punts, kickoffs and field goals carry extreme EPA values
# that swamp the signal that can render these features useless
PLAY_TYPES = ["pass", "run"]

# Columns actually needed out of the total 372
COLS = ["game_id", "season", "week", "posteam", "defteam",
        "play_type", "epa", "success"]


def load_pbp(season: int) -> pd.DataFrame:
    """Download one season's play-by-play, caching to data/raw."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"pbp_{season}.parquet"

    if not path.exists():
        print(f"  downloading {season} ...", end=" ", flush=True)
        pd.read_parquet(URL.format(season=season), columns=COLS).to_parquet(path)
        print("cached")

    return pd.read_parquet(path)


def team_game_stats(df: pd.DataFrame, include_playoffs=True) -> pd.DataFrame:
    """One row per (game_id, team) with offensive and defensive EPA."""
    d = df[
        df.play_type.isin(PLAY_TYPES)
        & df.epa.notna()
        & df.posteam.notna()      # posteam is null on timeouts and end-of-period rows
        & df.defteam.notna()
    ]
    
    if not include_playoffs:
        d = d[d.season_type == "REG"]

    keys = ["game_id", "season", "week"]

    # Same plays grouped from the offense's side
    off = (d.groupby(keys + ["posteam"])
             .agg(off_epa=("epa", "mean"),
                  off_success=("success", "mean"),
                  n_off_plays=("epa", "size"))
             .reset_index()
             .rename(columns={"posteam": "team"}))

    # Same plays grouped from the defense's side
    def_ = (d.groupby(keys + ["defteam"])
              .agg(def_epa=("epa", "mean"),
                   def_success=("success", "mean"),
                   n_def_plays=("epa", "size"))
              .reset_index()
              .rename(columns={"defteam": "team"}))

    # Higher off_epa is better for offense, lower def_epa is better for defense

    # Inner join: a team must appear on both sides of a game it played.
    return off.merge(def_, on=keys + ["team"], how="inner")


def sanity_checks(tg: pd.DataFrame) -> None:
    # Assertions that would catch a broken groupby or a bad merge
    per_game = tg.groupby("game_id").size()
    assert (per_game == 2).all(), \
        f"expected 2 rows per game:\n{per_game[per_game != 2].head()}"

    assert tg.duplicated(["game_id", "team"]).sum() == 0, "duplicate team in a game"
    assert tg[["off_epa", "def_epa"]].notna().all().all(), "null EPA"

    lo, hi = tg.n_off_plays.min(), tg.n_off_plays.max()
    assert 30 <= lo and hi <= 100, f"implausible play counts: {lo}-{hi}"

    # Conservation law: Every offensive play is some defense's play, so the
    # collection of off_epa values and the collection of def_epa values are the
    # same numbers in a different order. If these means differ, one of the two
    # groupings is wrong or the merge dropped rows and nothing downstream can
    # be trusted. This is the check that actually proves the table is correct.
    diff = abs(tg.off_epa.mean() - tg.def_epa.mean())
    assert diff < 1e-9, f"conservation violated: off-def = {diff:.2e}"

    print("  sanity checks passed")


def report(tg: pd.DataFrame) -> None:
    print(f"\n{'='*58}\nTEAM-GAME TABLE\n{'='*58}")
    print(f"rows        : {len(tg):,}  ({len(tg)//2:,} games)")
    print(f"seasons     : {tg.season.min()}-{tg.season.max()}")
    print(f"teams       : {tg.team.nunique()}")
    print(f"plays/game  : {tg.n_off_plays.mean():.1f} avg "
          f"({tg.n_off_plays.min()}-{tg.n_off_plays.max()})")
    print(f"conservation: off {tg.off_epa.mean():+.6f} == def {tg.def_epa.mean():+.6f}")

    latest = tg[tg.season == tg.season.max()]
    print(f"\n--- best offenses, {tg.season.max()} (EPA/play) ---")
    print(latest.groupby("team").off_epa.mean().nlargest(5).round(3).to_string())
    print(f"\n--- best defenses, {tg.season.max()} (EPA/play allowed, lower better) ---")
    print(latest.groupby("team").def_epa.mean().nsmallest(5).round(3).to_string())


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {len(SEASONS)} seasons")
    frames = [team_game_stats(load_pbp(s)) for s in SEASONS]
    tg = pd.concat(frames, ignore_index=True)
    tg = tg.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)

    sanity_checks(tg)
    report(tg)

    tg.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
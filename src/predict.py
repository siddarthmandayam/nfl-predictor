"""
predict_week.py  --  Phase 5

Generates predictions for an upcoming week and writes them to predictions/ with a
UTC timestamp. Files are committed before kickoff and never edited afterward, so
git history is the audit trail.

The new problem this file solves: features.py attaches a team's rolling form to a
game that has already happened. Here the game hasn't happened, so each team's
*current* form has to be carried forward to a future opponent. That is a different
code path, and it is the natural place for leakage to reappear at exactly the
moment it matters. serve_train_consistency_check() exists to catch that.

Usage:
    python src/predict_week.py            # next unplayed week
    python src/predict_week.py 2026 1     # explicit season and week
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import STATS, WINDOW
from model import FEATURES, build_model

TEAM_GAMES = Path("data/processed/team_games.parquet")
FEATS = Path("data/processed/features.parquet")
UPCOMING = Path("data/processed/upcoming.parquet")
PRED_DIR = Path("predictions")

# Refit policy: the model is refit from scratch every run, on every completed
# game available at that moment. As 2026 games finish they enter the training
# set. This is a deliberate choice, not an accident. It is stated here so that
# a prediction can always be reproduced from the data available on its timestamp.
REFIT_EVERY_RUN = True


def current_form(tg: pd.DataFrame, before=None) -> pd.DataFrame:
    """
    Each team's mean stats over their most recent WINDOW completed games.

    This is the same quantity features.py computes with shift(1) -- the mean of a
    team's last WINDOW games, excluding the one being predicted. Here the game
    being predicted simply doesn't exist yet, so no shift is needed: every row in
    `tg` is already in the past.

    `before` optionally restricts to games strictly earlier than a (season, week)
    tuple, which is what the consistency check uses.
    """
    tg = tg.sort_values(["team", "season", "week"])

    if before is not None:
        season, week = before
        tg = tg[(tg.season < season) | ((tg.season == season) & (tg.week < week))]

    recent = tg.groupby("team").tail(WINDOW)
    return recent.groupby("team")[STATS].mean()


def serve_train_consistency_check(tg: pd.DataFrame, feat: pd.DataFrame) -> None:
    """
    Prove the serving path produces the same numbers as the training path.

    Takes a real completed game, recomputes the home team's form using only games
    strictly before it, and asserts it matches the feature features.py assigned to
    that game. If the two paths ever diverge, the model is being served inputs it
    was never trained on -- which would not raise an error anywhere else.
    """
    g = feat.sort_values(["season", "week"]).iloc[-1]
    form = current_form(tg, before=(g.season, g.week))

    for stat in STATS:
        served = form.loc[g.home_team, stat]
        trained = g[f"home_{stat}_r{WINDOW}"]
        assert np.isclose(served, trained), (
            f"TRAIN/SERVE MISMATCH on {stat} for {g.home_team} "
            f"{g.season} wk{g.week}: serving path {served:.6f}, "
            f"training path {trained:.6f}"
        )

    print(f"  train/serve consistency verified "
          f"({g.home_team} {int(g.season)} wk{int(g.week)})")


def build_matchup_features(games: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    # Attach each team's current form to their upcoming game, home and away
    out = games.copy()

    for side in ("home", "away"):
        for stat in STATS:
            out[f"{side}_{stat}_r{WINDOW}"] = [
                form.loc[t, stat] if t in form.index else np.nan
                for t in out[f"{side}_team"]
            ]

    # Identical construction to features.py. If these ever drift apart the
    # consistency check above will not catch it, so keep them in one place if
    # this grows.
    out["epa_edge_off"] = out[f"home_off_epa_r{WINDOW}"] - out[f"away_off_epa_r{WINDOW}"]
    out["epa_edge_def"] = out[f"away_def_epa_r{WINDOW}"] - out[f"home_def_epa_r{WINDOW}"]
    out["rest_edge"] = out.home_rest - out.away_rest

    return out


def main() -> None:
    tg = pd.read_parquet(TEAM_GAMES)
    feat = pd.read_parquet(FEATS)
    upcoming = pd.read_parquet(UPCOMING)

    if len(sys.argv) == 3:
        season, week = int(sys.argv[1]), int(sys.argv[2])
    else:
        nxt = upcoming.sort_values(["season", "week"]).iloc[0]
        season, week = int(nxt.season), int(nxt.week)

    slate = upcoming[(upcoming.season == season) & (upcoming.week == week)]
    if slate.empty:
        sys.exit(f"no upcoming games for {season} week {week}")

    print(f"predicting {season} week {week} -- {len(slate)} games")
    print(f"  training on {len(feat):,} games through "
          f"{int(feat.season.max())}")

    serve_train_consistency_check(tg, feat)

    model = build_model().fit(feat[FEATURES], feat.home_win)

    form = current_form(tg)
    X = build_matchup_features(slate, form)

    missing = X[FEATURES].isna().any(axis=1)
    if missing.any():
        print(f"  WARNING: {missing.sum()} games dropped for missing features")
        X = X[~missing]

    X["home_win_prob"] = model.predict_proba(X[FEATURES])[:, 1]

    stamp = datetime.now(timezone.utc)
    out = pd.DataFrame({
        "generated_at_utc": stamp.isoformat(timespec="seconds"),
        "game_id": X.game_id,
        "season": X.season,
        "week": X.week,
        "gameday": X.gameday,
        "away_team": X.away_team,
        "home_team": X.home_team,
        "home_win_prob": X.home_win_prob.round(4),
        "spread_line": X.spread_line,
        "pick": np.where(X.home_win_prob > 0.5, X.home_team, X.away_team),
    }).sort_values("gameday")

    print(f"\n{'='*62}")
    print(f"{season} WEEK {week}   generated {stamp:%Y-%m-%d %H:%M} UTC")
    print("=" * 62)
    for _, r in out.iterrows():
        conf = max(r.home_win_prob, 1 - r.home_win_prob)
        print(f"  {r.gameday}  {r.away_team:>3} @ {r.home_team:<3}  "
              f"P(home)={r.home_win_prob:.3f}  pick {r.pick:<3} ({conf:.0%})  "
              f"line {r.spread_line:+.1f}")

    PRED_DIR.mkdir(exist_ok=True)
    path = PRED_DIR / f"{season}_week{week:02d}.csv"
    if path.exists():
        sys.exit(f"\n{path} already exists -- predictions are append-only. "
                 f"Delete it deliberately if you really mean to regenerate.")

    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print("Commit this file before kickoff. Do not edit it afterward.")


if __name__ == "__main__":
    main()
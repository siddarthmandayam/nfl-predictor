from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import calibration_plot, calibration_table, compare, market_prob

FEATS = Path("data/processed/features.parquet")
OUT = Path("data/processed/oos_predictions.parquet")

FEATURES = [
    "epa_edge_off", "epa_edge_def", "rest_edge",
    "home_off_epa_r5", "away_off_epa_r5",
    "home_def_epa_r5", "away_def_epa_r5",
]

# First test season. Starting later gives the first fit a real training set 
# instead of a single season.
FIRST_TEST_SEASON = 2010


def build_model():
    # Scaler + logistic regression. The pipeline keeps the scaler honest: it is
    # fit during .fit() on training rows only, and merely applied during .predict
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def walk_forward(feat: pd.DataFrame) -> pd.DataFrame:
    # Train through season N-1, predict season N, roll forward
    seasons = sorted(feat.season.unique())
    test_seasons = [s for s in seasons if s >= FIRST_TEST_SEASON]

    rows = []
    print(f"{'test':>6} {'train':>7} {'n':>5}  {'logloss':>8} {'acc':>7}")
    print("-" * 38)

    for s in test_seasons:
        train = feat[feat.season < s]
        test = feat[feat.season == s]
        if test.empty:
            continue

        model = build_model().fit(train[FEATURES], train.home_win)
        probs = model.predict_proba(test[FEATURES])[:, 1]

        ll = -np.mean(test.home_win * np.log(probs)
                      + (1 - test.home_win) * np.log(1 - probs))
        acc = ((probs > 0.5) == (test.home_win == 1)).mean()
        print(f"{s:>6} {len(train):>7,} {len(test):>5,}  {ll:>8.4f} {acc:>7.4f}")

        out = test[["game_id", "season", "week", "home_team", "away_team",
                    "home_win", "home_moneyline", "away_moneyline"]].copy()
        out["prob"] = probs
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def show_coefficients(feat: pd.DataFrame) -> None:
    model = build_model().fit(feat[FEATURES], feat.home_win)
    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values(key=abs,
                                                                     ascending=False)
    print("\n--- coefficients (standardized; sign is what matters) ---")
    for name, c in coefs.items():
        print(f"  {name:<20} {c:+.4f}")
    print(f"  {'intercept':<20} {model[-1].intercept_[0]:+.4f}")
    print("\n  Expected: epa_edge_off and epa_edge_def positive (both are built so")
    print("  that larger favors the home team). A negative sign means a feature is")
    print("  constructed backwards.")


def main() -> None:
    feat = pd.read_parquet(FEATS)
    print(f"loaded {len(feat):,} games with features "
          f"({feat.season.min()}-{feat.season.max()})\n")

    oos = walk_forward(feat)

    # Score the market on exactly the rows the model predicted. Games without a
    # moneyline are dropped from BOTH sides, not just the market's.
    oos["market"] = market_prob(oos)
    both = oos.dropna(subset=["market"]).copy()

    print(f"\n{'='*58}\nOUT-OF-SAMPLE RESULTS\n{'='*58}")
    print(f"model predictions : {len(oos):,} games "
          f"({oos.season.min()}-{oos.season.max()})")
    print(f"scored vs market  : {len(both):,} games (both available)\n")

    print(compare(both.home_win, {
        "model": both.prob.values,
        "market": both.market.values,
        "base rate": np.full(len(both), feat.home_win.mean()),
    }).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- model calibration (out-of-sample) ---")
    print(calibration_table(both.home_win, both.prob.values).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    calibration_plot(both.home_win,
                     {"model": both.prob.values, "market": both.market.values},
                     path=Path("reports/calibration_model.png"))

    show_coefficients(feat)

    oos.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd

GAMES = Path("data/processed/games.parquet")
REPORTS = Path("reports")

# Probabilities of exactly 0 or 1 make log loss infinite. Clip before scoring.
EPS = 1e-6


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def _metrics(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)

    acc = ((p > 0.5) == (y_true == 1)).mean()
    logloss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    brier = np.mean((p - y_true) ** 2)

    # ECE: bin by predicted probability, compare mean prediction to observed
    # rate in each bin, average the gaps weighted by bin size.
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            ece += (m.sum() / len(p)) * abs(y_true[m].mean() - p[m].mean())

    return {"n": len(p), "accuracy": acc, "log_loss": logloss,
            "brier": brier, "ece": ece}


def evaluate(y_true, y_prob, name="model", verbose=True):
    """Score one set of predictions."""
    m = _metrics(y_true, y_prob)
    m["name"] = name
    if verbose:
        print(f"{name:24s} n={m['n']:5d}  acc={m['accuracy']:.4f}  "
              f"logloss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
              f"ece={m['ece']:.4f}")
    return m


def compare(y_true, preds: dict):
    """Score several models on the same games. Sorted by log loss, best first."""
    rows = [_metrics(y_true, p) | {"model": k} for k, p in preds.items()]
    df = pd.DataFrame(rows).set_index("model").sort_values("log_loss")
    return df[["n", "accuracy", "log_loss", "brier", "ece"]]


def calibration_table(y_true, y_prob, n_bins=10):
    """Predicted vs observed win rate per bin. The core diagnostic."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                     "n": int(m.sum()),
                     "predicted": p[m].mean(),
                     "observed": y_true[m].mean()})
    out = pd.DataFrame(rows)
    out["gap"] = out.observed - out.predicted
    return out


def calibration_plot(y_true, preds: dict, path=REPORTS / "calibration.png"):
    """Diagonal = perfect. Below = overconfident. This goes in the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")

    for name, p in preds.items():
        t = calibration_table(y_true, p)
        ax.plot(t.predicted, t.observed, "o-", label=name, alpha=0.8)

    ax.set(xlabel="predicted probability", ylabel="observed win rate",
           xlim=(0, 1), ylim=(0, 1), title="Calibration: home team win probability")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------------
# market baseline
# ----------------------------------------------------------------------------

def moneyline_to_prob(ml):
    """American odds -> implied probability (still includes the bookmaker's cut)."""
    ml = np.asarray(ml, dtype=float)
    # np.where evaluates both branches, so the unused one can divide by zero at
    # ml == -100 or ml == 0. The result is discarded; silence the warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def market_prob(df):
    """
    De-vigged market probability that the home team wins.

    Raw implied probabilities sum to >1 because that margin is the book's
    profit. Normalizing by the sum removes it. Returns NaN where either
    moneyline is missing (all seasons before 2006).
    """
    ph = moneyline_to_prob(df.home_moneyline)
    pa = moneyline_to_prob(df.away_moneyline)
    total = ph + pa
    out = np.where(np.isfinite(total) & (total > 0), ph / total, np.nan)
    return pd.Series(out, index=df.index)


# ----------------------------------------------------------------------------
# self-test / baseline report
# ----------------------------------------------------------------------------

def main():
    g = pd.read_parquet(GAMES)
    y = g.home_win.values

    print(f"loaded {len(g):,} games ({g.season.min()}-{g.season.max()})\n")

    base_rate = y.mean()

    print("--- self-test ---")
    # A constant prediction at the base rate must reproduce the base rate as
    # accuracy. If this fails, the metric code is wrong, not the model.
    m = evaluate(y, np.full(len(y), base_rate), f"constant {base_rate:.4f}")
    assert abs(m["accuracy"] - base_rate) < 1e-9, "constant-baseline check failed"
    # It must also be perfectly calibrated by construction: one bin, no gap.
    assert m["ece"] < 1e-9, "constant baseline should have zero ECE"
    print("  passed: constant baseline reproduces base rate, ECE ~ 0\n")

    print("--- baselines, all games ---")
    preds = {
        "coin flip (0.500)": np.full(len(y), 0.5),
        f"base rate ({base_rate:.3f})": np.full(len(y), base_rate),
    }
    print(compare(y, preds).to_string(float_format=lambda x: f"{x:.4f}"))

    # "Always pick home" is a label, not a probability. Scoring it on log loss
    # would mean predicting 1.0 and being wrong 44% of the time -- the clip is
    # the only thing keeping it finite, and the number would be meaningless.
    # Reported as accuracy only, on purpose.
    print(f"\nalways pick home, accuracy only : {base_rate:.4f}")
    print("  (no log loss: it emits labels, not probabilities)")

    # Market baseline, restricted to games that have moneylines.
    mk = g.assign(p=market_prob(g)).dropna(subset=["p"])
    print(f"\n--- market baseline (moneyline available: {len(mk):,} games, "
          f"{mk.season.min()}-{mk.season.max()}) ---")
    ym = mk.home_win.values
    mkt_preds = {
        "market (de-vigged ML)": mk.p.values,
        f"base rate ({ym.mean():.3f})": np.full(len(ym), ym.mean()),
    }
    print(compare(ym, mkt_preds).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- market calibration ---")
    print(calibration_table(ym, mk.p.values).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    calibration_plot(ym, {"market": mk.p.values,
                          "base rate": np.full(len(ym), ym.mean())})


if __name__ == "__main__":
    main()
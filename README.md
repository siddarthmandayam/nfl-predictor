# 2026 NFL Weekly Game Predictor

A probabilistic model for predicting weekly game outcomes for the 2026 NFL regular
season, evaluated against real baselines and logged live before kickoff each week.
Will be trained on the 2002–2025 regular seasons. Live predictions begin Week 1 of
the 2026 season.

## Baselines

Every result is measured against these. Regular season, ties excluded.


| Baseline | Games | Accuracy | Log Loss |
| :--- | :---: | :---: | :---: |
| Coin flip (0.500) | 6,208 | 43.89% | 0.6931 |
| Base rate (0.561) | 6,208 | 56.11% | 0.6857 |
| Always pick Vegas favorite | 6,208 | 66.70% | — |
| Market probability (de-vigged moneyline, 2006–2025) | 5,051 | 66.52% | 0.6086 |


Home teams have won 56.11% of the previous 6208 games played in the previous 2002-2025 NFL regular seasons.
Predictions are probabilities that the home team wins; `evaluate.py` converts them to picks with a 0.5 threshold:
`acc = ((p > 0.5) == (y_true == 1)).mean()`
Therefore, a constant 0.5 coin flip never clears the 0.5 decision threshold and picks the away team every game resulting in a 0.4389 baseline accuracy.

Log loss is the primary metric. It is the only one of the four that punishes confident mistakes, and what the model is trained to minimize. The scale runs from 0.6931 (no information) down to 0.0000 (perfect foresight). The market reaches 0.6086 which is an 11.4% reduction against the base rate.

That narrow range is the central finding so far: NFL outcomes are largely irreducible. Participants with injury reports, weather, betting flow, and significant money at stake extract only a fraction of the available signal.

Honest expectations follow from that. Matching 0.6086 out-of-sample would be a strong result; 0.62–0.64 is a reasonable target. Anything below roughly 0.58 out-of-sample is treated as evidence of leakage rather than skill, and investigated as such.

Note that the base-rate model has perfect calibration (ECE = 0.0000) and no predictive value at all — it never commits to anything. Calibration is necessary but not sufficient, which is why four metrics are reported rather than one.

ATS (against the spread) performance will be tracked separately. Break-even at standard -110 juice is 52.4%; results indistinguishable from chance will be reported as such.

## Results

Walk-forward validation: for each test season, the model is fit only on seasons
strictly before it. Predictions are pooled and scored once. **3,397 games,
2010–2025, all out-of-sample.**

| Model | Accuracy | Log loss | Brier | ECE |
| :--- | :---: | :---: | :---: | :---: |
| Market (de-vigged moneyline) | 67.62% | **0.6033** | 0.2081 | 0.0191 |
| **Logistic regression on rolling EPA** | **63.03%** | **0.6405** | 0.2247 | 0.0301 |
| Base rate (0.552) | 55.20% | 0.6879 | 0.2474 | 0.0088 |

The model reduces log loss 6.9% below the base rate; the market reduces it 12.3%.
So rolling EPA alone captures roughly **56% of the market's edge** over knowing
nothing.

**Where it is weak.** The model is overconfident in the middle of its range. Shows gaps of −0.02 to −0.04 across the 0.4–0.7 bins, giving ECE of 0.0301 against the market's 0.0191. When it says 65%, home teams win 62%. Probability calibration is the obvious next experiment.

**Noise floor.** Per-season log loss ranges from 0.5957 (2012) to 0.6699 (2015) — a spread of 0.07, wider than the entire gap to the market. Any variant comparison
must be judged on the pooled figure across all sixteen test seasons. Differences under ~0.01 on a single season are not meaningful.

**Coefficients** (standardized) point the right way on all seven features:
`epa_edge_off` +0.305 is strongest, `home_def_epa_r5` −0.142 correctly encodes
that allowing more EPA lowers win probability, and `home_off_epa_r5` / 
`away_off_epa_r5` are near-symmetric at +0.217 / −0.212. Sign checking is the main
reason to start with a linear model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run in order. Each stage reads what the previous one wrote.

```bash
# schedules -> games.parquet + upcoming.parquet; prints baselines
python src/build_game_table.py

# scores baselines, writes reports/calibration.png
python src/evaluate.py

# play-by-play -> team_games.parquet (downloads ~24 seasons, cached)
python src/build_team_games.py

# rolling features -> features.parquet; runs the leakage test
python src/features.py

# walk-forward validation -> oos_predictions.parquet
python src/model.py
```

Each script runs its own assertions. If one fires, the upstream data changed and
the pipeline should not be trusted until it is resolved.

**Order matters.** `features.py` reads both `team_games.parquet` and
`games.parquet`; `model.py` reads `features.parquet`. Changing an early stage
without re-running the later ones leaves stale intermediate files that fail
confusingly — for example, widening the season range in `build_team_games.py`
without re-running `features.py` leaves the model training on the old subset.

## Data

nflverse — free, public, play-by-play data back to 1999. Schedules are read directly from nfldata/data/games.csv, which already includes closing spreads, moneylines, rest days, and venue conditions.

The nfl_data_py wrapper is deliberately not used: as of v0.3.3 it pins pandas<2.0 and numpy<2.0, which will not build on current Python. Reading the source files directly removes the dependency.

Handling decisions, all verified against the data rather than assumed:
- Team abbreviations normalized to current (OAK→LV, SD→LAC, STL→LA). These are the only three that change; nflverse uses LA and JAX.
- spread_line is positive when the home team is favored. Confirmed by asserting home win rate rises monotonically across it (18.6% at ≤ −10, up to 87.7% at > +10).
- 15 tied games dropped (0.2%). A binary classifier has nowhere to put them.
- Seasons before 2002 excluded to keep divisional alignment consistent.
- Moneylines are unavailable before 2006, so the market baseline runs on 5,051 of the 6,208 games.

Play-by-play files include playoffs (13 games/season) and are aggregated to one row per team per game. 2022 has 271 regular-season games rather than 272: the Bills–Bengals game was cancelled after Damar Hamlin's cardiac arrest and never replayed. Both facts are flagged and under test rather than absorbed into a row count.

## Features

Team strength is measured as EPA per play (expected points added), aggregated from play-by-play to one row per team per game, then averaged over each team's previous 5 games. Offensive and defensive EPA are computed separately by grouping the same plays by `posteam` and by `defteam` — so `def_epa` is EPA *allowed*, where lower is better.

Correctness is enforced by a conservation identity: league-wide mean `off_epa` must equal league-wide mean `def_epa`, since every offensive play is some defense's play. It holds to nine decimal places (−0.014295 across 12,998 team-games, 2002–2025).

**Leakage guard.** A rolling mean includes the current game unless it is shifted. `features.py` shifts by one game before rolling, and `test_no_leakage()` proves it: for a chosen team and mid-season week, it recomputes the feature by hand from strictly earlier games and asserts a match, then asserts the value is *inconsistent* with a window that includes the current game. For example, Baltimore's Week 10 2024 feature is built from weeks 5–9.

Rolling windows reset at each season boundary because rosters and schemes turn over heavily in the offseason. This costs 18.7% of games to warm-up (1,158 of 6,208, the first three weeks of every season) and is one of the variants to be tested later.

## Validation

**Leakage.** Every feature is a rolling window shifted so it contains only
information available before kickoff. Enforced by `test_no_leakage()`, which runs
on every feature build. See the Features section above.

**Temporal cross-validation.** Random k-fold trains on the future. Validation is
walk-forward: train through season *N*, test on *N+1*, roll forward, then pool the
out-of-sample predictions and score once.

The market probability is used strictly as a comparison and is never a model
feature. Including it would dominate every other input and produce an expensive
line-copier.

# 2026 NFL Weekly Game Predictor

A probabilistic model for predicting weekly game outcomes for the 2026 NFL regular
season, evaluated against real baselines and logged live before kickoff each week.
Trained on the 2002–2025 regular seasons. Live predictions begin Week 1 of the
2026 season.

## Status

**Complete and live.** All five phases built and validated. Predictions are logged
weekly through the 2026 season.

| Phase | Status |
|---|---|
| 1. Game table + validation | Done |
| 2. Evaluation harness | Done |
| 3. Rolling EPA features + leakage tests | Done |
| 4. Walk-forward model | Done |
| 5. Live weekly predictions | Live |

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

That narrow range is a central finding: NFL outcomes are largely irreducible. Participants with injury reports, weather, betting flow, and significant money at stake extract only a fraction of the available signal.

Honest expectations followed from that, written before any model existed: matching
0.6086 out-of-sample would be a strong result, 0.62–0.65 a reasonable target, and
anything below roughly 0.58 evidence of leakage rather than skill. The result came
in at 0.6448 which is inside the predicted band, and well clear of the leakage threshold.

Note that the base-rate model has perfect calibration (ECE = 0.0000) and no predictive value at all — it never commits to anything. Calibration is necessary but not sufficient, which is why four metrics are reported rather than one.

ATS (against the spread) performance will be tracked separately. Break-even at standard -110 juice is 52.4%; results indistinguishable from chance will be reported as such.

## Results

Walk-forward validation: for each test season, the model is fit only on seasons
strictly before it. Predictions are pooled and scored once. **4,161 games,
2010–2025, all out-of-sample.**

| Model | Accuracy | Log loss | Brier | ECE |
| :--- | :---: | :---: | :---: | :---: |
| Market (de-vigged moneyline) | 66.57% | **0.6094** | 0.2110 | 0.0165 |
| **Logistic regression on rolling EPA** | **62.56%** | **0.6448** | 0.2268 | 0.0222 |
| Base rate (0.554) | 55.42% | 0.6874 | 0.2471 | 0.0070 |

The market figure here (0.6094) differs from the 0.6086 in Baselines because it is
computed on the 4,161 out-of-sample games the model predicted, not all 5,051 games
with moneylines. Comparisons are always made on identical rows.

The model reduces log loss 6.2% below the base rate; the market reduces it 11.4%.
Rolling EPA alone captures roughly **55% of the market's edge** over knowing nothing.

It does not beat the market, and it was not expected to. A first-pass model that
appeared to would be evidence of leakage rather than skill.

**Where it is weak.** The model is overconfident through the middle of its range —
when it says 65%, home teams win 63%. ECE is 0.0222 against the market's 0.0165.
Probability calibration is the next experiment.

**Noise floor.** Per-season log loss ranges from 0.6178 (2011) to 0.6728 (2021), a
spread of 0.055 — wider than the 0.035 gap to the market. Variant comparisons are
therefore judged on the pooled figure across all sixteen test seasons, not on any
single season.

**Coefficients** (standardized) point the right way on all seven features:
`epa_edge_off` +0.284 is strongest; `home_def_epa_r5` −0.146 correctly encodes that
allowing more EPA lowers win probability; `home_off_epa_r5` +0.212 and
`away_off_epa_r5` −0.185 are near-symmetric with opposite signs. Sign checking is
the main reason to start with a linear model.

## Live predictions

Predictions are generated and committed **before kickoff**, then never edited. Git
history is the audit trail, every file's commit timestamp precedes the games it
predicts. `predict.py` refuses to overwrite an existing prediction file.

Week 1 2026 was logged on 2026-08-25, fifteen days before the opening game.

Each `predictions/2026_weekNN.csv` holds the generated-at UTC timestamp, both
teams, the model's home win probability, the closing spread, and the pick.

**Known weakness in early-season predictions.** Week 1 features are rolling EPA
carried over from the end of the 2025 season. The model has no knowledge of free
agency, the draft, retirements, or coaching changes. It treats these as the same
teams that finished last season. Stated here before the games rather than offered
afterward as an explanation as one of the potential weaknesses of this model.

**On disagreements with the market.** The model is less confident than the closing
line on several Week 1 games and picks against it on others. Given it scores 0.6448
against the market's 0.6094 out-of-sample, the honest prior is that where they
disagree, the market is more often right. Disagreement is not evidence of an edge.

**Train/serve consistency.** The serving path builds features differently from the
training path in carrying each team's current form forward to a game that hasn't
happened, rather than attaching form to a completed game. `predict.py` asserts the
two produce identical values on a real past game, because divergence there would
silently feed the model inputs it was never trained on.

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

# generate and log predictions for a week (refuses to overwrite)
python src/predict.py 2026 1
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

Rolling windows carry across season boundaries. This was tested against resetting
each season: no measurable difference in prediction quality, but carrying over
recovers 1,108 games (warm-up drops from 18.7% to 0.8%), improves calibration, and
makes Week 1 predictable at all. See DECISIONS.md.

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

## Repository

src/build_game_table.py schedules -> games.parquet, upcoming.parquet
src/evaluate.py metrics, calibration, market baseline
src/build_team_games.py play-by-play -> team-game EPA
src/features.py rolling features + leakage test
src/model.py walk-forward validation
src/predict.py weekly predictions + train/serve check
predictions/ timestamped weekly predictions (append-only)
DECISIONS.md running log of choices, findings, and dead ends


`predictions/` is append-only. Files are committed before games are played and
never edited afterward.
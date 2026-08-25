Running record of what I tried, what happened, and lessons learned.

## 2026-08-25: live prediction loop

**Did:** Built `src/predict.py`. Fits on all completed games, carries each team's
current rolling form forward to an upcoming matchup, writes a timestamped CSV to
`predictions/`. Generated and committed Week 1 2026 fifteen days before kickoff.

**Found:**

- **The serving path is not the training path.** `features.py` attaches a team's
  form to a game that already happened; serving carries current form forward to a
  game that hasn't. Different code, same quantity and nothing would raise an
  error if they diverged. Added `serve_train_consistency_check()`, which recomputes
  a real past game's features through the serving path and asserts they match what
  training produced. Verified on PIT 2025 wk18.
- The model disagrees with the closing line on several Week 1 games. Most notably
  NO @ DET (market ≈70% DET, model 50.2%) and ARI @ LAC (market ≈78% LAC, model
  62.5%). Given the model loses to the market out-of-sample, the honest prior is
  that the market is more often right where they disagree.

**Decided:**

- **Refit on every run**, on every completed game available at that moment. As 2026
  games finish they enter the training set. Stated as policy so any prediction can
  be reproduced from the data available at its timestamp.
- **Prediction files are immutable.** `predict.py` exits rather than overwrite an
  existing file. Regenerating requires deleting deliberately. Git timestamps are
  what make the track record auditable rather than a claim.
- **Document the Week 1 blind spot before the games.** Features are 2025
  end-of-season form; the model knows nothing of free agency, the draft, or
  coaching changes. Naming this in advance is worth more than explaining it after.

**Next:** Log predictions weekly through the season. Then the remaining variant
sweep (window length, min periods, playoff inclusion) and probability calibration,
judged on pooled log loss.

## 2026-08-23: season-boundary reset, tested and reversed

**Did:** Tested `RESET_EACH_SEASON` both ways, rebuilt features and re-ran walk-forward validation for each.

**Found:**

| | games | model | market | gap | ECE |
|---|---|---|---|---|---|
| Reset each season | 3,397 | 0.6405 | 0.6033 | 0.0372 | 0.0301 |
| Carry across seasons | 4,161 | 0.6448 | 0.6094 | 0.0354 | 0.0222 |

- **The raw log losses are not comparable — different row sets.** Carrying over
  adds 1,108 previously-dropped games, all early-season, which are harder to
  predict. The tell is that the *market* also got worse on the larger set
  (0.6033 → 0.6094). The games changed, not the model. This is the same error I
  guarded against when scoring the market on identical rows, and I nearly repeated
  it on my own experiment.
- **Gap to market is the fair comparison**, since it normalizes for game
  difficulty: 0.0354 carrying vs 0.0372 resetting. Slight edge to carrying, but far
  inside the noise floor. **No detectable quality difference.**
- Carrying over recovers 1,108 games (warm-up 18.7% → 0.8%) and improves
  calibration (ECE 0.0222 vs 0.0301).
- Noise floor now measured precisely: per-season log loss 0.6178 (2011) to 0.6728
  (2021), a 0.055 spread which is wider than the 0.035 gap to the market.

**Decided:**

- **`RESET_EACH_SEASON = False` permanently.** My reasoning was that offseason
  roster turnover means January form says little about September and I predicted
  carrying over would hurt. It doesn't, at least not measurably. Hypothesis not
  supported.
- The decision rests on operational grounds rather than accuracy: more training
  data, better calibration, and Week 1 becomes predictable at all.
- **This was forced by the live loop.** Planning surfaced that weeks 1–3 of
  2026 would have no features under season reset. Carrying over only at prediction
  time would have created train/serve skew as the model would see a feature
  distribution in production that never appeared in training. Consistency required
  changing the training config, not just the serving path.
- Fixed the warm-up print string to derive its wording from `RESET_EACH_SEASON`
  rather than hardcoding "each season," which was silently wrong after the flip.

**Next:** live weekly predictions, now unblocked for Week 1. Remaining variants (window length, min periods, playoff inclusion) still to sweep, judged on pooled log loss. The playoff question from 2026-08-21 needs this noise floor to be answerable at all: a 0.055 per-season spread means the effect of a few games inside a rolling window is invisible except in the pooled figure.

## 2026-08-23: walk-forward logistic regression

**Did:** Built `src/model.py`. Scaler + logistic regression in a pipeline, fit separately for each test season on all prior seasons only, predictions pooled and scored once against the market on identical games. 3,397 out-of-sample games, 2010–2025.

The validation scheme is the point of this file, not the model. For each test season, the model is fit only on seasons strictly before it, then predicts that season. Predictions are pooled and scored once at the end, so every number reported is out-of-sample.

Three things this file is careful about, because each silently inflates results:
  1. The scaler is fit inside the pipeline, on training data only. Fitting it on
     everything leaks the test distribution into training.
  2. The market is scored on exactly the same rows as the model. Different row
     sets make the comparison meaningless.
  3. Nothing is refit after seeing test scores.

**Found:**

- **Model: 63.03% accuracy, 0.6405 log loss, 0.2247 Brier, ECE 0.0301.**
  Market: 0.6033. Base rate: 0.6879.
- The model reduces log loss 6.9% below the base rate; the market reduces it 12.3%.
  Rolling EPA alone captures roughly **56% of the market's edge**. It does not beat
  the market and was not expected to — a first-pass model that appeared to would be
  evidence of leakage, not skill.
- **All seven coefficient signs are correct.** `epa_edge_off` +0.305 (strongest),
  `home_off_epa_r5` +0.217 vs `away_off_epa_r5` −0.212 (near-symmetric, as they
  should be), `home_def_epa_r5` −0.142 (allowing more EPA lowers win probability —
  the inverted sign I flagged, confirmed correct), `epa_edge_def` +0.138,
  `rest_edge` +0.062, `away_def_epa_r5` +0.050. This is the main argument for
  starting linear: a backwards feature would have been visible immediately.
- **The model is overconfident in the middle of its range.** Gaps of −0.02 to −0.04
  across the 0.4–0.7 bins; when it says 65%, home teams win 62%. Same direction as
  the market's miscalibration but larger (ECE 0.0301 vs 0.0191).
- **Noise floor is large.** Per-season log loss ranges 0.5957 (2012) to 0.6699
  (2015) — a 0.07 spread, wider than the entire gap to the market. Variant
  comparisons must use the pooled figure across all sixteen test seasons.
  Differences under ~0.01 on a single season are meaningless.
- Redundant features: `epa_edge_off` is exactly `home_off_epa_r5 − away_off_epa_r5`,
  and all three are in the model. Logistic regression handles the collinearity but
  splits the coefficient across correlated inputs, muddying interpretation.

**Decided:**

- **Scaler goes inside the pipeline**, so it is fit on training rows only. Fitting
  it on the full dataset would leak the test distribution into training.
- **Market scored on identical rows.** Games without a moneyline are dropped from
  both sides, not just the market's. 3,397 of 3,398 games qualify.
- **First test season is 2010**, so the first fit trains on eight seasons rather
  than one.
- **Report the loss to the market plainly.** The honest finding is that rolling EPA
  captures about half the market's edge. Framing that as a failure would be wrong,
  and hiding it would be worse.
- **A separate full-data fit is used only to read coefficients**, never scored. Its
  numbers appear nowhere in the results.

**Next:** the live weekly loop, before Week 1. Then the variant sweep
(window length, min periods, season reset, playoff inclusion) and probability
calibration, judged on pooled log loss. The playoff-inclusion question from 2026-08-21 needs this noise floor to be answerable at all — a 0.07 per-season spread means the effect of a few games in a rolling window is invisible except in the pooled figure.

## 2026-08-23: backfill to 2002

**Did:** Extended `build_team_games.py` from 3 seasons to 24 (2002–2025) and
re-ran the feature build.

**Found:**

- 12,998 team-game rows / 6,499 games; **5,050 games with features** after warm-up,
  up from 670. Warm-up rises slightly to 18.7% (1,158 of 6,208) because earlier
  seasons have 16-game schedules.
- **The join dropped zero games**. 5,050 + 1,158 = 6,208 exactly. I expected losses
  from historical abbreviations (`OAK`/`SD`/`STL` in play-by-play vs. normalized
  names in `games.parquet`).
- **Reason: nflverse already standardizes `posteam`/`defteam` to current
  abbreviations**, even in 2010 data — it returns `LV`, `LAC`, `LA`, `JAX`. But
  `game_id` still carries the historical name (`2010_01_ARI_STL`). Two conventions
  in one file. The join worked by luck, not design; now verified rather than assumed.
- Conservation holds exactly on the full range: −0.014295.
- Feature separation is cleaner with more data: by quintile of `epa_edge_off`, home
  win rate runs 37.4% / 46.4% / 57.0% / 63.8% / 75.7%. Still in-sample.

**Decided:** Backfill before modeling. Three seasons gave one usable test season,
which is not enough to trust any walk-forward result.

**Next:** Logistic regression, walk-forward validation, scored against the
market on identical test games.

## 2026-08-23: rolling features

**Did:** Built `src/features.py` which team_games.parquet into pre-kickoff features and joins them onto games. Rolling 5-game means of offensive and defensive EPA and success rate, shifted one game, joined onto games.parquet as `home_*` and `away_*`, plus matchup difference features. Wrote `test_no_leakage()` as an assertion inside the module so it runs every time.

The only thing that matters in this file is `.shift(1)`. A rolling mean over a team's last N games *includes the current game* unless you shift it. Without the shift, a Week 8 feature contains Week 8's result. That single character is the leakage guard, and `test_no_leakage()` below exists to prove it is working.

**Found:**

- **Leakage test passes:** BAL's 2024 Week 10 feature is built from weeks 5–9.
  The test hand-computes the mean from strictly earlier games and asserts a match,
  then asserts the value is *inconsistent* with a window containing the current
  game. Off-by-one is how this bug presents, so a single equality check isn't
  enough and the second assertion is the one that catches it.
- **Features separate outcomes.** By quintile of `epa_edge_off`, home win rate
  runs 36.6% / 48.5% / 57.5% / 61.2% / 76.9%. Monotonic across all five.
  **This is in-sample and descriptive, not a result**. It says the features
  correlate with outcomes, not that a model generalizes. The honest number comes
  from held-out seasons.
- **Warm-up costs 17.6% of games** (143 of 813). Resetting windows each season
  plus requiring 3 prior games means weeks 1–3 have no features for any team.
  This is the quantified price of the season-reset choice.
- Only 670 usable games from 2022–2024 — about 168 per season. Too few for a
  meaningful walk-forward split.
- - *(Superseded by the 2026-08-23 backfill: 5,050 games, 18.7% warm-up.)*

**Decided:**

- **Shift before rolling**, inside the groupby, so the window can only see earlier
  rows and one team's games can't bleed into another's at a team boundary.
- **Reset windows at season boundaries** (`RESET_EACH_SEASON = True`). Rosters,
  coaches and schemes turn over heavily in the offseason, so January form is weak
  evidence about September. Costs 17.6% of games. **To be tested, not assumed.**
- **WINDOW = 5, MIN_PERIODS = 3 are provisional.** Not derived from anything.
- **Add explicit difference features** (`epa_edge_off`, `epa_edge_def`,
  `rest_edge`). A logistic model can form these itself, but stating them makes the
  coefficients directly readable. Positive favors the home team.
- **Market probability still excluded** from features.

**Next:** backfill `build_team_games.py` to 2002 before modeling. 670 games gives
only one usable test season, and the full range should yield roughly 5,000. Then do logistic regression with walk-forward validation, compared against the market on identical test games.

## 2026-08-21: team-game aggregation

Built `src/build_team_games.py`. Downloads and caches nflverse play-by-play for 2022–2024, filters to scrimmage plays, aggregates to one row per team per game with offensive and defensive EPA. No rolling yet. `src/build_game_table.py` writes data/processed/games.parquet (played games with the target) and upcoming.parquet (the 2026 schedule). Both scripts run their own assertions; if one fires, the upstream data changed and the pipeline should not be trusted until it is resolved.

**Found:**

- **1,708 rows / 854 games, not the 816 expected** from 3 × 272. Two causes, both
  real: play-by-play includes playoffs (39 games over three seasons), and 2022 had
  only 271 regular-season games since the Bills–Bengals game was cancelled after Damar Hamlin's cardiac arrest and never made up.
- My sanity checks passed anyway. They verify internal consistency (2 rows per
  game, conservation, plausible play counts) but cannot verify that I loaded the
  data I intended to load. Worth remembering: assertions catch corruption, not
  wrong assumptions.
- **Conservation check holds exactly:** mean off_epa = mean def_epa = −0.010733.
  Every offensive play is some defense's play, so these are the same numbers in a
  different order. This is the check that actually proves the double-groupby and
  merge are correct.
- 2024 leaders come out BAL 0.224 / BUF 0.191 / DET 0.166 on offense, PHI −0.116 /
  DEN −0.101 / MIN −0.086 on defense. Matches reality, so the grouping is right.
- Mean 62.1 scrimmage plays per team-game, range 33–95.

**Decided:**

- **Filter to `pass` and `run` plays only.** Punts, kickoffs and field goals carry
  extreme EPA values that swamp the signal.
- **`def_epa` is EPA allowed, so lower is better** — the opposite direction from
  `off_epa`. Noted explicitly because it's a sign error waiting to happen in the
  rolling features.
- **Drop rows where `posteam` or `defteam` is null** (timeouts, end-of-period
  markers) to avoid a phantom team appearing in the groupby.
- **Playoff games: TESTING — reasoning at the end of this entry.** Not a leakage
  question: a January game precedes the following September, so including it in a
  Week 1 feature uses only past information. The real tradeoff is that playoff
  games are recent evidence of team strength, but they're a biased sample of teams
  and the target is regular-season only.
- **Start with three seasons** for iteration speed; backfill to 2002 once the
  rolling features are validated.

**Next:** rolling features. Sort by team and date, take a rolling mean over the previous N games, `.shift(1)` so the current game is excluded, then join twice onto `games.parquet` as home and away. Write the leakage test first: select a mid-season game, assert every feature value uses only data from strictly earlier weeks.

Also going to test both with playoffs and without playoffs. While playoffs is more data, the concern is that the playoffs usually consists of the best teams in the NFL. As a result, I suspect games will be tougher and EPA for teams will be lower against good competition. Therefore, using playoff data may result in good teams being penalized which could lead to bias.

## 2026-08-19: evaluate.py

Built src/evaluate.py. Reusable scoring functions (accuracy, log loss, Brier, ECE), calibration tables and plots, and moneyline-to-probability conversion. Written before any model exists, so no metric was selected after seeing results. Verified on my own machine; output matches expectations exactly.

Four numbers, answering different questions:

| Metric | What it asks | Notes |
| :--- | :--- | :--- |
| **Accuracy** | Did the pick win? | Ignores confidence entirely. Least informative. |
| **Log loss** | How good were the probabilities? | Punishes confident mistakes hard. **Primary metric.** |
| **Brier** | Mean squared error on probabilities | Gentler than log loss on confident mistakes. Sanity companion. |
| **ECE** | When you say 70%, do you win 70%? | A model can score well on log loss and still be miscalibrated. |

Market baseline: 66.52% accuracy, 0.6086 log loss, 0.2108 Brier, ECE 0.0186 over 5,051 games. This is the number to beat.
The achievable range is narrow. No information = 0.6931 (that's −ln(0.5)), base rate = 0.6857, market = 0.6086, perfect = 0.0000. The market is an 11.4% log loss reduction over the base rate. Most NFL variance is irreducible.
Perfect calibration is trivially achievable and worthless. The base-rate model has ECE = 0.0000 because it never commits to anything. Calibration is necessary but not sufficient; four metrics get reported for this reason.
Coin flip scores 43.89% accuracy, not 50%. A constant 0.5 never satisfies `p > 0.5` in the accuracy calculation in `evaluate.py`, so it picks away every game, and away teams win 43.89% (the complement of 56.11%). A thresholding artifact. Notable because real models produce near 0.5 predictions constantly, and accuracy treats 0.502 and 0.498 as opposite picks while log loss treats them as nearly identical. Reinforces log loss as the primary metric since it needs no threshold, so no arbitrary tie-break affects the score.
Vig averages 2.72% (raw implied probabilities sum to 1.0272 before removal).
Possible market miscalibration on modest home favorites: predicted 0.65 → observed 0.617. Only the 0.6–0.7 bin is individually notable (z = −2.13); pooled over 0.4–0.7, z = −2.49.

All in all, log loss is and should be the primary metric. It's the only one of the four that punishes confident errors, and it's what a logistic model is trained to minimize so scoring on it is coherent with fitting on it. Accuracy is reported only next to baselines.

Market probability should never be a model feature. It would dominate every other input and produce an expensive line-copier. Comparison only.

Market baseline is restricted to 2006+. Moneylines don't exist before then, so it covers 5,051 of 6,208 games. Documented rather than silently dropped.

Next: create src/build_team_games.py which looks at one row per team per game with offensive and defensive EPA from play-by-play. No rolling yet. Validate with the conservation check: league-wide mean off_epa must equal league-wide mean def_epa, since every offensive play is some defense's play. That identity can only hold if both groupings are correct.

## 2026-08-17: build_game_table.py

Did: Built src/build_game_table.py. Reads nflverse schedules, normalizes team abbreviations, filters to regular season, builds the home_win target, runs sanity assertions, writes games.parquet (6,208 played games) and upcoming.parquet (272 scheduled 2026 games).

**Found:**

- Baselines, 6,208 games (2002–2025), ties excluded: always pick home 56.11%, always pick the Vegas favorite 66.70%. Every game in range has a closing spread, so there's no sampling excuse for omitting the comparison.
- spread_line is positive when the home team is favored, and home win rate is monotonic across it with no reversals.
- The two bins straddling zero read 46.2% and 54.7%. Close games are near coin flips, which is a useful reality check on the whole project.
- Only three abbreviations ever change: OAK→LV, SD→LAC, STL→LA. nflverse uses LA (not LAR) and JAX (not JAC); WAS is stable throughout. Smaller problem than expected.
- 15 tied games out of 6,967 (0.2%).
- Game count per season changes: 256 through 2020, 272 from 2021 (16 → 17 games per team). Matters for any per-season aggregate.
- The full 2026 schedule is already published, so the live loop can be built before the season starts rather than during it.

**Decided:**

- Skip nfl_data_py. v0.3.3 pins pandas<2.0 and numpy<2.0, so pip tries to build pandas 1.x from source and fails on current Python. Reading the nflverse CSV directly works, removes a dependency, and removes a version-conflict risk.
- Start at 2002, the first season with the current 32-team / 8-division alignment. Keeps scheduling structure consistent.
- Drop ties rather than encoding them as 0.5. At 0.2% it cannot matter, and it keeps the target binary. Documented in the README so it isn't a hidden choice.
- Assert in the pipeline, don't eyeball. Checks for duplicate game_id, a team appearing twice in one week, implausible home win rate, and spread monotonicity. All fail loudly on upstream schema drift.
- Write parquet, not CSV, for intermediate files because types are preserved, so gameday stays a real datetime instead of being re-parsed on every read.
- Report accuracy only alongside baselines. 66.70% is the real bar; an accuracy figure without that context is noise.
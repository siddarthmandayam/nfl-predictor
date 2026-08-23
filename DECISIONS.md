Running record of what I tried, what happened, and lessons learned.

## 2026-08-23 — backfill to 2002

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
  from held-out seasons in Phase 4.
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

Built `src/build_team_games.py`. Downloads and caches nflverse play-by-play for 2022–2024, filters to scrimmage plays, aggregates to one row per team per game with offensive and defensive EPA. No rolling yet.

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
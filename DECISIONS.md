Running record of what I tried, what happened, and lessons learned.

08-19-2026: evaluate.py

Built src/evaluate.py. Reusable scoring functions (accuracy, log loss, Brier, ECE), calibration tables and plots, and moneyline-to-probability conversion. Written before any model exists, so no metric was selected after seeing results. Verified on my own machine; output matches expectations exactly.

Market baseline: 66.52% accuracy, 0.6086 log loss, 0.2108 Brier, ECE 0.0186 over 5,051 games. This is the number to beat.
The achievable range is narrow. No information = 0.6931 (that's −ln(0.5)), base rate = 0.6857, market = 0.6086, perfect = 0.0000. The market is an 11.4% log loss reduction over the base rate. Most NFL variance is irreducible.
Perfect calibration is trivially achievable and worthless. The base-rate model has ECE = 0.0000 because it never commits to anything. Calibration is necessary but not sufficient; four metrics get reported for this reason.
Coin flip scores 43.89% accuracy, not 50%. A constant 0.5 never satisfies p > 0.5 (see line 40 of evaluate.py), so it picks away every game, and away teams win 43.89% (the complement of 56.11%). A thresholding artifact. Notable because real models produce near 0.5 predictions constantly, and accuracy treats 0.502 and 0.498 as opposite picks while log loss treats them as nearly identical. Reinforces log loss as the primary metric since it needs no threshold, so no arbitrary tie-break affects the score.
Vig averages 2.72% (raw implied probabilities sum to 1.0272 before removal).
Possible market miscalibration on modest home favorites: predicted 0.65 → observed 0.617. Only the 0.6–0.7 bin is individually notable (z = −2.13); pooled over 0.4–0.7, z = −2.49.

All in all, log loss is and should be the primary metric. It's the only one of the four that punishes confident errors, and it's what a logistic model is trained to minimize so scoring on it is coherent with fitting on it. Accuracy is reported only next to baselines.

Market probability should never be a model feature. It would dominate every other input and produce an expensive line-copier. Comparison only.

Market baseline is restricted to 2006+. Moneylines don't exist before then, so it covers 5,051 of 6,208 games. Documented rather than silently dropped.

Next: create src/build_team_games.py which looks at one row per team per game with offensive and defensive EPA from play-by-play. No rolling yet. Validate with the conservation check: league-wide mean off_epa must equal league-wide mean def_epa, since every offensive play is some defense's play. That identity can only hold if both groupings are correct.

08-17-2026: build_game_table.py

Did: Built src/build_game_table.py. Reads nflverse schedules, normalizes team abbreviations, filters to regular season, builds the home_win target, runs sanity assertions, writes games.parquet (6,208 played games) and upcoming.parquet (272 scheduled 2026 games).

Found:

Baselines, 6,208 games (2002–2025), ties excluded: always pick home 56.11%, always pick the Vegas favorite 66.70%. Every game in range has a closing spread, so there's no sampling excuse for omitting the comparison.
spread_line is positive when the home team is favored, and home win rate is monotonic across it with no reversals.
The two bins straddling zero read 46.2% and 54.7%. Close games are near coin flips, which is a useful reality check on the whole project.
Only three abbreviations ever change: OAK→LV, SD→LAC, STL→LA. nflverse uses LA (not LAR) and JAX (not JAC); WAS is stable throughout. Smaller problem than expected.
15 tied games out of 6,967 (0.2%).
Game count per season changes: 256 through 2020, 272 from 2021 (16 → 17 games per team). Matters for any per-season aggregate.
The full 2026 schedule is already published, so the live loop can be built before the season starts rather than during it.

Decided:

Skip nfl_data_py. v0.3.3 pins pandas<2.0 and numpy<2.0, so pip tries to build pandas 1.x from source and fails on current Python. Reading the nflverse CSV directly works, removes a dependency, and removes a version-conflict risk.
Start at 2002, the first season with the current 32-team / 8-division alignment. Keeps scheduling structure consistent.
Drop ties rather than encoding them as 0.5. At 0.2% it cannot matter, and it keeps the target binary. Documented in the README so it isn't a hidden choice.
Assert in the pipeline, don't eyeball. Checks for duplicate game_id, a team appearing twice in one week, implausible home win rate, and spread monotonicity. All fail loudly on upstream schema drift.
Write parquet, not CSV, for intermediate files because types are preserved, so gameday stays a real datetime instead of being re-parsed on every read.
Report accuracy only alongside baselines. 66.70% is the real bar; an accuracy figure without that context is noise.
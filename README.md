2026 NFL Weekly Game Predictor

A probabilistic model for predicting weekly game outcomes for the 2026 NFL regular season, evaluated against real baselines and logged live before kickoff each week. Trained on previous 2002–2025 regular seasons. Live predictions begin Week 1 of the 2026 season.

Baselines

Every result is measured against these. Regular season, ties excluded.

| Baseline |	Games |	Accuracy |	Log Loss |
| Coin flip (0.500) |	6,208 |	43.89% |	0.6931 |
| Base rate (0.561) |	6,208 |	56.11% |	0.6857 |
| Always pick Vegas favorite |	6,208 |	66.70% |	— |
| Market probability (de-vigged moneyline, 2006–2025) |	5,051 |	66.52% |	0.6086 |


| Baseline | Games | Accuracy | Log Loss |
| :--- | :---: | :---: | :---: |
| Coin flip (0.500) | 6,208 | 43.89% | 0.6931 |
| Base rate (0.561) | 6,208 | 56.11% | 0.6857 |
| Always pick Vegas favorite | 6,208 | 66.70% | — |
| Market probability (de-vigged moneyline, 2006–2025) | 5,051 | 66.52% | 0.6086 |


Home teams have won 56.11% of the previous 6208 games played in the previous 2002-2025 NFL regular seasons.
My model outputs a probability that the home team wins and determines whether to pick the home team if greater than 50% in evaluate.py line 40:
acc = ((p > 0.5) == (y_true == 1)).mean()
Therefore, a constant 0.5 coin flip never clears the 0.5 decision threshold and picks the away team every game resulting in a 0.4389 baseline accuracy.

Log loss is the primary metric. It is the only one of the four that punishes confident mistakes, and what the model is trained to minimize. The scale runs from 0.6931 (no information) down to 0.0000 (perfect foresight). The market reaches 0.6086 which is an 11.4% reduction against the base rate.

That narrow range is the central finding so far: NFL outcomes are largely irreducible. Participants with injury reports, weather, betting flow, and significant money at stake extract only a fraction of the available signal.

Honest expectations follow from that. Matching 0.6086 out-of-sample would be a strong result; 0.62–0.64 is a reasonable target. Anything below roughly 0.58 out-of-sample is treated as evidence of leakage rather than skill, and investigated as such.

Note that the base-rate model has perfect calibration (ECE = 0.0000) and no predictive value at all — it never commits to anything. Calibration is necessary but not sufficient, which is why four metrics are reported rather than one.

ATS (against the spread) performance will be tracked separately. Break-even at standard -110 juice is 52.4%; results indistinguishable from chance will be reported as such.

Setup

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/build_game_table.py    # builds the game table, prints baselines
python src/evaluate.py            # scores baselines, writes calibration plot

build_game_table.py writes data/processed/games.parquet (played games with the target) and upcoming.parquet (the 2026 schedule). Both scripts run their own assertions; if one fires, the upstream data changed and the pipeline should not be trusted until it is resolved.

Data

nflverse — free, public, play-by-play data back to 1999. Schedules are read directly from nfldata/data/games.csv, which already includes closing spreads, moneylines, rest days, and venue conditions.

The nfl_data_py wrapper is deliberately not used: as of v0.3.3 it pins pandas<2.0 and numpy<2.0, which will not build on current Python. Reading the source files directly removes the dependency.

Handling decisions, all verified against the data rather than assumed:
Team abbreviations normalized to current (OAK→LV, SD→LAC, STL→LA). These are the only three that change; nflverse uses LA and JAX.
spread_line is positive when the home team is favored. Confirmed by asserting home win rate rises monotonically across it (18.6% at ≤ −10, up to 87.7% at > +10).
15 tied games dropped (0.2%). A binary classifier has nowhere to put them.
Seasons before 2002 excluded to keep divisional alignment consistent.
Moneylines are unavailable before 2006, so the market baseline runs on 5,051 of the 6,208 games.

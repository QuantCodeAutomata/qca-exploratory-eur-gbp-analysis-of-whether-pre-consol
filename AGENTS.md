# Repository Knowledge — qca-exploratory-eur-gbp-analysis

## Project Overview
Replication of a quantitative finance paper studying whether pre-consolidation ADX predicts
breakout success on FX 1-minute data.

## Experiments
- **exp_1**: EUR/GBP exploratory heatmap analysis (Oct 2017 – Dec 2018)
- **exp_2**: GBP/USD out-of-sample three-strategy comparison (same calendar length)

## Key Ambiguities Logged
1. Paper says "one year" but Oct 2017–Dec 2018 is 15 months → use full 15-month span
2. Width rule: paper says 15 pips in flowchart, 8-15 pips in method section → use 8-15 pips
3. ADX formulas in paper are nonstandard → use Wilder-style 14-period ADX via `ta` library
4. ADX averaging window = floor(0.1 * consolidation_length_minutes)
5. ADX >= 25 = strong trend (Discrete strategy)
6. Entry at first bar close outside consolidation range
7. Intrabar tie-break: if both stop and target hit in same bar, stop wins (conservative)
8. GBP/USD dates: mirror EUR/GBP span (Oct 2017 – Dec 2018) as documented assumption

## Module Structure
- `src/data_loader.py` — massive API data fetching
- `src/indicators.py` — ATR, ADX computation (using `ta` library)
- `src/consolidation.py` — custom consolidation detection algorithm
- `src/simulation.py` — trade simulation for all three strategies
- `src/exp1_heatmap.py` — Experiment 1 main runner
- `src/exp2_strategies.py` — Experiment 2 main runner
- `tests/test_indicators.py` — indicator tests
- `tests/test_consolidation.py` — consolidation algorithm tests
- `tests/test_simulation.py` — simulation tests
- `run_experiments.py` — top-level runner

## Libraries Used
- `ta` (ADX, ATR) — Context7 confirmed
- `matplotlib`, `seaborn` — visualizations
- `pandas`, `numpy` — data manipulation
- `massive` — FX data retrieval
- `pytest` — testing

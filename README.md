# Portfolio Playoff — Quantitative Research Pipeline

Reproducible factor research for the **Portfolio Playoff** competition at IIT Bombay.

This repository contains the complete implementation of Experiments 001–004A:
systematic evaluation of momentum, earnings momentum, factor complementarity,
and quality signals on the official 30-stock competition universe.

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Required Data](#required-data)
5. [Validating the Environment](#validating-the-environment)
6. [Running Experiments](#running-experiments)
7. [Experiment Summaries](#experiment-summaries)
8. [Point-In-Time Correctness](#point-in-time-correctness)
9. [Result Reconciliation](#result-reconciliation)
10. [Universe](#universe)

---

## What This Project Is

A research pipeline that evaluates quantitative signals for constructing a
long-only equity portfolio from a fixed 30-stock universe.  The experiments
are purely research artefacts: they document which signals were studied and
why certain factors were selected.  They do **not** constitute a live trading
system or investment advice.

---

## Repository Structure

```
Portfolio-Playoff/
├── config/
│   ├── universe.py           # SINGLE SOURCE OF TRUTH: COMPETITION_UNIVERSE, SECTOR_MAP
│   ├── sectors.py            # Re-exports from universe.py
│   ├── research_config.py    # Date ranges, rebalance settings, statistical parameters
│   └── locked_results.yaml   # Expected IC values for automated reconciliation
│
├── src/
│   ├── data/
│   │   ├── prices.py         # Price ingestion, forward return computation
│   │   ├── fundamentals.py   # TTM aggregation
│   │   └── pit_processor.py  # Point-in-time fundamental alignment
│   ├── features/
│   │   ├── common.py         # zscore_cross_section, winsorize_cross_section
│   │   ├── momentum.py       # All momentum features
│   │   ├── earnings.py       # Earnings momentum and acceleration features
│   │   └── quality.py        # ROE, ROA, margins, leverage — with sector masking
│   ├── research/
│   │   ├── statistics.py     # IC, HAC t-stat, residualization, winsorization
│   │   ├── metrics.py        # IC series summarization
│   │   ├── portfolio_sorts.py# Quintile sorts, 2×2 double sorts
│   │   └── robustness.py     # Leave-one-stock-out analysis
│   └── utils/
│       ├── io.py             # load_csv / save_csv (project-root relative)
│       ├── dates.py          # generate_rebalance_schedule
│       └── logging.py        # Centralised logger
│
├── experiments/
│   ├── exp001_momentum/run.py
│   ├── exp002_earnings_momentum/run.py
│   ├── exp003_complementarity/run.py
│   └── exp004A_quality/run.py
│
├── data/
│   ├── raw/                  # Required input files (NOT committed — see below)
│   │   ├── prices_30.csv
│   │   ├── fundamentals_30.csv
│   │   └── filings_dates.csv
│   └── DATA_DICTIONARY.md
│
├── outputs/                  # Auto-created by experiments
│   ├── tables/
│   └── reports/
│
├── tests/                    # pytest test suite
├── research/
│   └── reproducibility_manifest.yaml
│
├── run_all.py                # Run all experiments sequentially
├── reconcile.py              # Compare outputs against locked results
├── validate_data.py          # Check data files before running experiments
└── requirements.txt
```

---

## Environment Setup

### 1. Clone

```bash
git clone https://github.com/Sirius-5107/Portfolio-Playoff.git
cd Portfolio-Playoff
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or with `pyproject.toml`:

```bash
pip install -e .
```

### 3. Verify imports

```bash
python -c "from config.universe import COMPETITION_UNIVERSE; print(len(COMPETITION_UNIVERSE), 'stocks')"
# Expected: 30 stocks
```

---

## Required Data

Three files must be placed under `data/raw/` before running any experiment.

| File | Description | Required Columns |
|------|-------------|------------------|
| `prices_30.csv` | Daily adjusted close prices (2010–present) | `date, ticker, adj_close` |
| `fundamentals_30.csv` | Quarterly financials per ticker | `ticker, quarter_end, eps, net_income, revenue, ebit, ocf, capex, equity, total_assets, interest_expense` |
| `filings_dates.csv` | Actual BSE/NSE filing dates | `ticker, quarter_end, filing_date` |

**These files are not committed to the repository** because they contain
data sourced from third-party providers.  Obtain them from:

- `prices_30.csv` — Yahoo Finance (`yfinance`), NSE historical data
- `fundamentals_30.csv` — Screener.in, Bloomberg, or company filings
- `filings_dates.csv` — BSE filing portal, Screener.in

See `data/DATA_DICTIONARY.md` for full column specifications.

---

## Validating the Environment

Before running experiments, validate:

```bash
# 1. Check that required data files exist and are well-formed
python validate_data.py

# 2. Run the test suite (no data files needed)
python -m pytest tests/ -v

# 3. Verify the universe
python -c "from config.universe import validate_universe, COMPETITION_UNIVERSE, SECTOR_MAP; validate_universe(COMPETITION_UNIVERSE, SECTOR_MAP); print('Universe OK')"
```

---

## Running Experiments

### Run a single experiment

```bash
# Experiment 001 — Historical Momentum
python experiments/exp001_momentum/run.py

# Experiment 002 — Historical Earnings Momentum
python experiments/exp002_earnings_momentum/run.py

# Experiment 003 — Momentum × Earnings Complementarity
python experiments/exp003_complementarity/run.py

# Experiment 004A — Quality Incremental Signal Audit
python experiments/exp004A_quality/run.py
```

### Run all experiments and reconcile

```bash
python run_all.py
```

Outputs are written to `outputs/tables/` and `outputs/reports/`.

### Reconcile outputs against locked results

```bash
python reconcile.py
```

---

## Experiment Summaries

### Experiment 001 — Historical Momentum

Evaluates seven momentum features over the 2015–2023 research window.

Key results (research IC, 20D target):

| Factor | Research IC |
|--------|-------------|
| `mom_5d` | ≈ −0.012 (DISCARD) |
| `mom_20d` | ≈ −0.003 (DISCARD) |
| `mom_60d` | ≈ +0.054 (KEEP) |
| `mom_252d` | ≈ +0.038 (MODIFY → 12-1) |
| `mom_12_1` | ≈ +0.050 (KEEP) |
| `sector_relative_mom_60d` | ≈ +0.061 (KEEP — primary signal) |
| `sector_relative_mom_252d` | ≈ +0.042 |

### Experiment 002 — Historical Earnings Momentum

Evaluates eight earnings-derived features with strict PIT treatment.
Primary signal: `eps_growth_acceleration` (IC ≈ +0.058).

### Experiment 003 — Momentum × Earnings Complementarity

Tests the composite:

```
BaseScore = Z(sector_relative_mom_60d) + Z(eps_growth_acceleration)
```

Equal-weighted (1:1), not OLS-estimated.
BaseScore IC ≈ +0.0915, HAC t ≈ 3.6.

### Experiment 004A — Quality Incremental Signal Audit

Tests whether quality factors (ROE, ROA, FCF Margin, etc.) add incremental
predictive power to the BaseScore via paired Delta-IC:

```
Delta_IC_t = IC_{Base+ROE, t} - IC_{Base, t}
```

Mean ΔIC ≈ +0.0068, HAC t ≈ 2.3.  ROE shows a modest but statistically
present incremental contribution.

---

## Point-In-Time Correctness

Every fundamental observation follows a strict timeline:

```
quarter_end
    → actual filing date (from filings_dates.csv)
        → first trading day STRICTLY AFTER filing date
            → feature becomes eligible (usable_date)
```

If an actual filing date is unavailable, the fallback is
`quarter_end + 45 calendar days`, and the row is flagged with
`is_fallback_date = True`.

The `merge_pit_features_to_grid()` function uses `pd.merge_asof` with
`direction="backward"` to guarantee that only information available on or
before each rebalance date enters the feature vector.

Lookahead tests are in `tests/test_pit.py`.

---

## Result Reconciliation

After running all experiments, `reconcile.py` compares each key metric
against the expected values in `config/locked_results.yaml`.

Results are reported as PASS or INVESTIGATE (not FAIL — see note below).

> **Important**: if a metric is outside its tolerance window, investigate
> the root cause (data version, date range change, methodology drift).
> Do **not** manipulate code until the numbers match.

---

## Universe

The competition universe is defined once in `config/universe.py` as
`COMPETITION_UNIVERSE`.  It contains exactly 30 stocks, validated on import.

The alias `UNIVERSE` is available for backward compatibility but
`COMPETITION_UNIVERSE` is the canonical name.

Banned tickers (must not appear): `LTIM`, `GAIL`, `ADANIPORTS`.

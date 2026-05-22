# C++17 Limit Order Book
> Low latency C++17 trading core integrated with a Python research and execution stack

[![C++ Build](https://img.shields.io/badge/C%2B%2B-Build%20%26%20Tests-blue?logo=github-actions)](https://github.com/yajatmishra/limit-order-book/.github/workflows/build_cpp.yml)
[![Python Tests](https://img.shields.io/badge/Python-416%20tests-green?logo=pytest)](https://github.com/yajatmishra/limit-order-book/.github/workflows/test_python.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project is a quantitative trading research platform built around a low-latency C++17 market data and execution core with a Python research layer for modelling, simulation, and analysis. The system ingests raw NASDAQ ITCH 5.0 market data, reconstructs the limit order book in real time, generates microstructure-based signals, simulates execution, and evaluates strategy performance through an event-driven backtesting pipeline.

The architecture is split into two layers:

**C++17 core** : limit order book reconstruction, binary ITCH parsing, lock-free messaging, shared-memory snapshot publishing, event dispatch, and order/fill simulation

**Python research stack** : signal research, statistical modelling, validation, execution logic, risk analysis, and dashboarding

The two layers communicate through a shared-memory seqlock interface, allowing Python models to consume real-time order book snapshots with minimal overhead.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  NASDAQ FTP  │  Yahoo Finance v8  │  PCAP replay                │
│  ITCH 5.0    │  daily OHLCV       │  (offline)                  │
└──────┬───────┴────────┬───────────┴──────┬──────────────────────┘
       │                │                  │
       ▼                ▼                  ▼
┌──────────────┐  ┌───────────┐   ┌───────────────────┐
│ItchDownloader│  │DailyDown- │   │  itch_parser.cpp  │  C++17
│download_itch │  │loader     │   │  pcap_replayer.cpp│  core
│  .py         │  │.py        │   │  feed_handler.cpp │
└──────┬───────┘  └─────┬─────┘   └────────┬──────────┘
       │                │                  │ LOBSnapshot
       ▼                ▼                  ▼  (seqlock SHM)
  DataCatalog      Parquet store   ┌──────────────────────┐
  (SQLite)         data/           │  limit_order_book.cpp│
                                   │  price_level.cpp     │
                                   │  order.hpp           │
                                   └────────┬─────────────┘
                                            │
                              ┌─────────────▼──────────────┐
                              │  shm_writer.cpp (seqlock)  │
                              │  ring_buffer.hpp (SPSC)    │
                              │  event_bus.cpp (typed)     │
                              │  order_router.cpp          │
                              │  fill_simulator.cpp        │
                              └─────────────┬──────────────┘
                                            │  ShmReader
                                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    Python research stack                      │
│                                                               │
│  microstructure/   signals/        validation/                │
│  ├ ofi.py          ├ feature_pipeline.py  ├ purged_cv.py      │
│  ├ pin_model.py    ├ mean_reversion.py    ├ walk_forward.py   │
│  ├ spread_decomp   ├ momentum.py          ├ sharpe_deflator   │
│  ├ queue_model     ├ cointegration.py     ├ regime_tester     │
│  └ avellaneda_s    ├ kalman_pairs.py      └ tca.py            │
│                    ├ hmm_regime.py                            │
│  execution/        ├ garch_x.py          risk/                │
│  ├ market_impact   └ signal_combiner     ├ kelly_sizer.py     │
│  ├ vwap.py                               ├ pnl_reporter.py    │
│  ├ twap.py         backtester/           ├ position_tracker   │
│  └ participation   ├ engine.py           └ circuit_breakers   │
│                    ├ portfolio.py                             │
│  dashboard/        └ tearsheet.py        data/                │
│  ├ app.py                                ├ download_itch.py   │
│  ├ lob_depth_chart ← live LOB depth      ├ download_daily     │
│  ├ ofi_panel       ← OFI + price impact  └ data_catalog.py    │
│  ├ pnl_panel       ← equity/drawdown                          │
│  └ regime_panel    ← HMM state overlay                        │
└───────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
limit-order-book/
├── core/                          # C++17 latency-critical engine
│   ├── lob/
│   │   ├── order.hpp              # Order POD, Side enum
│   │   ├── price_level.hpp/.cpp   # Price level with O(1) qty aggregation
│   │   └── limit_order_book.hpp/.cpp  # Full dual-sided LOB (add/cancel/exec)
│   ├── feed_handler/
│   │   ├── itch_parser.hpp/.cpp   # NASDAQ ITCH 5.0 binary parser
│   │   ├── pcap_replayer.hpp/.cpp # PCAP file replay at configurable speed
│   │   └── feed_handler.hpp/.cpp  # Wires parser → LOB → SHM
│   ├── shared_memory/
│   │   └── shm_writer.hpp/.cpp    # Seqlock LOBSnapshot writer
│   ├── event_bus/
│   │   ├── event_types.hpp        # Typed event variants (Fill, Quote, …)
│   │   └── event_bus.hpp/.cpp     # TypedEventBus with compile-time dispatch
│   └── execution/
│       ├── order_router.hpp/.cpp  # Routing + simulated exchange gateway
│       └── fill_simulator.hpp/.cpp # Probabilistic fill model
│
├── python/                        # Python research stack
│   ├── microstructure/            
│   ├── signals/                   
│   ├── validation/                
│   ├── execution/                 
│   ├── risk/                      
│   ├── backtester/                
│   └── dashboard/                 
│
├── data/                          # Data utilities
│   ├── download_itch.py           # NASDAQ ITCH FTP downloader
│   ├── download_daily.py          # Yahoo Finance v8 OHLCV downloader
│   └── data_catalog.py            # SQLite-backed data catalog
│
├── tests/
│   ├── cpp/                       # 4 Catch2 test files
│   │   ├── test_lob.cpp
│   │   ├── test_itch_parser.cpp
│   │   ├── test_ring_buffer.cpp
│   │   └── test_event_bus.cpp
│   └── python/                    
│       ├── test_ofi.py            
│       ├── test_avellaneda_stoikov.py  
│       ├── test_kalman_pairs.py   
│       ├── test_purged_cv.py     
│       ├── test_walk_forward.py   
│       ├── test_risk.py           
│       ├── test_execution.py      
│       ├── test_backtester.py     
│       └── test_signals.py        
│
├── CMakeLists.txt                 # CMake 3.20+, FetchContent Catch2 v3.5.4
├── pyproject.toml                 # PEP 517/518, pytest config, ruff config
├── requirements.txt               # Pinned runtime deps
└── .github/workflows/
    ├── build_cpp.yml              # C++ matrix CI
    └── test_python.yml            # Python matrix CI
```

---

## C++ Core

### Limit Order Book (`core/lob/`)

Full dual-sided price-level LOB. Orders are stored in `std::map<price, PriceLevel>` (asks ascending, bids descending via `std::greater<>`). Each `PriceLevel` maintains a `std::list<Order>` for FIFO priority and an O(1) total-quantity counter. Supports:

- `add_order(order)` — O(log P) where P = distinct price levels
- `cancel_order(order_id)` — O(1) via `std::unordered_map<id, iterator>` cancel map
- `execute_order(order_id, qty)` — partial and full execution, O(1)
- `best_bid()` / `best_ask()` — O(1) via `std::map::begin()`
- `depth(side, levels)` — returns vector of `(price, qty)` pairs

### ITCH 5.0 Parser (`core/feed_handler/itch_parser.cpp`)

Stateless framing loop over raw binary: `2-byte BE length | 1-byte type | body`. Body header layout: `stock_locate(2) + tracking(2) + ts_hi(2) + ts_lo(4)` = 10 bytes before message-specific fields (base offset `_B = 10`). Handles all ITCH 5.0 message types relevant to LOB reconstruction:

| Type | Message | Fields parsed |
|------|---------|---------------|
| `A` | Add Order (no MPID) | order_ref, side, shares, stock, price |
| `F` | Add Order with MPID | same + attribution |
| `E` | Order Executed | order_ref, executed_shares, match_number |
| `C` | Order Executed with Price | + execution_price |
| `X` | Order Cancel | order_ref, cancelled_shares |
| `D` | Order Delete | order_ref |
| `U` | Order Replace | orig_ref, new_ref, shares, price |
| `P` | Trade (non-cross) | order_ref, side, shares, stock, price, match |
| `Q` | Cross Trade | shares, stock, cross_price, match, cross_type |

### Lock-Free SPSC Ring Buffer (`core/feed_handler/`)

Cache-line-aligned (64-byte) SPSC queue using `std::atomic<size_t>` head/tail with acquire/release memory ordering. Zero heap allocation after construction. Template over element type `T` and capacity `N` (must be power of 2). `try_push` / `try_pop` return `bool` — the producer never blocks the consumer.

### Seqlock SHM Writer (`core/shared_memory/shm_writer.cpp`)

Publishes `LOBSnapshot` structs to shared memory using a seqlock: writer increments sequence to odd before writing, back to even after. Readers spin until they observe a stable even sequence number with no change across their read window. This gives wait-free reads at the cost of potential retry on the reader side. Python `ShmReader` uses `mmap` + `ctypes` to consume directly without serialisation.

### TypedEventBus (`core/event_bus/`)

Compile-time type-safe publish/subscribe bus using `std::variant<Fill, Quote, Cancel, …>` and `std::unordered_map<type_index, vector<callback>>`. Subscribers register with `subscribe<EventType>(callback)`. `publish(event)` dispatches by `std::type_index` — no virtual dispatch, no heap allocation per event.

### OrderRouter + FillSimulator (`core/execution/`)

`OrderRouter` maintains a symbol→venue routing table and a pending order book. `FillSimulator` models fill probability as a function of queue position, spread, and time-in-force. Simulated exchange acknowledges orders synchronously via the event bus so the backtester can use the same code path as live routing.

---

## Python Research Stack

### Microstructure (`python/microstructure/`)

**Order Flow Imbalance** (`ofi.py`) — Cont, Kukanov & Stoikov (2014) multi-level OFI with exponential decay weights across bid/ask levels. Price impact regression verified on 5,000 synthetic bars: **β = 6 × 10⁻⁶, R² = 0.4397**.

**PIN Model** (`pin_model.py`) — Easley, Kiefer, O'Hara & Paperman (1996). EM estimation of α (informed-event probability), δ (bad-news probability), ε (uninformed arrival rate), μ (informed arrival rate). Returns `PINResult` with `.pin`, `.alpha`, `.delta`, `.epsilon`, `.mu`.

**Spread Decomposition** (`spread_decomp.py`) — Glosten-Milgrom adverse selection + Roll (1984) serial covariance estimator. Decomposes effective spread into adverse-selection component (λ), order-processing cost, and inventory cost.

**Queue Model** (`queue_model.py`) — Cont & de Larrard (2013) queue imbalance model. Estimates fill probability at the best bid/ask as a function of queue length ratio Q_b / Q_a. Returns `QueueResult` with fill probabilities at ±k ticks.

**Avellaneda-Stoikov Market Making** (`avellaneda_stoikov.py`) — Stochastic-control optimal quoting. At mid=100, γ=0.1, σ=0.2, k=1.5, t_remaining=0.5: **bid = 99.3536, ask = 100.6464, spread = 1.2928**. Simulated over 252 steps (seed=42): **final PnL = 72.49, fill count = 112**.

### Signals (`python/signals/`)

**Feature Pipeline** (`feature_pipeline.py`) — Computes the full feature vector from `LOBSnapshot` data: OFI (multi-level), spread (bps), mid-price return, bid/ask depth ratio, VWAP deviation, rolling volatility (ewm), autocorrelation lag-1. All features z-scored in a rolling window. Returns `pd.DataFrame`.

**Mean Reversion** (`mean_reversion.py`) — Ornstein-Uhlenbeck half-life estimator via OLS on `Δx_t = a + b·x_{t-1} + ε_t`. Half-life = `-ln(2)/ln(1+b)`. Z-score signal with configurable entry/exit thresholds.

**Momentum** (`momentum.py`) — Time-series momentum (Moskowitz, Ooi & Pedersen 2012) with configurable lookback. Includes cross-sectional rank normalisation and turnover filtering. Returns signed position targets in `[-1, 1]`.

**Cointegration** (`cointegration.py`) — Engle-Granger two-step with ADF residual test + Johansen trace/eigenvalue tests. Verified: synthetic pair (β=1.5, ε~N(0, 0.5), T=2000): **hedge_ratio = 1.4923, ADF = −16.191, p = 4.17 × 10⁻²⁹, cointegrated at 1% (critical value −3.96)**.

**Kalman Pairs** (`kalman_pairs.py`) — Time-varying hedge ratio via linear Kalman filter. State = [β, α]ᵀ, observation = y_t − β·x_t − α. Q and R tuned by log-likelihood maximisation. Rolling covariance tracks regime changes in the spread.

**HMM Regime Detection** (`hmm_regime.py`) — K-state Gaussian HMM, Baum-Welch EM in log-space. Verified K=2 on 5,000 return bars (seed=42): **converged in 28 iterations, LL = 32,097.7. Low-vol state: σ = 0.0992%/bar, P(low→high) = 0.0287, expected duration ≈ 35 bars. High-vol state: σ = 0.0303%/bar, P(high→low) = 0.0067, expected duration ≈ 149 bars**.

**GARCH-X** (`garch_x.py`) — GARCH(1,1) + optional exogenous regressor X_t (e.g. OFI, news indicator). Scipy L-BFGS-B with 5 random restarts. Verified on 1,000 synthetic bars (true ω=1e-6, α=0.05, β=0.93): **recovered ω = 5.54 × 10⁻⁷, α = 0.0433, β = 0.9446, converged, log-likelihood = 3,621.96**.

**Signal Combiner** (`signal_combiner.py`) — IC-weighted ensemble of arbitrary `Signal` objects. Supports rolling IC (information coefficient) estimation, decay weighting, and signal orthogonalisation via Gram-Schmidt.

### Validation (`python/validation/`)

**Purged K-Fold CV** (`purged_cv.py`) — De Prado (2018) time-series cross-validation that purges training samples overlapping with the test window and applies an embargo period to prevent leakage. Verified: 5 splits on 1,000 observations with embargo_pct=1%: **fold 0 → train=790, test=200, n_purged=0, n_embargoed=10**.

**Walk-Forward** (`walk_forward.py`) — Expanding-window and rolling-window walk-forward with configurable train/test ratio. Stores per-fold metrics for detecting strategy degradation over time.

**Sharpe Deflator** (`sharpe_deflator.py`) — Bailey & López de Prado (2014) Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR). PSR = P(SR* > SR_benchmark | SR_hat, T, skew, kurt). Minimum Track Record Length (MTRL) = number of observations needed for PSR ≥ 0.95. Verified on 252 obs, SR_hat≈0.78: **PSR = 0.7843, DSR = 0.7843, MTRL = 1,098 obs**.

**Regime Tester** (`regime_tester.py`) — Tests whether a signal has different characteristics across HMM-identified regimes. Permutation test for regime-conditional Sharpe differences.

**TCA** (`tca.py`) — Transaction cost analysis: arrival-price slippage, implementation shortfall, VWAP slippage, market impact decomposition.

### Execution (`python/execution/`)

**Almgren-Chriss** (`market_impact.py`) — Discrete-time optimal liquidation. Minimises `E[cost] + λ·Var[cost]` under linear market impact. Verified: Q=10,000 shares, T=1 day, N=10 intervals, σ=1.5%, η=2.5×10⁻⁷, γ=2.5×10⁻⁸, ADV=1M, λ=1×10⁻⁶: **κ = 30.075, E[cost] = 2.265 bps, Var[cost] = 2,255.5**. Front-loaded: first interval executes 9,506 of 10,000 shares.

**VWAP** (`vwap.py`) — Bucket-schedule VWAP with flat, U-shaped, and custom participation profiles. Verified: 10,000 shares over 390 1-min buckets, flat profile, spread=5 bps: **implementation shortfall = 2.50 bps**.

**TWAP** (`twap.py`) — Time-weighted average price execution with configurable slice count and randomised order sizing (±10% jitter) to reduce market impact predictability.

**Participation Rate** (`participation_rate.py`) — POV (percentage of volume) strategy. Tracks realised participation rate against target and adjusts slice sizes dynamically.

### Risk (`python/risk/`)

**Kelly Sizer** (`kelly_sizer.py`) — Full, half, and fractional Kelly position sizing. Three entry points: `size_binary(p_win, payoff)`, `size_from_sharpe(sharpe_ratio, sigma)`, `size_from_moments(mu, sigma)`. Verified: p=0.55, payoff=2×: **full Kelly = 0.3250, half Kelly = 0.1625**.

**PnL Reporter** (`pnl_reporter.py`) — Computes Sharpe, Sortino, Calmar, max drawdown, VaR (historical and parametric), CVaR (ES), skewness, kurtosis, monthly returns heatmap. Used by `Tearsheet` for final attribution.

**Position Tracker** (`position_tracker.py`) — Real-time mark-to-market with per-symbol and aggregate P&L. Tracks gross/net exposure, turnover, and fill notional.

**Circuit Breakers** (`circuit_breakers.py`) — Hard stop-loss and drawdown-from-peak limits. `CircuitBreaker.check(equity, peak)` returns `TripResult` with `tripped`, `reason`, and `recommended_action`.

### Backtester (`python/backtester/`)

**Engine** (`engine.py`) — Event-driven backtester consuming `LOBSnapshot` objects. Supports two feed modes: `ShmReader` (live C++ SHM feed) and `ItchReplayer` (offline binary file replay). Per-tick loop: snapshot → strategy.on_snapshot() → OrderRouter → FillSimulator → Portfolio.update(). Returns `EngineResult`.

**Portfolio** (`portfolio.py`) — Tracks cash and positions. Equity formula: `equity = cash + Σ(position × current_mark_price)`. Cash decrements on buy fills, increments on sell fills — already reflects all notional. Exposes `mark_to_market(prices)`, `pnl_series`, `returns`.

**Tearsheet** (`tearsheet.py`) — Wraps `PnLReporter` + `SharpeDeflator`. `Tearsheet.compute(result)` returns an object with `.sharpe`, `.sortino`, `.calmar`, `.max_drawdown`, `.var_95`, `.cvar_95`, `.psr`, `.dsr`, `.mtrl`. Dashboard replay (2,000 snaps, seed=42, OFIMomentum strategy): **1,379 fills, P&L = −$58.60, Sharpe = −14.17, Sortino = −1.25, MaxDD = 0.0614%**.

### Dashboard (`python/dashboard/`)

Five-module Plotly Dash application. Dark theme (`#0f172a` background, `#fbbf24` amber accents). Generates a synthetic replay session at startup and renders four panels:

**LOB Depth Chart** (`lob_depth_chart.py`) — Horizontal bar chart with bid (green) and ask (red) mountains. Cumulative depth fill shown as semi-transparent area. Top 5 levels with price labels.

**OFI Panel** (`ofi_panel.py`) — Two subplots: (1) rolling OFI time series with ±1σ bands, threshold lines; (2) ΔMid vs OFI scatter with OLS regression line, β and R² annotated (β = 6×10⁻⁶, R² = 0.44).

**PnL Panel** (`pnl_panel.py`) — Three subplots: equity curve, return distribution, underwater drawdown chart. Metrics annotation box (monospace, amber border) displays Sharpe, Sortino, MaxDD, VaR, PSR.

**Regime Panel** (`regime_panel.py`) — Mid-price coloured by Viterbi HMM state (K=2, green=low-vol, red=high-vol). Stacked γ_t(k) probability area chart below. State statistics annotated in legend.

**App** (`app.py`) — 2×2 CSS grid layout. Three Dash callbacks: LOB tick animation at 500 ms interval, play/pause toggle button, manual snapshot slider. Session runs completely offline — no network calls.

---

## Data Utilities

### ITCH Downloader (`data/download_itch.py`)

Downloads NASDAQ ITCH 5.0 files from the EMI FTP server (`emi.nasdaq.com`). File naming convention: ISO date `2024-01-15` maps to FTP filename `01152024.NASDAQ_ITCH50.gz`. Optional SHA-256 verification and gzip decompression on download.

```python
from data.download_itch import ItchDownloader
dl = ItchDownloader(dest_dir="~/data/itch")
files = dl.download_range("2024-01-02", "2024-01-31", skip_weekends=True)
```

### Daily Downloader (`data/download_daily.py`)

Downloads adjusted OHLCV data from Yahoo Finance v8 API with automatic fallback between `query1` and `query2` endpoints. Saves as Parquet. Handles splits/dividends adjustment transparently.

```python
from data.download_daily import DailyDownloader
dl = DailyDownloader(dest_dir="~/data/daily")
dl.download_many(["AAPL","MSFT","SPY"], start="2020-01-01", end="2024-12-31")
```

### Data Catalog (`data/data_catalog.py`)

SQLite-backed catalog for tracking downloaded files. Indexed on `(data_type, date)` and `symbol`. Supports SHA-256 integrity verification, tag-based search, and storage space reporting.

```python
from data.data_catalog import DataCatalog, CatalogEntry
cat = DataCatalog("~/.limit_order_book/catalog.db")
entry = CatalogEntry(data_type="itch", date="2024-01-15",
                     filename="01152024.NASDAQ_ITCH50.gz",
                     path="/data/itch/...", download_ts=CatalogEntry.now_ts(),
                     size_bytes=4_800_000_000)
cat.add(entry)
results = cat.find(data_type="itch", date="2024-01-15")
```

---

## Installation

### Prerequisites

- CMake ≥ 3.20
- GCC ≥ 12 or Clang ≥ 14 (C++17 required)
- Python ≥ 3.10
- pip ≥ 23

### C++ Build

```bash
git clone https://github.com/yajatmishra/limit-order-book.git
cd limit-order-book

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSIGMA_SANITIZE=OFF

cmake --build build --parallel $(nproc)

# Run C++ tests
cd build && ctest --output-on-failure --parallel $(nproc)
```

Debug build with Address Sanitizer + Undefined Behaviour Sanitizer:

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSIGMA_SANITIZE=ON

cmake --build build-debug --parallel $(nproc)
cd build-debug && ctest --output-on-failure
```

### Python Install

```bash
# Runtime only
pip install -r requirements.txt

# Full development install (adds pytest, ruff, mypy, pre-commit)
pip install -e ".[dev]"

# Research extras (adds jupyter, matplotlib, scikit-learn)
pip install -e ".[dev,research]"
```

### Running the Test Suite

```bash
# All 416 Python tests
PYTHONPATH=python pytest tests/python/ -v

# With coverage report
PYTHONPATH=python pytest tests/python/ \
  --cov=python --cov-report=html:htmlcov \
  --cov-report=term-missing

# Specific module
PYTHONPATH=python pytest tests/python/test_backtester.py -v
```

### Running the Dashboard

```bash
cd limit-order-book
PYTHONPATH=python python python/dashboard/app.py
# Open http://localhost:8050
```

The dashboard generates a synthetic replay session on startup (2,000 LOB snapshots, ~0.2 s) and requires no network access.

---

## CLI Entry Points

After `pip install -e .`:

```bash
# Download ITCH data for a date range
sigma-download-itch --dest ~/data/itch --start 2024-01-02 --end 2024-01-31

# Download daily OHLCV
sigma-download-daily --dest ~/data/daily --symbols AAPL MSFT SPY --start 2020-01-01

# Query the data catalog
sigma-catalog status
sigma-catalog find --data-type itch --date 2024-01-15
sigma-catalog verify --recompute-sha256
```

---

## CI / CD

### C++ Matrix (`.github/workflows/build_cpp.yml`)

6 job combinations: Ubuntu 22.04 × {GCC-12, Clang-17} × {Release, Debug+ASan+UBSan} and macOS 14 × AppleClang × {Release, Debug+ASan}. Each job:

1. Installs the compiler (Ubuntu only; macOS uses Xcode CLT)
2. Restores the CMake + Catch2 FetchContent cache (keyed on `CMakeLists.txt` hash)
3. Configures with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`
4. Builds with `--parallel $(nproc)`
5. Runs CTest with `--output-junit ctest-results.xml`
6. Uploads XML results as artifacts (14-day retention)
7. Uploads test binaries on failure (3-day retention)

### Python Matrix (`.github/workflows/test_python.yml`)

6 job combinations: {Ubuntu 22.04, macOS 14} × {Python 3.10, 3.11, 3.12}. Each job:

1. Sets up Python with pip cache keyed on `requirements.txt` + `pyproject.toml`
2. Installs runtime requirements + dev extras (pytest ≥ 8, pytest-cov, pytest-xdist, ruff)
3. Runs Ruff lint/format check (non-blocking on PRs, blocking on main)
4. Runs pytest with `--numprocesses=auto --dist=worksteal` (parallel)
5. Uploads coverage XML to Codecov (Ubuntu/3.12 only)
6. Uploads HTML coverage report as artifact (7-day retention)
7. Runs three offline smoke tests: DataCatalog, DailyDownloader, ItchDownloader

Concurrency groups cancel in-progress runs on the same branch/workflow to avoid redundant CI spend.

---

## Performance Notes

All numbers below are produced by running the actual code in this repository — see [METHODOLOGY.md](METHODOLOGY.md) for the scripts used.

| Component | Metric | Value |
|-----------|--------|-------|
| OFI → ΔMid regression | R² | 0.4397 |
| OFI → ΔMid regression | β | 6 × 10⁻⁶ |
| HMM K=2, low-vol state | σ/bar | 0.0992% |
| HMM K=2, high-vol state | σ/bar | 0.0303% |
| HMM K=2 | Baum-Welch iterations | 28 |
| GARCH-X (α_true=0.05) | α recovered | 0.0433 |
| GARCH-X (β_true=0.93) | β recovered | 0.9446 |
| Almgren-Chriss (Q=10k, T=1d) | E[cost] | 2.265 bps |
| Almgren-Chriss | κ (decay rate) | 30.075 |
| VWAP (10k shares, 390 buckets) | Implementation shortfall | 2.50 bps |
| Kelly (p=0.55, payoff=2×) | Full Kelly fraction | 0.3250 |
| Engle-Granger cointegration | Hedge ratio | 1.4923 |
| Engle-Granger cointegration | ADF stat | −16.191 |
| Purged CV (5 folds, N=1000, embargo=1%) | Train / test per fold | 790 / 200 |
| AS market maker (q=0, t=0.5) | Bid-ask spread | 1.2928 |
| Sharpe Deflator (SR≈0.78, T=252) | PSR | 0.7843 |
| Sharpe Deflator | MTRL | 1,098 obs |
| Dashboard replay (2000 snaps) | Fill count | 1,379 |
| Dashboard replay | Max drawdown | 0.0614% |

---

## References

- Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.
- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Baum, L. E. et al. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. *Annals of Mathematical Statistics*, 41(1), 164–171.
- Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47–88.
- Cont, R. & de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM Journal on Financial Mathematics*, 4(1), 1–25.
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
- Easley, D., Kiefer, N. M., O'Hara, M. & Paperman, J. B. (1996). Liquidity, information, and infrequently traded stocks. *Journal of Finance*, 51(4), 1405–1436.
- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251–276.
- Glosten, L. R. & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market. *Journal of Financial Economics*, 14(1), 71–100.
- Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.
- Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
- NASDAQ (2019). *ITCH 5.0 Protocol Specification*. nasdaq.com/market-activity.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance*, 39(4), 1127–1139.

---

## License

MIT — see [LICENSE](LICENSE).

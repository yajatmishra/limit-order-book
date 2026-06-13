<div align="center">

# Limit Order Book

**A low-latency C++17 market-data and execution core, paired with a Python research stack and a live Plotly Dash dashboard.**

[![C++ Build & Tests](https://github.com/yajatmishra/limit-order-book/actions/workflows/build_cpp.yml/badge.svg)](https://github.com/yajatmishra/limit-order-book/actions/workflows/build_cpp.yml)
[![Python Tests](https://github.com/yajatmishra/limit-order-book/actions/workflows/test_python.yml/badge.svg)](https://github.com/yajatmishra/limit-order-book/actions/workflows/test_python.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](CMakeLists.txt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-133%20C%2B%2B%20%2B%20416%20Py-success)](tests/)

[Live Demo](#live-demo) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Deployment](#deployment) · [Methodology](METHODOLOGY.md)

</div>

---

## Overview

This is a quantitative trading research platform. It is built around a low-latency C++17 market-data and execution core, and it is integrated with a Python research layer for signal generation, strategy validation, execution optimisation, and risk management.

The system ingests raw NASDAQ ITCH 5.0 market data and reconstructs the full limit order book in real time. It then generates microstructure-based signals, simulates execution, and evaluates performance through an event-driven backtesting pipeline. The two layers communicate through a shared-memory seqlock interface, which lets the Python models consume real-time order-book snapshots with minimal overhead.

The C++17 core handles limit order book reconstruction, binary ITCH 5.0 parsing, lock-free single-producer single-consumer messaging, seqlock shared-memory snapshot publishing, typed event dispatch, and order and fill simulation.

The Python research stack provides market-microstructure models, statistical signal research, walk-forward validation, execution algorithms, risk sizing, and the dashboard.

---

## Live Demo

The dashboard is an interactive session-replay tool. It shows five linked panels that render a synthetic trading day of 2,000 ITCH snapshots with a fixed seed of 42. It runs entirely offline, so it needs no live market data, and it has a dark theme, a responsive layout, and scrub and play controls.

You can deploy your own copy in a few minutes. See the [Deployment](#deployment) section below. Once it is running, it is served at `https://limit-order-book-dashboard.onrender.com`.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/yajatmishra/limit-order-book)

<table>
  <tr>
    <td width="50%"><img src="docs/images/lob_depth.png" alt="LOB Depth Chart" /></td>
    <td width="50%"><img src="docs/images/pnl_panel.png" alt="P&amp;L Panel" /></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/images/ofi_panel.png" alt="OFI Panel" /></td>
    <td width="50%"><img src="docs/images/regime_panel.png" alt="Regime Panel" /></td>
  </tr>
</table>

| Panel | What it shows |
|---|---|
| **LOB Depth** | A symmetric mountain chart of resting bid (green) and ask (red) quantity across five price levels, with a cumulative depth overlay and a mid-price rule. It scrubs across snapshots. |
| **P&L** | The equity curve, per-bar returns, the underwater drawdown chart, and a metrics box with Sharpe, Sortino, Calmar, MaxDD, PSR, and DSR. |
| **OFI** | Rolling order-flow imbalance from Cont, Kukanov and Stoikov (2014) with plus and minus one standard deviation bands, plus a scatter of change in mid price against OFI with an OLS fit. |
| **Regime** | The mid price coloured by a two-state Gaussian HMM Viterbi path, with a stacked posterior-probability area chart below it. |

---

## Quick Start

### Run the dashboard locally

```bash
git clone https://github.com/yajatmishra/limit-order-book.git
cd limit-order-book
pip install -r requirements.txt

# Development server (Dash)
PYTHONPATH=python python python/dashboard/app.py      # http://localhost:8050

# Production server (the same command Render runs)
gunicorn wsgi:server --preload --bind 0.0.0.0:8050
```

### Run with Docker

```bash
docker build -t lob-dashboard .
docker run --rm -p 8050:8050 lob-dashboard            # http://localhost:8050
```

---

## Deployment

The dashboard ships ready for production. A [`wsgi.py`](wsgi.py) entry point exposes the Flask `server`, which is served by gunicorn. Deploy configurations are included for the common targets.

### Render (recommended)

This repository contains a [`render.yaml`](render.yaml) Blueprint. Either click the Deploy to Render button above, or follow these steps.

1. Push the repository to GitHub.
2. On [render.com](https://render.com), choose New, then Blueprint, and connect the repository.
3. Render reads `render.yaml`, installs `requirements.txt`, and starts the service with this command:
   ```
   gunicorn wsgi:server --workers 1 --threads 8 --timeout 120 --preload --bind 0.0.0.0:$PORT
   ```
4. The first build takes a few minutes because the numpy, scipy, and pandas wheels are large. After that the app is live at `https://<service-name>.onrender.com`.

The free plan sleeps after about 15 minutes of inactivity, so the first request after a nap takes a few seconds to wake the service. The `--preload` flag builds the synthetic session once in the gunicorn master process and shares it with the workers through copy-on-write.

### Other hosts

| Target | How |
|---|---|
| Docker (any cloud, VPS, Cloud Run, or Fly.io) | [`Dockerfile`](Dockerfile), which honours the `$PORT` variable |
| Heroku-style platforms | [`Procfile`](Procfile), with `web: gunicorn wsgi:server ...` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  NASDAQ FTP (ITCH 5.0)  │  Yahoo Finance v8  │  PCAP replay     │
└──────────┬──────────────┴────────┬────────────┴──────┬──────────┘
           │                       │                   │
           ▼                       ▼                   ▼
   ItchDownloader           DailyDownloader     itch_parser.cpp
   download_itch.py         download_daily.py   pcap_replayer.cpp
           │                       │            feed_handler.cpp
           ▼                       ▼                   │ LOBSnapshot
       DataCatalog           Parquet store             ▼  (seqlock SHM)
       (SQLite)               data/          ┌──────────────────────┐
                                             │  limit_order_book.cpp│
                                             │  price_level.cpp     │
                                             │  order.hpp           │
                                             └──────────┬───────────┘
                                                        │
                                          ┌─────────────▼────────────┐
                                          │  shm_writer.cpp (seqlock)│
                                          │  ring_buffer.hpp (SPSC)  │
                                          │  event_bus.cpp (typed)   │
                                          │  order_router.cpp        │
                                          │  fill_simulator.cpp      │
                                          └─────────────┬────────────┘
                                                        │  ShmReader
                                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│                       Python research stack                       │
│                                                                   │
│  microstructure/    signals/          validation/                 │
│  ├ ofi.py           ├ feature_pipeline  ├ purged_cv.py            │
│  ├ pin_model.py     ├ mean_reversion    ├ walk_forward.py         │
│  ├ spread_decomp    ├ momentum          ├ sharpe_deflator         │
│  ├ queue_model      ├ cointegration     ├ regime_tester           │
│  └ avellaneda_s     ├ kalman_pairs      └ tca.py                  │
│                     ├ hmm_regime                                  │
│  execution/         ├ garch_x           risk/                     │
│  ├ market_impact    └ signal_combiner   ├ kelly_sizer.py          │
│  ├ vwap.py                              ├ pnl_reporter.py         │
│  ├ twap.py          backtester/         ├ position_tracker        │
│  └ participation    ├ engine.py         └ circuit_breakers        │
│                     ├ portfolio.py                                │
│  dashboard/         └ tearsheet.py      data/                     │
│  ├ app.py                               ├ download_itch.py        │
│  ├ lob_depth_chart                      ├ download_daily.py       │
│  ├ ofi_panel                            └ data_catalog.py         │
│  ├ pnl_panel                                                      │
│  └ regime_panel                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## C++ Core

<details>
<summary><b>Limit Order Book</b> (<code>core/lob/</code>)</summary>

The book is dual-sided and stored by price level. Asks live in a `std::map<Price, PriceLevel>` in ascending order, and bids live in a `std::map<Price, PriceLevel, std::greater<>>` in descending order. Each `PriceLevel` holds a `std::list<Order>` for FIFO priority and an O(1) `total_qty` counter. An auxiliary `std::unordered_map<OrderId, iterator>` cancel map gives O(1) cancellation, which matters on NASDAQ because cancel rates exceed 95 percent.

| Operation | Complexity |
|---|---|
| `add_order` | O(log P), where P is the number of distinct price levels |
| `cancel_order` | O(1) via the cancel map and a list erase |
| `execute_order` (partial or full) | O(1) |
| `best_bid` and `best_ask` | O(1) via `map::begin()` |
| `depth(n)` | O(n) |
</details>

<details>
<summary><b>ITCH 5.0 Parser</b> (<code>core/feed_handler/</code>)</summary>

This is a stateless framing loop over raw binary. The frame layout is a 2-byte big-endian length, a 1-byte type, and a body. The body header is always 10 bytes (`stock_locate`, `tracking_number`, and `timestamp_hi` and `timestamp_lo`). Prices are `uint32` values in units of one ten-thousandth. The parser handles all of the LOB-relevant message types: Add Order (`A` and `F`), Execute (`E` and `C`), Cancel (`X`), Delete (`D`), Replace (`U`), and Trade (`P` and `Q`). There is no virtual dispatch and no heap allocation per message.
</details>

<details>
<summary><b>Lock-Free SPSC Ring Buffer</b> (<code>core/shared_memory/ring_buffer.hpp</code>)</summary>

The `head_` and `tail_` atomics are aligned to 64-byte cache lines and use acquire and release ordering. The capacity N must be a power of two so the index can use a bitmask. There is zero heap allocation after construction, and `try_push` and `try_pop` are non-blocking.
</details>

<details>
<summary><b>Seqlock SHM Writer</b> (<code>core/shared_memory/shm_writer.cpp</code>)</summary>

This publishes `LOBSnapshot` structs to a POSIX shared-memory region using a seqlock. The writer increments the sequence to an odd value before the write and back to an even value after it. Readers spin until they observe a stable even sequence with no change across their copy window. This gives wait-free reads that stay correct under concurrent writes.
</details>

<details>
<summary><b>TypedEventBus</b> (<code>core/event_bus/</code>)</summary>

This uses a `std::variant<Fill, Quote, Signal, Order, ...>` and a `std::unordered_map<type_index, vector<callback>>`. The `subscribe<T>(cb)` and `publish(event)` calls dispatch by `std::type_index`. There are no virtual functions and no per-event heap allocation.
</details>

---

## Python Research Stack

See [METHODOLOGY.md](METHODOLOGY.md) for the full mathematical derivations and verified numerical results.

| Module | Model | Key Result |
|---|---|---|
| `microstructure/ofi.py` | Cont, Kukanov and Stoikov (2014) multi-level OFI | beta = 6e-6, R-squared = 0.44 |
| `microstructure/avellaneda_stoikov.py` | Avellaneda-Stoikov stochastic-control market making | Spread = 1.2928 at q=0, t=0.5 |
| `microstructure/pin_model.py` | Easley et al. (1996) EM estimation | PIN = alpha*mu / (alpha*mu + 2*epsilon) |
| `signals/hmm_regime.py` | K-state Gaussian HMM, Baum-Welch in log space | Converged in 28 iterations, LL = 32,097.7 |
| `signals/garch_x.py` | GARCH(1,1) with an exogenous regressor | alpha plus beta recovered within 0.08 percent of truth |
| `signals/kalman_pairs.py` | Time-varying Kalman hedge ratio | Q and R tuned by log predictive likelihood |
| `signals/cointegration.py` | Engle-Granger and Johansen | ADF = -16.19, p = 4.2e-29 |
| `execution/market_impact.py` | Almgren-Chriss optimal liquidation | Expected cost = 2.265 bps, kappa = 30.075 |
| `execution/vwap.py` | VWAP with U-shaped and flat profiles | Implementation shortfall = 2.50 bps (10k shares, 390 buckets) |
| `validation/purged_cv.py` | De Prado (2018) purged K-fold | Train = 790, test = 200, embargo = 10 (fold 0) |
| `validation/sharpe_deflator.py` | Bailey and López de Prado (2014) PSR, DSR, and MTRL | PSR = 0.784, MTRL = 1,098 observations at SR around 0.78 |
| `risk/kelly_sizer.py` | Full, half, and fractional Kelly | f* = 0.325 at p = 0.55, payoff = 2x |
| `backtester/engine.py` | Event-driven, with ShmReader and ItchReplayer modes | 1,379 fills over 2,000 snapshots |

---

## Repository Layout

```
limit-order-book/
├── core/                        # C++17 latency-critical engine
│   ├── lob/                     # Order, PriceLevel, LimitOrderBook
│   ├── feed_handler/            # ITCH parser, PCAP replayer, FeedHandler
│   ├── shared_memory/           # Seqlock ShmWriter, SPSC RingBuffer
│   ├── event_bus/               # TypedEventBus, event_types
│   └── execution/               # OrderRouter, FillSimulator
├── python/                      # Python research stack
│   ├── microstructure/          # OFI, PIN, spread decomp, AS, queue model
│   ├── signals/                 # Feature pipeline, MR, momentum, HMM, GARCH-X, Kalman
│   ├── validation/              # Purged CV, walk-forward, PSR and DSR, TCA
│   ├── execution/               # VWAP, TWAP, participation rate, Almgren-Chriss
│   ├── risk/                    # Kelly, PnL reporter, position tracker, circuit breakers
│   ├── backtester/              # Engine, Portfolio, Tearsheet
│   └── dashboard/               # Plotly Dash app, four panels, and assets
├── data/                        # Data utilities (ITCH and daily downloaders, SQLite catalog)
├── tests/
│   ├── cpp/                     # 4 Catch2 test suites (133 tests)
│   └── python/                  # 9 pytest modules (416 tests)
├── docs/images/                 # Dashboard snapshots
├── wsgi.py                      # Production WSGI entry point (gunicorn wsgi:server)
├── render.yaml, Procfile, Dockerfile   # Deploy configs
├── CMakeLists.txt               # CMake 3.20 or newer, FetchContent Catch2 v3.5.4
├── pyproject.toml               # PEP 517 and 518, pytest, ruff, and mypy config
└── .github/workflows/           # C++ and Python CI matrices (12 jobs)
```

---

## Installation (full build)

The prerequisites are CMake 3.20 or newer, GCC 12 or newer or Clang 14 or newer, and Python 3.10 or newer.

### C++ build

```bash
# Release
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLOB_SANITIZE=OFF
cmake --build build --parallel $(nproc)
cd build && ctest --output-on-failure --parallel $(nproc)

# Debug with ASan and UBSan
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DLOB_SANITIZE=ON
cmake --build build-debug --parallel $(nproc)
cd build-debug && ctest --output-on-failure
```

### Python install and tests

```bash
pip install -e ".[dev]"                           # adds pytest, ruff, and mypy
PYTHONPATH=python pytest tests/python/ -v          # 416 tests
PYTHONPATH=python pytest tests/python/ --cov=python --cov-report=html:htmlcov
```

### CLI entry points (after `pip install -e .`)

```bash
lob-download-itch  --dest ~/data/itch  --start 2024-01-02 --end 2024-01-31
lob-download-daily --dest ~/data/daily --symbols AAPL MSFT SPY --start 2020-01-01
lob-catalog status
```

---

## Verified Performance Numbers

All of these figures are produced by running the code in this repository. See [METHODOLOGY.md](METHODOLOGY.md) for the derivations.

| Component | Metric | Value |
|---|---|---|
| OFI to change-in-mid regression | beta and R-squared | 6e-6 and 0.44 |
| HMM with K=2, Baum-Welch | Iterations to convergence | 28 |
| GARCH-X (alpha=0.05, beta=0.93) | alpha plus beta recovered | 0.9879 (true value 0.98) |
| Engle-Granger (beta=1.5, T=2000) | Hedge ratio and ADF | 1.4923 and -16.191 |
| Almgren-Chriss (Q=10k, T=1 day) | Expected cost and kappa | 2.265 bps and 30.075 |
| VWAP (10k shares, 390 buckets) | Implementation shortfall | 2.50 bps |
| Avellaneda-Stoikov (q=0, t=0.5) | Bid-ask spread | 1.2928 |
| Kelly (p=0.55, payoff=2x) | Full Kelly fraction | 0.3250 |
| PSR (T=252, SR around 0.78) | PSR and minimum track record | 0.7843 and 1,098 observations |
| Dashboard replay (2,000 snaps, seed=42) | Fills and max drawdown | 1,379 and 0.0614 percent |

---

## Continuous Integration

The C++ matrix is defined in [`build_cpp.yml`](.github/workflows/build_cpp.yml). It runs 6 jobs: Ubuntu 22.04 with GCC 12 and Clang 17, and macOS 14 with AppleClang, each in Release and in Debug with ASan and UBSan. Catch2 v3.5.4 is fetched with FetchContent.

The Python matrix is defined in [`test_python.yml`](.github/workflows/test_python.yml). It runs 6 jobs across Ubuntu 22.04 and macOS 14 with Python 3.10, 3.11, and 3.12. It runs Ruff lint, pytest with pytest-xdist, a Codecov upload, and offline smoke tests for the data utilities.

---

## References

<details>
<summary>Click to expand</summary>

- Almgren, R. and Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5-39.
- Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.
- Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94-107.
- Cont, R., Kukanov, A. and Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.
- Cont, R. and de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM Journal on Financial Mathematics*, 4(1), 1-25.
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
- Easley, D., Kiefer, N. M., O'Hara, M. and Paperman, J. B. (1996). Liquidity, information, and infrequently traded stocks. *Journal of Finance*, 51(4), 1405-1436.
- Engle, R. F. and Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251-276.
- Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market. *Journal of Financial Economics*, 14(1), 71-100.
- Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2), 231-254.
- Moskowitz, T. J., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228-250.
- NASDAQ (2019). *ITCH 5.0 Protocol Specification*.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance*, 39(4), 1127-1139.

</details>

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.

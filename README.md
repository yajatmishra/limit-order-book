<div align="center">

# ⚡ Sigma Edge — C++17 Limit Order Book

**Low-latency C++17 market-data & execution core, paired with a full Python microstructure research stack and a live Plotly Dash dashboard.**

[![C++ Build & Tests](https://github.com/yajatmishra/limit-order-book/actions/workflows/build_cpp.yml/badge.svg)](https://github.com/yajatmishra/limit-order-book/actions/workflows/build_cpp.yml)
[![Python Tests](https://github.com/yajatmishra/limit-order-book/actions/workflows/test_python.yml/badge.svg)](https://github.com/yajatmishra/limit-order-book/actions/workflows/test_python.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](CMakeLists.txt)
[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10–3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-133%20C%2B%2B%20%2B%20416%20Py-success)](tests/)

[**Live Demo**](#-live-demo) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [Deployment](#-deployment) · [Methodology](METHODOLOGY.md)

</div>

---

## Overview

A quantitative trading research platform built around a low-latency **C++17** market-data and execution core, integrated with a **Python** research layer for signal generation, strategy validation, execution optimisation, and risk management.

The system ingests raw NASDAQ ITCH 5.0 market data, reconstructs the full limit order book in real time, generates microstructure-based signals, simulates execution, and evaluates performance through an event-driven backtesting pipeline. The two layers communicate through a **shared-memory seqlock interface**, letting Python models consume real-time order-book snapshots with minimal overhead.

- **C++17 core** — limit order book reconstruction, binary ITCH 5.0 parsing, lock-free SPSC messaging, seqlock shared-memory snapshot publishing, typed event dispatch, and order/fill simulation.
- **Python research stack** — market-microstructure models, statistical signal research, walk-forward validation, execution algorithms, risk sizing, and a live Plotly Dash dashboard.

---

## ▶ Live Demo

An interactive **session-replay dashboard** — five linked panels rendering a synthetic trading day (2,000 ITCH snapshots, `seed=42`) entirely offline. Dark theme, responsive layout, scrub/play controls.

> **Try it:** deploy your own in ~5 minutes — see [Deployment](#-deployment). Once live it runs at `https://limit-order-book-dashboard.onrender.com`.

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
| **LOB Depth** | Symmetric mountain chart of resting bid (green) / ask (red) quantity across 5 levels, cumulative depth overlay, mid-price rule. Scrubs across snapshots. |
| **P&L** | Equity curve, per-bar returns, underwater drawdown, and a monospace metrics box (Sharpe, Sortino, Calmar, MaxDD, PSR, DSR). |
| **OFI** | Rolling order-flow imbalance (Cont, Kukanov & Stoikov 2014) with ±1σ bands, plus a ΔMid-vs-OFI scatter with OLS fit. |
| **Regime** | Mid-price coloured by a 2-state Gaussian-HMM Viterbi path, with a stacked posterior-probability area chart. |

---

## 🚀 Quick Start

### Run the dashboard locally

```bash
git clone https://github.com/yajatmishra/limit-order-book.git
cd limit-order-book
pip install -r requirements.txt

# Dev server (Dash) …
PYTHONPATH=python python python/dashboard/app.py      # → http://localhost:8050

# … or the production server (exactly what Render runs)
gunicorn wsgi:server --preload --bind 0.0.0.0:8050
```

### Run with Docker

```bash
docker build -t lob-dashboard .
docker run --rm -p 8050:8050 lob-dashboard            # → http://localhost:8050
```

---

## 🌐 Deployment

The dashboard ships production-ready: a [`wsgi.py`](wsgi.py) entry point exposes the Flask `server`, served by **gunicorn**. Deploy configs are included for the common targets.

### Render (one-click)

This repo contains a [`render.yaml`](render.yaml) Blueprint. Either click the **Deploy to Render** button above, or:

1. Push the repo to GitHub.
2. On [render.com](https://render.com) → **New → Blueprint** → connect the repo.
3. Render reads `render.yaml`, installs `requirements.txt`, and starts:
   ```
   gunicorn wsgi:server --workers 1 --threads 8 --timeout 120 --preload --bind 0.0.0.0:$PORT
   ```
4. First build takes a few minutes (numpy/scipy/pandas wheels); the app is then live at `https://<service-name>.onrender.com`.

> The free plan sleeps after ~15 min idle — the first request after a nap takes a few seconds to wake. `--preload` builds the synthetic session once in the gunicorn master and shares it with workers via copy-on-write.

### Anywhere else

| Target | How |
|---|---|
| **Docker** (any cloud / VPS / Cloud Run / Fly.io) | [`Dockerfile`](Dockerfile) — honours `$PORT` |
| **Heroku-style PaaS** | [`Procfile`](Procfile) — `web: gunicorn wsgi:server …` |

---

## 🏗 Architecture

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

## 🧩 C++ Core

<details>
<summary><b>Limit Order Book</b> — <code>core/lob/</code></summary>

Dual-sided price-level LOB. Asks in `std::map<Price, PriceLevel>` (ascending), bids in `std::map<Price, PriceLevel, std::greater<>>` (descending). Each `PriceLevel` holds a `std::list<Order>` for FIFO priority and an O(1) `total_qty` counter. An auxiliary `std::unordered_map<OrderId, iterator>` cancel map gives O(1) cancellation — critical for NASDAQ where cancel rates exceed 95%.

| Operation | Complexity |
|---|---|
| `add_order` | O(log P) — P = distinct price levels |
| `cancel_order` | O(1) — cancel map + list erase |
| `execute_order` (partial or full) | O(1) |
| `best_bid` / `best_ask` | O(1) — `map::begin()` |
| `depth(n)` | O(n) |
</details>

<details>
<summary><b>ITCH 5.0 Parser</b> — <code>core/feed_handler/</code></summary>

Stateless framing loop over raw binary: `[2-byte BE length][1-byte type][body]`. Body header is always 10 bytes (`stock_locate`, `tracking_number`, `timestamp_hi/lo`). Prices are `uint32` in units of 1/10000. Handles all LOB-relevant message types: Add Order (`A`/`F`), Execute (`E`/`C`), Cancel (`X`), Delete (`D`), Replace (`U`), Trade (`P`/`Q`). No virtual dispatch, no heap allocation per message.
</details>

<details>
<summary><b>Lock-Free SPSC Ring Buffer</b> — <code>core/shared_memory/ring_buffer.hpp</code></summary>

64-byte cache-line-aligned `head_` and `tail_` atomics with acquire/release ordering. Capacity N must be a power of 2 (bitmask index). Zero heap allocation after construction. `try_push` / `try_pop` are non-blocking.
</details>

<details>
<summary><b>Seqlock SHM Writer</b> — <code>core/shared_memory/shm_writer.cpp</code></summary>

Publishes `LOBSnapshot` structs to a POSIX shared-memory region using a seqlock: writer increments sequence to odd before write, back to even after. Readers spin until they observe a stable even sequence with no change across their copy window — wait-free reads, correct under concurrent writes.
</details>

<details>
<summary><b>TypedEventBus</b> — <code>core/event_bus/</code></summary>

`std::variant<Fill, Quote, Signal, Order, …>` + `std::unordered_map<type_index, vector<callback>>`. `subscribe<T>(cb)` and `publish(event)` dispatch by `std::type_index` — no virtual functions, no per-event heap allocation.
</details>

---

## 📈 Python Research Stack

See [METHODOLOGY.md](METHODOLOGY.md) for full mathematical derivations and verified numerical results.

| Module | Model | Key Result |
|---|---|---|
| `microstructure/ofi.py` | Cont, Kukanov & Stoikov (2014) multi-level OFI | β = 6×10⁻⁶, R² = 0.44 |
| `microstructure/avellaneda_stoikov.py` | AS stochastic-control market making | Spread = 1.2928 at q=0, t=0.5 |
| `microstructure/pin_model.py` | Easley et al. (1996) EM estimation | PIN = αμ / (αμ + 2ε) |
| `signals/hmm_regime.py` | K-state Gaussian HMM, Baum-Welch (log-space) | Converged in 28 iters, LL = 32,097.7 |
| `signals/garch_x.py` | GARCH(1,1) + exogenous regressor | α+β recovered to within 0.08% of truth |
| `signals/kalman_pairs.py` | Time-varying Kalman hedge ratio | Q/R tuned by log predictive likelihood |
| `signals/cointegration.py` | Engle-Granger + Johansen | ADF = −16.19, p = 4.2×10⁻²⁹ |
| `execution/market_impact.py` | Almgren-Chriss optimal liquidation | E[cost] = 2.265 bps, κ = 30.075 |
| `execution/vwap.py` | VWAP with U-shaped/flat profiles | IS = 2.50 bps (10k shares, 390 buckets) |
| `validation/purged_cv.py` | De Prado (2018) purged K-fold | Train=790, test=200, embargo=10 (fold 0) |
| `validation/sharpe_deflator.py` | Bailey & López de Prado (2014) PSR/DSR/MTRL | PSR=0.784, MTRL=1,098 obs at SR≈0.78 |
| `risk/kelly_sizer.py` | Full/half/fractional Kelly | f*=0.325 at p=0.55, payoff=2× |
| `backtester/engine.py` | Event-driven, ShmReader + ItchReplayer modes | 1,379 fills over 2,000 snaps |

---

## 📂 Repository Layout

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
│   ├── validation/              # Purged CV, walk-forward, PSR/DSR, TCA
│   ├── execution/               # VWAP, TWAP, participation rate, Almgren-Chriss
│   ├── risk/                    # Kelly, PnL reporter, position tracker, circuit breakers
│   ├── backtester/              # Engine, Portfolio, Tearsheet
│   └── dashboard/               # Plotly Dash app + 4 panels + assets/
├── data/                        # Data utilities (ITCH/daily downloaders, SQLite catalog)
├── tests/
│   ├── cpp/                     # 4 Catch2 test suites (133 tests)
│   └── python/                  # 9 pytest modules (416 tests)
├── docs/images/                 # Dashboard snapshots
├── wsgi.py                      # Production WSGI entry point (gunicorn wsgi:server)
├── render.yaml · Procfile · Dockerfile   # Deploy configs
├── CMakeLists.txt               # CMake 3.20+, FetchContent Catch2 v3.5.4
├── pyproject.toml               # PEP 517/518, pytest + ruff + mypy config
└── .github/workflows/           # C++ and Python CI matrices (12 jobs)
```

---

## 🛠 Installation (full build)

**Prerequisites:** CMake ≥ 3.20, GCC ≥ 12 or Clang ≥ 14, Python ≥ 3.10.

### C++ build

```bash
# Release
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLOB_SANITIZE=OFF
cmake --build build --parallel $(nproc)
cd build && ctest --output-on-failure --parallel $(nproc)

# Debug + ASan/UBSan
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DLOB_SANITIZE=ON
cmake --build build-debug --parallel $(nproc)
cd build-debug && ctest --output-on-failure
```

### Python install & tests

```bash
pip install -e ".[dev]"                           # + pytest, ruff, mypy
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

## ✅ Verified Performance Numbers

All figures are produced by running the code in this repository. See [METHODOLOGY.md](METHODOLOGY.md) for derivations.

| Component | Metric | Value |
|---|---|---|
| OFI → ΔMid regression | β · R² | 6 × 10⁻⁶ · 0.44 |
| HMM K=2, Baum-Welch | Iterations to convergence | 28 |
| GARCH-X (α=0.05, β=0.93) | α + β recovered | 0.9879 (true 0.98) |
| Engle-Granger (β=1.5, T=2000) | Hedge ratio · ADF | 1.4923 · −16.191 |
| Almgren-Chriss (Q=10k, T=1d) | E[cost] · κ | 2.265 bps · 30.075 |
| VWAP (10k shares, 390 buckets) | Implementation shortfall | 2.50 bps |
| Avellaneda-Stoikov (q=0, t=0.5) | Bid-ask spread | 1.2928 |
| Kelly (p=0.55, payoff=2×) | Full Kelly fraction | 0.3250 |
| PSR (T=252, SR≈0.78) | PSR · min track record | 0.7843 · 1,098 obs |
| Dashboard replay (2,000 snaps, seed=42) | Fills · max drawdown | 1,379 · 0.0614% |

---

## 🔬 CI

- **C++ matrix** ([`build_cpp.yml`](.github/workflows/build_cpp.yml)) — 6 jobs: Ubuntu 22.04 × {GCC-12, Clang-17} × {Release, Debug+ASan+UBSan} and macOS 14 × AppleClang × {Release, Debug+ASan}. Catch2 v3.5.4 via FetchContent.
- **Python matrix** ([`test_python.yml`](.github/workflows/test_python.yml)) — 6 jobs: {Ubuntu 22.04, macOS 14} × {Python 3.10, 3.11, 3.12}. Ruff lint, pytest with `pytest-xdist`, Codecov upload, and offline smoke tests for the data utilities.

---

## 📚 References

<details>
<summary>Click to expand</summary>

- Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.
- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47–88.
- Cont, R. & de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM Journal on Financial Mathematics*, 4(1), 1–25.
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
- Easley, D., Kiefer, N. M., O'Hara, M. & Paperman, J. B. (1996). Liquidity, information, and infrequently traded stocks. *Journal of Finance*, 51(4), 1405–1436.
- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251–276.
- Glosten, L. R. & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market. *Journal of Financial Economics*, 14(1), 71–100.
- Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.
- Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
- NASDAQ (2019). *ITCH 5.0 Protocol Specification*.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance*, 39(4), 1127–1139.

</details>

---

## 📄 License

MIT — see [LICENSE](LICENSE).

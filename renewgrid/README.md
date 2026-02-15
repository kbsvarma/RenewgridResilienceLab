# RenewGrid

Reproducible Python 3.11 project for renewable-grid resilience experiments. This phase includes a "hello pipeline" using NASA POWER and EIA.

## Setup

```bash
cd renewgrid
uv sync --extra dev
uv run python -c "import pyarrow; print(pyarrow.__version__)"
cp .env.example .env
```

Alternate setup without `uv`:

```bash
cd renewgrid
pip install -e ".[dev]"
cp .env.example .env
```

Parquet output requires pyarrow; if you see to_parquet engine errors, your install is incomplete.

If `uv` is not installed, the `Makefile` falls back to system `python` and CLI tools. `make pipeline` runs a preflight check and prints whether `pyarrow` is available.

## Commands

- `make test`: run pytest.
- `make lint`: run ruff, black, isort checks.
- `make run`: launch Streamlit app.
- `make pipeline`: run the Phase 0 hello pipeline and write daily parquet outputs.
- `make inspect`: print row counts and columns for hello-pipeline parquet outputs.
- `make preview`: print first 5 rows from hello-pipeline parquet outputs.
- `make phase1`: build CAISO/ERCOT daily datasets and write evaluation reports.

## Phase 0 Quickstart

```bash
cd renewgrid
uv sync --extra dev
cp .env.example .env
make pipeline
make inspect
```

Phase 0 processed outputs are daily:
- `data/processed/nasa_power_daily.parquet`
- `data/processed/eia_rto_daily.parquet`

## Phase 1 Quickstart

```bash
cd renewgrid
uv sync --extra dev
cp .env.example .env
PYTHONPATH=src python -m renewgrid.scripts.phase1_run --days 180
```

or with pip:

```bash
cd renewgrid
pip install -e ".[dev]"
cp .env.example .env
PYTHONPATH=src python -m renewgrid.scripts.phase1_run --days 180
```

Outputs:
- datasets: `data/processed/CAISO_daily_<start>_<end>.parquet` and `data/processed/ERCOT_daily_<start>_<end>.parquet`
- reports: `reports/phase1/*_demand_mw_avg_report.json` and `.md`
- when optional RARE columns are available, additional reports are produced for `solar_gen`/`wind_gen` (or `solar_cf`/`wind_cf`)

## Config

Environment variables:
- `EIA_KEY`: EIA Open Data API key.
- `NREL_KEY`: NREL developer API key.

Default region presets:
- `CAISO`: respondent `CISO`
- `ERCOT`: respondent `ERCO`

Daily demand aggregation choice:
- EIA hourly demand is aggregated to UTC-day **mean** value and exposed as `demand_mw_avg`.
- This keeps units consistent as MW-average at daily resolution for Phase 1.

## Units Contract

- Canonical demand unit for Phase 1 and Phase 2 is **daily average power in MW**, stored as `demand_mw_avg`.
- If daily energy is needed, derive it as: `demand_mwh = demand_mw_avg * 24`.
- Daily aggregation and timestamps use **UTC day** boundaries.

Conversion helpers live in `src/renewgrid/util/units.py`.

## Data Schema Contract

Required columns:
- `date`
- `demand_mw_avg`

Optional columns:
- `solar_cf`, `wind_cf` (preferred canonical renewable signals)
- `solar_gen_mwh`, `wind_gen_mwh` (fallback when capacity factors cannot be computed)
- `weather_*` columns from NASA POWER

Validation helpers live in `src/renewgrid/util/schema.py`.

Optional RARE validation file:
- Provide `--rare-path /absolute/path/to/rare_daily.parquet` to merge optional daily solar/wind series.

## RARE Optional Input Schema

Expected base columns:
- `date`
- `region`

Supported value variants:
- preferred canonical: `solar_cf`, `wind_cf` (0..1)
- alternatively: `solar_gen`, `wind_gen`, plus optional `solar_capacity`, `wind_capacity`

Normalization behavior:
- if capacities are present, capacity factors are computed and canonicalized to `solar_cf`/`wind_cf`
- if capacities are missing, generation is retained as `solar_gen_mwh`/`wind_gen_mwh` with a warning

## Data Freshness And Limitations

- This repository is a **research stress-test tool**, not an operational forecasting platform.
- Daily series are derived from external APIs and subject to source latency/revisions.
- Phase 1 reports are for comparative model benchmarking, not real-time dispatch operations.
- Reproducibility scope: deterministic daily aggregation/feature logic with network-free tests; upstream API data can still change over time.
- Data-source policy: no scraping and no paywalled sources in the core pipeline.

## Modules

### `renewgrid.data`
API connectors for NASA POWER, EIA, NSRDB, RARE/PUDL and pipeline merge utilities.

### `renewgrid.features`
Feature builders on merged weather-demand data.

### `renewgrid.forecast`
Forecast baselines and model wrappers with evaluation helpers.

### `renewgrid.opt`
Dispatch optimization utilities.

### `renewgrid.stress`
Stress simulation and later ERA5 calibration hooks (Phase 2).

### `renewgrid.metrics`
Resilience metrics and curve generation.

### `renewgrid.app`
Streamlit UI entrypoint.

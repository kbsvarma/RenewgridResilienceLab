# RenewGrid

Reproducible Python 3.11 project for renewable-grid resilience experiments. This phase includes a "hello pipeline" using NASA POWER and EIA.

## Setup

```bash
cd renewgrid
uv sync --extra dev
cp .env.example .env
```

If `uv` is not installed, the `Makefile` falls back to system `python` and CLI tools.

## Commands

- `make test`: run pytest.
- `make lint`: run ruff, black, isort checks.
- `make run`: launch Streamlit app.
- `make pipeline`: run the Phase 1 hello pipeline and write parquet outputs.
- `make inspect`: print row counts and columns for hello-pipeline parquet outputs.
- `make preview`: print first 5 rows from hello-pipeline parquet outputs.

## Config

Environment variables:
- `EIA_KEY`: EIA Open Data API key.
- `NREL_KEY`: NREL developer API key.

Default region presets:
- `CAISO`: respondent `CISO`
- `ERCOT`: respondent `ERCO`

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

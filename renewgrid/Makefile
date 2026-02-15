.PHONY: test lint run preflight-pyarrow pipeline inspect preview

UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
	RUN_PYTEST := pytest
	RUN_RUFF := ruff
	RUN_BLACK := black
	RUN_ISORT := isort
	RUN_STREAMLIT := streamlit
	RUN_PYTHON := python
else
	RUN_PYTEST := uv run pytest
	RUN_RUFF := uv run ruff
	RUN_BLACK := uv run black
	RUN_ISORT := uv run isort
	RUN_STREAMLIT := uv run streamlit
	RUN_PYTHON := uv run python
endif

test:
	$(RUN_PYTEST)

lint:
	$(RUN_RUFF) check src tests
	$(RUN_BLACK) --check src tests
	$(RUN_ISORT) --check-only src tests

run:
	$(RUN_STREAMLIT) run src/renewgrid/app/streamlit_app.py

preflight-pyarrow:
	@PYTHONPATH=src $(RUN_PYTHON) -c "import sys; from renewgrid.util.parquet import has_parquet_engine; ok = has_parquet_engine(); print(f'pyarrow_available={ok}'); sys.exit(0 if ok else 1)"

pipeline: preflight-pyarrow
	PYTHONPATH=src $(RUN_PYTHON) -m renewgrid.data.merge

inspect:
	PYTHONPATH=src $(RUN_PYTHON) -c "from pathlib import Path; import pandas as pd; base=Path('data/processed'); files=('nasa_power_daily.parquet','eia_rto_hourly.parquet'); [print(f'--- {name} ---') or print('missing' if not (base/name).exists() else 'rows='+str(len((df:=pd.read_parquet(base/name))))+' columns='+str(list(df.columns))) for name in files]"

preview:
	PYTHONPATH=src $(RUN_PYTHON) -c "from pathlib import Path; import pandas as pd; base=Path('data/processed'); files=('nasa_power_daily.parquet','eia_rto_hourly.parquet'); [print(f'--- {name} (head=5) ---') or print('missing' if not (base/name).exists() else pd.read_parquet(base/name).head(5).to_string(index=False)) for name in files]"

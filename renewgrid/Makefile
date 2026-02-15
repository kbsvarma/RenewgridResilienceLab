.PHONY: test lint run pipeline inspect

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

pipeline:
	PYTHONPATH=src $(RUN_PYTHON) -m renewgrid.data.merge

inspect:
	PYTHONPATH=src $(RUN_PYTHON) -c "from pathlib import Path; import pandas as pd; base=Path('data/processed'); files=('nasa_power_daily.parquet','eia_rto_daily.parquet'); [print(f'--- {name} ---') or print('missing' if not (base/name).exists() else 'rows='+str(len((df:=pd.read_parquet(base/name))))+' columns='+str(list(df.columns))) for name in files]"

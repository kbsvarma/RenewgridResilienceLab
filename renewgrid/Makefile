.PHONY: test lint run

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run black --check src tests
	uv run isort --check-only src tests

run:
	uv run streamlit run src/renewgrid/app/streamlit_app.py

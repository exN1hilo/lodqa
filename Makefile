CONFIG ?= configs/main.yaml

.PHONY: install test lint assets experiments figures render demo all

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

assets:
	uv run python -m lodq.meshes

experiments:
	uv run python -m lodq.experiments --config $(CONFIG)

figures:
	uv run python -m lodq.plots

render:
	uv run python -m lodq.render

demo:
	uv run streamlit run app/streamlit_app.py

all: install test

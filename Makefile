QUICK ?= 0
ifeq ($(QUICK),1)
CONFIG = configs/smoke.yaml
else
CONFIG = configs/main.yaml
endif

START := $(shell date +%s)

# $(call timed,command): run command, then print how long this target took.
define timed
@echo "==> $@"
@start=$$(date +%s); $(1); status=$$?; \
	echo "<== $@: $$(( $$(date +%s) - start )) s"; exit $$status
endef

.PHONY: install test lint assets experiments figures render precompute demo all

install:
	$(call timed,uv sync)

test:
	$(call timed,uv run pytest -q)

lint:
	uv run ruff check src tests

assets:
	$(call timed,uv run python -m lodq.meshes)

experiments:
	$(call timed,uv run python -m lodq.experiments --config $(CONFIG))

figures:
	$(call timed,uv run python -m lodq.plots)

render:
	$(call timed,uv run python -m lodq.render)

precompute:
	uv run python scripts/precompute_path.py --exact

demo:
	uv run streamlit run app/streamlit_app.py

all: install test assets experiments figures render
	@echo "all: $$(( $$(date +%s) - $(START) )) s total (CONFIG=$(CONFIG))"

UV ?= uv
PYTHON ?= python
RUN_ID ?= demo001
PRED_DIR ?= data/runs/$(RUN_ID)/predictions
REPORT ?= data/reports/$(RUN_ID).md

.PHONY: install test lint report dirs

install:
	$(UV) sync --group dev --group dev-mcp --extra pdf --extra glmocr

dirs:
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw','data/gold','data/runs','data/reports']]"

test:
	$(UV) run --group dev-mcp pytest

lint:
	$(UV) run ruff check src tests

report:
	$(UV) run $(PYTHON) -m free_doc_extract.cli eval --gold-dir data/gold --pred-dir $(PRED_DIR) --output $(REPORT)

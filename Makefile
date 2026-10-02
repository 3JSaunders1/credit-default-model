SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

RUN_ID  := $(shell date +%Y%m%d_%H%M%S)
RUN_DIR := reports/runs/$(RUN_ID)
LOG     := $(RUN_DIR)/run_log.txt

.PHONY: all run-dir download data features train evaluate recalibrate macro explain archive test

all: run-dir data features train evaluate recalibrate macro explain archive
	@echo "Run complete: $(RUN_DIR)"

run-dir:
	@mkdir -p $(RUN_DIR)

download:
	python -m src.credit_default.download

data: run-dir
	python -m src.credit_default.data 2>&1 | tee -a $(LOG)

features: run-dir
	python -m src.credit_default.features 2>&1 | tee -a $(LOG)

train: run-dir
	python -m src.credit_default.train 2>&1 | tee -a $(LOG)

evaluate: run-dir
	python -m src.credit_default.evaluate 2>&1 | tee -a $(LOG)

recalibrate: run-dir
	python -m src.credit_default.recalibrate 2>&1 | tee -a $(LOG)

macro: run-dir
	python -m src.credit_default.fred 2>&1 | tee -a $(LOG)
	python -m src.credit_default.compare_macro 2>&1 | tee -a $(LOG)

explain: run-dir
	python -m src.credit_default.explain 2>&1 | tee -a $(LOG)

archive: run-dir
	cp reports/figures/* $(RUN_DIR)/
	@echo "Run ID:     $(RUN_ID)" > $(RUN_DIR)/run_info.txt
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'not committed yet')" >> $(RUN_DIR)/run_info.txt
	@echo "" >> $(RUN_DIR)/run_info.txt
	@echo "--- Config used ---" >> $(RUN_DIR)/run_info.txt
	@cat src/credit_default/config.py >> $(RUN_DIR)/run_info.txt

test:
	python -m pytest -v
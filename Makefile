SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

RUN_ID  := $(shell date +%Y%m%d_%H%M%S)
RUN_DIR := reports/runs/$(RUN_ID)
LOG     := $(RUN_DIR)/run_log.txt

.PHONY: all run-dir download data features train evaluate thresholds recalibrate macro explain monitor woe el tune spark archive test docker-build docker-test docker-run

all: run-dir data features train evaluate thresholds recalibrate macro explain monitor woe el archive
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

thresholds: run-dir
	python -m src.credit_default.thresholds 2>&1 | tee -a $(LOG)

recalibrate: run-dir
	python -m src.credit_default.recalibrate 2>&1 | tee -a $(LOG)

macro: run-dir
	python -m src.credit_default.fred 2>&1 | tee -a $(LOG)
	python -m src.credit_default.compare_macro 2>&1 | tee -a $(LOG)

explain: run-dir
	python -m src.credit_default.explain 2>&1 | tee -a $(LOG)

monitor: run-dir
	python -m src.credit_default.monitor 2>&1 | tee -a $(LOG)

woe: run-dir
	python -m src.credit_default.woe 2>&1 | tee -a $(LOG)

el: run-dir
	python -m src.credit_default.expected_loss 2>&1 | tee -a $(LOG)

tune: run-dir
	python -m src.credit_default.tune 2>&1 | tee -a $(LOG)

spark: run-dir
	python -m src.credit_default.spark_pipeline 2>&1 | tee -a $(LOG)
	python -m src.credit_default.compare_engines 2>&1 | tee -a $(LOG)

archive: run-dir
	cp reports/figures/* $(RUN_DIR)/
	@echo "Run ID:     $(RUN_ID)" > $(RUN_DIR)/run_info.txt
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'not committed yet')" >> $(RUN_DIR)/run_info.txt
	@echo "" >> $(RUN_DIR)/run_info.txt
	@echo "--- Config used ---" >> $(RUN_DIR)/run_info.txt
	@cat src/credit_default/config.py >> $(RUN_DIR)/run_info.txt

test:
	python -m pytest -v

docker-build:   ## Build the Docker image
	docker build -t credit-default-model .

docker-test: docker-build   ## Run the test suite inside the container
	docker run --rm credit-default-model

docker-run: docker-build    ## Run the full pipeline inside the container, using local data and credentials
	docker run --rm \
	-v "$(PWD)/data:/app/data" \
	-v "$(PWD)/models:/app/models" \
	-v "$(PWD)/reports:/app/reports" \
	-v "$(HOME)/.kaggle:/root/.kaggle:ro" \
	--env-file .env \
	credit-default-model make all
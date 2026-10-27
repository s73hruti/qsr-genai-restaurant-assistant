.PHONY: install install-llm install-vector corpus data train test lint eval eval-llm retrieval-neural ablation calibrate api ui monitor docker

install:
	pip install -e ".[dev]"

install-llm:
	pip install -e ".[dev,llm]"

install-vector:
	pip install -e ".[dev,vector]"

corpus:            ## regenerate knowledge base + retrieval eval set
	python scripts/build_corpus.py

data:
	python scripts/generate_data.py

train:
	python scripts/train_all.py

test:
	pytest -q

lint:
	ruff check src tests scripts

eval:              ## offline evaluation (no API key); fails if thresholds regress
	python -m qsr_assistant.eval.evaluate --check

eval-llm:          ## real provider + LLM judge; needs API key and optional judge provider key
	python -m qsr_assistant.eval.evaluate --llm online --workers 1

retrieval-neural:   ## neural retrieval benchmark; downloads model weights on first run
	python scripts/run_neural_retrieval_benchmark.py

ablation:          ## retrieval ablation only (writes reports/ablation.md via `make eval`)
	python -c "from qsr_assistant.eval.retrieval_eval import run_ablation, to_markdown; print(to_markdown(run_ablation()))"

calibrate:         ## re-calibrate the retrieval abstention threshold
	python -m qsr_assistant.eval.abstain

api:
	uvicorn qsr_assistant.api.main:app --reload --port 8000

ui:
	streamlit run src/qsr_assistant/app/streamlit_app.py

monitor:
	python -m qsr_assistant.monitoring.drift

docker:
	docker compose up --build

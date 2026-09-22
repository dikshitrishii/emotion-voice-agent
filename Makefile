.PHONY: install install-dev test lint format typecheck run-backend run-frontend benchmark

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check backend tests

format:
	ruff format backend tests

typecheck:
	mypy backend

run-backend:
	uvicorn backend.main:app --reload --port 8000

run-frontend:
	streamlit run frontend/app.py

benchmark:
	python scripts/benchmark_latency.py

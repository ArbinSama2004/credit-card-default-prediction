.DEFAULT_GOAL := help
.PHONY: help mlflow-up mlflow-down mlflow-logs mlflow-ui \
        backend-install backend-run backend-test backend-lock \
        frontend-install frontend-run frontend-lock \
        docker-build docker-up docker-down clean

## ---- help -----------------------------------------------------------------

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## ---- MLflow (Stage 1) ------------------------------------------------------

mlflow-up: ## Build and start the local MLflow tracking server (http://localhost:5001)
	docker compose up -d --build mlflow

mlflow-down: ## Stop the MLflow container
	docker compose stop mlflow

mlflow-logs: ## Tail MLflow container logs
	docker compose logs -f mlflow

mlflow-ui: ## Open the MLflow UI in the browser
	open http://localhost:5001

## ---- Backend (Stage 2) ------------------------------------------------------

backend-install: ## Sync backend uv environment
	cd backend && uv sync

backend-run: ## Run the FastAPI dev server on :8000
	cd backend && uv run uvicorn app.main:app --reload --port 8000

backend-test: ## Run backend test suite
	cd backend && uv run pytest -v

backend-lock: ## Re-lock backend dependencies
	cd backend && uv lock

## ---- Frontend (Stage 3) -----------------------------------------------------

frontend-install: ## Sync frontend uv environment
	cd frontend && uv sync

frontend-run: ## Run the Streamlit dashboard on :8501
	cd frontend && uv run streamlit run app/Home.py

frontend-lock: ## Re-lock frontend dependencies
	cd frontend && uv lock

## ---- Docker (mlflow + backend as of Stage 2; frontend joins in Stage 3) -----

docker-build: ## Build all service images
	docker compose build

docker-up: ## Start mlflow + backend (frontend once uncommented in Stage 3)
	docker compose up -d --build

docker-down: ## Stop the full stack
	docker compose down

## ---- Housekeeping ------------------------------------------------------------

clean: ## Stop containers and remove local MLflow/Docker volumes
	docker compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

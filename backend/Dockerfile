# Build context is the REPO ROOT (see docker-compose.yml) so we can pull in
# ml/artifacts without duplicating them into the backend/ folder.
FROM python:3.12-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first for better layer caching.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App code.
COPY backend/app ./app

# Trained model + preprocessing artifacts, produced in Colab (Stage 1.5)
# and downloaded into ml/artifacts/ on the host.
COPY ml/artifacts ./artifacts

ENV PATH="/app/.venv/bin:$PATH" \
    ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

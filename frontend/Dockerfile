FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY .streamlit ./.streamlit

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/Home.py"]

# Local MLflow tracking server.
#
# Uses the classic FileStore backend (not a SQL backend) *on purpose*: the
# default local tracking mode you get in a Colab notebook
# (`mlflow.set_tracking_uri("file:./mlruns")`, or no call at all) also uses
# FileStore. That means the `mlruns/` folder you download from Colab can be
# copied straight into this server's data directory and it just shows up —
# no DB import step. See docs/mlflow-workflow.md.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.16.2

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "file:/mlflow/mlruns", \
     "--default-artifact-root", "file:/mlflow/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

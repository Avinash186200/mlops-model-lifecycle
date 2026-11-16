# Scoring API image. The registry is mounted at run time, not baked in, so
# a promote or rollback does not need a new image:
#   docker build -t mlcycle-api .
#   docker run -p 8000:8000 -v "$PWD/artifacts/registry:/registry" mlcycle-api
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" && useradd --create-home app

USER app
ENV MLCYCLE_REGISTRY=/registry
ENV MLCYCLE_PREDICTION_LOG=/home/app/predictions.sqlite
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "mlcycle.serving:app", "--host", "0.0.0.0", "--port", "8000"]

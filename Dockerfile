FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY data/knowledge ./data/knowledge
RUN pip install -e ".[llm,qdrant,graph,mcp]"
# Bake data + models into the image so the container starts instantly
RUN python scripts/train_all.py \
 && useradd -m app && chown -R app /app
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health')" || exit 1
# $PORT is honoured so the same image runs on Render, Cloud Run, Fly and Hugging Face Spaces
CMD ["sh", "-c", "uvicorn qsr_assistant.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

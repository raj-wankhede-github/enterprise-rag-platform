# The parser service: Docling with OCR and table-structure recovery.
#
# Pinned to Python 3.12 for the same reason as the models container: the ML wheel ecosystem
# trails the language, and the API must not be held back by it.
#
# This is the largest image in the stack by a wide margin, which is exactly why it is a separate
# one. A deployment that never ingests a scan never builds or runs it -- it sits behind the
# `parse` compose profile, and the loader registry returns a clean "no loader handles this"
# error when it is absent rather than failing at import time.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=4

# Docling's OCR and image handling need these system libraries; the slim base has none of them.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      poppler-utils \
      tesseract-ocr \
      tesseract-ocr-eng \
      tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
      "fastapi>=0.118" \
      "uvicorn[standard]>=0.37" \
      "python-multipart>=0.0.20" \
      "docling>=2.20"

# Models are fetched at build time, not on boot. A service that downloads weights to start
# cannot start in an air-gapped BYOC deployment, and turns a registry outage into an outage of
# the ingestion path.
RUN python -c "from docling.utils.model_downloader import download_models; download_models()" || \
    echo "model prefetch unavailable in this docling version; first parse will fetch"

COPY server.py /srv/server.py

RUN useradd --create-home --uid 10001 parser && chown -R parser:parser /srv
USER parser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=10 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

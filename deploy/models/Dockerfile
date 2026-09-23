# The models service: cross-encoder reranking and embeddings, over HTTP.
#
# Pinned to Python 3.12, not the 3.14 the API uses. That is the whole reason this is a separate
# container rather than a library: the ML wheel ecosystem trails the language by a release or
# two, and coupling the API's runtime to it would hold the API back indefinitely.
#
# ONNX Runtime rather than torch. torch CPU is 1-2.5 GB of image for inference we can do in
# ~150 MB, and on a 24-pair batch at 288 tokens the int8 ONNX path is 1.5-3x faster.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_WAIT_POLICY=PASSIVE \
    OMP_NUM_THREADS=4 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=0

WORKDIR /srv

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
      "fastapi>=0.118" \
      "uvicorn[standard]>=0.37" \
      "onnxruntime>=1.20" \
      "optimum[onnxruntime]>=1.24" \
      "transformers>=4.48" \
      "tokenizers>=0.21"

# The model is baked into the image, not downloaded at start. A service that fetches weights on
# boot cannot start in an air-gapped BYOC deployment, and turns a registry outage into an
# outage of the whole answer path.
ARG RERANKER_MODEL=BAAI/bge-reranker-base
ENV RERANKER_MODEL=${RERANKER_MODEL}

# The bi-encoder. A separate model because a cross-encoder scores pairs and has no per-text
# vector to extract -- they share a runtime and a container, not weights.
#
# bge-base-en-v1.5 is the default: 768 dimensions, Apache-2.0, and strong on the retrieval
# benchmarks that resemble this workload. Set EMBEDDING_MODEL=none at build time to omit it, and
# /embed then answers 503 rather than serving wrong vectors.
ARG EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}

COPY export_model.py /srv/export_model.py
RUN python /srv/export_model.py --model "${RERANKER_MODEL}" --out /srv/model       --task sequence-classification

# Two RUN layers rather than one: the reranker is cached when only the embedder changes, which
# is most of the time while the embedder is being chosen.
RUN if [ "${EMBEDDING_MODEL}" != "none" ]; then       python /srv/export_model.py --model "${EMBEDDING_MODEL}" --out /srv/embedder         --task feature-extraction ;     else echo "no embedder baked in; /embed will answer 503" ; fi

COPY server.py /srv/server.py

RUN useradd --create-home --uid 10001 models && chown -R models:models /srv
USER models

EXPOSE 8000
# A longer start period than the reranker alone needed: two models load and both warm up at
# their served batch shape before /healthz turns green.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

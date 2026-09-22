"""Index settings and mappings.

Built in Python rather than shipped as static JSON because the vector dimension, the kNN engine
and the quantization mode all vary by deployment, and a templated string would put those in the
one place we most need to hash. ``mapping_digest`` and ``analysis_digest`` feed the generation
fingerprint, so any change here is automatically a new generation and a background rebuild --
which is the intended cost, not an accident to work around.

Two field-level decisions carry most of the retrieval quality:

* ``content.exact`` uses a whitespace tokenizer plus ``word_delimiter_graph`` with
  ``preserve_original`` and ``catenate_all``, as **both** the index and the search analyzer. That
  is what makes ``AB-1234/X``, ``AB1234X`` and ``ab 1234 x`` reach the same postings. Dense
  vectors are unreliable on opaque identifiers; this field is why BM25 earns its place.
* ``bm25_body`` uses ``b = 0.4`` rather than the 0.75 default. Chunks are already length
  normalized to 200-400 tokens, so the default over-penalizes the legitimately longer table
  chunks -- which are exactly the ones holding numbers that appear nowhere in prose.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

KnnEngine = Literal["lucene", "faiss", "nmslib"]
KnnMode = Literal["in_memory", "on_disk"]

_ANALYSIS_PATH: Final[Path] = Path(__file__).with_name("analysis.json")

#: Bumped by hand when the *shape* of a mapping changes in a way the digest would not capture
#: on its own -- for example renaming a field whose old name is still read somewhere.
MAPPING_VERSION: Final[str] = "v1"


@lru_cache(maxsize=1)
def analysis_settings() -> dict[str, Any]:
    """The shared analysis chain. Identical in the chunk and parent indices, by construction:
    a parent-leg query and a child-leg query must tokenize the same text the same way.
    """
    with _ANALYSIS_PATH.open(encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    return result


def _digest(payload: Any) -> str:
    """A stable 16-hex digest of a JSON-serializable structure.

    ``sort_keys`` matters: dict ordering must not change the fingerprint, or a harmless
    refactor would trigger a full reindex of every tenant.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def analysis_digest() -> str:
    return _digest(analysis_settings())


def _knn_method(*, engine: KnnEngine, m: int, ef_construction: int, quantize: bool) -> dict[str, Any]:
    method: dict[str, Any] = {
        "name": "hnsw",
        "engine": engine,
        "parameters": {"m": m, "ef_construction": ef_construction},
    }
    if quantize:
        # Lucene's 7-bit scalar quantization keeps full-precision values for rescoring, so the
        # graph shrinks ~4x while recall stays near-lossless. The aggressive step (on_disk
        # binary quantization with oversampled rescoring) is a deployment mode, not a default.
        method["parameters"]["encoder"] = {
            "name": "sq",
            "parameters": {"confidence_interval": 0.9},
        }
    return method


def chunk_mapping(
    *,
    dimension: int,
    engine: KnnEngine = "lucene",
    m: int = 16,
    ef_construction: int = 128,
    quantize: bool = False,
) -> dict[str, Any]:
    """The child-chunk mapping: the only index holding vectors."""
    return {
        # A connector that invents a field fails loudly at ingest instead of silently creating a
        # mapping that differs between pools -- which would make two tenants' indices
        # incomparable without anyone noticing.
        "dynamic": "strict",
        # Halves stored size. We never read the vector back: rebuilds re-embed from Postgres, or
        # reuse the fp16 copy in chunk_vectors.
        "_source": {"excludes": ["embedding"]},
        "properties": {
            # --- identity and generation -----------------------------------------------------
            "tenant_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "doc_version_id": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "collection_id": {"type": "keyword"},
            "ordinal": {"type": "integer"},
            "generation_fingerprint": {"type": "keyword"},
            # --- searchable text -------------------------------------------------------------
            "content": {
                "type": "text",
                "analyzer": "text_en",
                "similarity": "bm25_body",
                "fields": {"exact": {"type": "text", "analyzer": "text_exact"}},
            },
            # The contextual-retrieval prefix. Indexed separately from content so citations can
            # quote only verbatim source text while both still contribute to BM25 and the vector.
            "context_line": {
                "type": "text",
                "analyzer": "text_en",
                "similarity": "bm25_body",
                "fields": {"exact": {"type": "text", "analyzer": "text_exact"}},
            },
            "title": {
                "type": "text",
                "analyzer": "text_en",
                "similarity": "bm25_title",
                "fields": {
                    "raw": {"type": "keyword", "normalizer": "lc_ascii", "ignore_above": 512},
                    "exact": {"type": "text", "analyzer": "text_exact"},
                },
            },
            "heading_path": {
                "type": "text",
                "analyzer": "text_en",
                "similarity": "bm25_title",
                "fields": {
                    "raw": {"type": "keyword", "normalizer": "lc_ascii", "ignore_above": 1024},
                    "tree": {"type": "text", "analyzer": "heading_path_tree"},
                },
            },
            # --- vector ----------------------------------------------------------------------
            "embedding": {
                "type": "knn_vector",
                "dimension": dimension,
                # Equivalent to cosine over L2-normalized vectors, and cheaper. Normalization is
                # the Embedder's responsibility and is part of its id.
                "space_type": "innerproduct",
                "method": _knn_method(engine=engine, m=m, ef_construction=ef_construction, quantize=quantize),
            },
            # --- access control (every field here is read by search/dsl.py) ------------------
            "visibility_rank": {"type": "integer"},
            "acl_mode": {"type": "keyword"},
            "access_groups": {"type": "keyword"},
            "denied_groups": {"type": "keyword"},
            "allowed_user_ids": {"type": "keyword"},
            "denied_user_ids": {"type": "keyword"},
            "is_active": {"type": "boolean"},
            # --- filterable metadata ---------------------------------------------------------
            "doc_type": {"type": "keyword", "normalizer": "lc_ascii"},
            "source_system": {"type": "keyword", "normalizer": "lc_ascii"},
            "source_uri": {"type": "keyword", "index": False, "doc_values": False},
            "author": {
                "type": "keyword",
                "normalizer": "lc_ascii",
                "fields": {"text": {"type": "text", "analyzer": "text_en"}},
            },
            "language": {"type": "keyword"},
            "publication_date": {"type": "date"},
            "effective_from": {"type": "date"},
            "effective_to": {"type": "date"},
            "is_superseded": {"type": "boolean"},
            "supersedes": {"type": "keyword"},
            "superseded_by": {"type": "keyword"},
            # rank_feature is available for in-engine priors, but the default is to apply
            # authority and recency AFTER fusion so each leg stays independently measurable.
            "authority_rank": {"type": "rank_feature"},
            "freshness": {"type": "rank_feature"},
            # --- provenance and dedup --------------------------------------------------------
            "block_kinds": {"type": "keyword"},
            "page_from": {"type": "integer"},
            "page_to": {"type": "integer"},
            "token_count": {"type": "integer"},
            "content_sha256": {"type": "keyword"},
            "simhash64": {"type": "keyword", "index": False},
            "indexed_at": {"type": "date"},
            "extra": {"type": "object", "enabled": False},
        },
    }


def parent_mapping() -> dict[str, Any]:
    """Parent sections: same analysis, no vector.

    Kept in a separate index rather than as a ``join`` field or ``nested`` documents. A join
    forces parent and child onto one shard and makes kNN on children awkward; ``nested`` cannot
    usefully hold a top-level ``knn_vector``. The cost is one batched ``_mget`` at expansion
    time; the benefit is a fourth retrieval leg -- BM25 over whole sections -- for free.
    """
    mapping = chunk_mapping(dimension=1)
    properties: dict[str, Any] = dict(mapping["properties"])
    for field in ("embedding", "parent_id", "ordinal", "chunk_id", "context_line", "simhash64"):
        properties.pop(field, None)
    properties["parent_ordinal"] = {"type": "integer"}
    properties["child_count"] = {"type": "integer"}
    return {"dynamic": "strict", "properties": properties}


def index_settings(
    *,
    shards: int,
    replicas: int,
    refresh_interval: str = "30s",
    knn: bool = True,
) -> dict[str, Any]:
    return {
        "index": {
            "knn": knn,
            "number_of_shards": shards,
            "number_of_replicas": replicas,
            "refresh_interval": refresh_interval,
            "similarity": {
                # b=0.4, not the 0.75 default: chunks are already length-normalized, so the
                # default over-penalizes longer table chunks.
                "bm25_body": {"type": "BM25", "k1": 0.9, "b": 0.4},
                "bm25_title": {"type": "BM25", "k1": 1.2, "b": 0.75},
            },
            "analysis": analysis_settings(),
        }
    }


def chunk_index_body(
    *,
    dimension: int,
    shards: int,
    replicas: int,
    engine: KnnEngine = "lucene",
    quantize: bool = False,
    refresh_interval: str = "30s",
) -> dict[str, Any]:
    return {
        "settings": index_settings(shards=shards, replicas=replicas, refresh_interval=refresh_interval),
        "mappings": chunk_mapping(dimension=dimension, engine=engine, quantize=quantize),
    }


def parent_index_body(*, shards: int, replicas: int, refresh_interval: str = "30s") -> dict[str, Any]:
    return {
        "settings": index_settings(shards=shards, replicas=replicas, refresh_interval=refresh_interval, knn=False),
        "mappings": parent_mapping(),
    }


def mapping_digest(*, dimension: int, engine: KnnEngine = "lucene", quantize: bool = False) -> str:
    """Digest of the chunk mapping. Part of the generation fingerprint.

    The dimension is included deliberately: switching embedding model changes it, which must
    force a new index rather than an incompatible write into the existing one.
    """
    return _digest(
        {
            "version": MAPPING_VERSION,
            "chunks": chunk_mapping(dimension=dimension, engine=engine, quantize=quantize),
            "parents": parent_mapping(),
        }
    )


def searchable_fields() -> frozenset[str]:
    """Top-level field names in the chunk mapping.

    Used by ``tests/unit/test_mappings.py`` to assert that every field ``search/dsl.py``
    filters on actually exists -- a filter on an unmapped field silently matches nothing under
    ``dynamic: strict``, which would look like "no results" rather than like a bug.
    """
    return frozenset(chunk_mapping(dimension=8)["properties"])

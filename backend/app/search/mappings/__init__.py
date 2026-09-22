"""OpenSearch index settings and mappings."""

from __future__ import annotations

from app.search.mappings.builder import (
    MAPPING_VERSION,
    analysis_digest,
    analysis_settings,
    chunk_index_body,
    chunk_mapping,
    index_settings,
    mapping_digest,
    parent_index_body,
    parent_mapping,
    searchable_fields,
)

__all__ = [
    "MAPPING_VERSION",
    "analysis_digest",
    "analysis_settings",
    "chunk_index_body",
    "chunk_mapping",
    "index_settings",
    "mapping_digest",
    "parent_index_body",
    "parent_mapping",
    "searchable_fields",
]

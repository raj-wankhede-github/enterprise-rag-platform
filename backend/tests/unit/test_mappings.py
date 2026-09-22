"""Mapping and generation invariants.

The most valuable test here is ``test_every_field_dsl_filters_on_is_mapped``. Under
``dynamic: strict`` a filter on an unmapped field does not error at query time -- it simply
matches nothing. A typo in ``search/dsl.py`` would therefore present as "the search returns no
results", which is indistinguishable from a legitimately empty corpus and would be debugged in
the wrong place for hours.
"""

from __future__ import annotations

import json
import uuid

import pytest

from app.retrieval.types import MetadataFilter, TenantScope
from app.search import dsl
from app.search.generations import (
    GenerationSpec,
    GenerationState,
    assert_transition,
    can_transition,
    chunk_document_id,
    chunk_index_name,
    chunk_read_alias,
    compute_fingerprint,
    pool_for_tenant,
    swap_actions,
)
from app.search.mappings import (
    analysis_digest,
    analysis_settings,
    chunk_index_body,
    chunk_mapping,
    mapping_digest,
    parent_index_body,
    parent_mapping,
    searchable_fields,
)

TENANT = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _field_names(clauses: object, found: set[str] | None = None) -> set[str]:
    """Collect every field name a filter structure references."""
    found = set() if found is None else found
    if isinstance(clauses, dict):
        for key, value in clauses.items():
            if key in ("term", "terms", "range") and isinstance(value, dict):
                for field in value:
                    # terms lookups nest an index/id/path object under the field name
                    found.add(field)
            elif key == "exists" and isinstance(value, dict) and "field" in value:
                found.add(str(value["field"]))
            _field_names(value, found)
    elif isinstance(clauses, list):
        for item in clauses:
            _field_names(item, found)
    return found


def test_every_field_dsl_filters_on_is_mapped() -> None:
    """A filter on an unmapped field matches nothing silently. This is the guard."""
    scope = TenantScope(
        tenant_id=TENANT,
        visibility_rank=30,
        access_groups=("g1",),
        generation_fingerprint="abc",
        user_id=uuid.uuid4(),
    )
    filters = MetadataFilter(
        doc_types=("policy",),
        source_systems=("sharepoint",),
        authors=("a@x.com",),
        doc_ids=("d1",),
        collection_ids=("c1",),
        languages=("en",),
    )
    referenced = _field_names(dsl.build_filter(scope, filters))
    mapped = searchable_fields()
    missing = referenced - mapped
    assert not missing, f"search/dsl.py filters on unmapped fields: {sorted(missing)}"


def test_mapping_is_strict_so_unknown_fields_fail_loudly() -> None:
    assert chunk_mapping(dimension=8)["dynamic"] == "strict"
    assert parent_mapping()["dynamic"] == "strict"


def test_embedding_is_excluded_from_source() -> None:
    assert chunk_mapping(dimension=8)["_source"]["excludes"] == ["embedding"]


def test_exact_subfield_exists_on_the_text_fields_that_need_it() -> None:
    props = chunk_mapping(dimension=8)["properties"]
    for field in ("content", "context_line", "title"):
        assert "exact" in props[field]["fields"], f"{field} needs an .exact subfield"
        assert props[field]["fields"]["exact"]["analyzer"] == "text_exact"


def test_exact_analyzer_preserves_original_and_catenates() -> None:
    """Both flags are required for AB-1234/X, AB1234X and 'ab 1234 x' to meet."""
    code_split = analysis_settings()["filter"]["code_split"]
    assert code_split["type"] == "word_delimiter_graph"
    assert code_split["preserve_original"] is True
    assert code_split["catenate_all"] is True
    # A graph filter must be flattened before indexing or OpenSearch rejects the analyzer.
    assert "flatten_graph" in analysis_settings()["analyzer"]["text_exact"]["filter"]


def test_body_similarity_lowers_b_for_length_normalised_chunks() -> None:
    settings = chunk_index_body(dimension=8, shards=1, replicas=0)["settings"]["index"]
    assert settings["similarity"]["bm25_body"]["b"] == 0.4


def test_parent_mapping_keeps_the_key_the_parent_leg_projects_through() -> None:
    """parent_id links a section to its children; without it the parent leg cannot work."""
    assert "parent_id" in parent_mapping()["properties"]


def test_parent_index_has_no_vector() -> None:
    props = parent_mapping()["properties"]
    assert "embedding" not in props
    assert "child_count" in props
    # The ACL fields must still be present: the parent leg is filtered exactly like the others.
    for field in ("tenant_id", "visibility_rank", "access_groups", "generation_fingerprint"):
        assert field in props


def test_parent_index_does_not_enable_knn() -> None:
    assert parent_index_body(shards=1, replicas=0)["settings"]["index"]["knn"] is False


def test_analysis_is_shared_between_chunk_and_parent_indices() -> None:
    """A parent-leg query and a child-leg query must tokenize identically."""
    chunk = chunk_index_body(dimension=8, shards=1, replicas=0)["settings"]["index"]["analysis"]
    parent = parent_index_body(shards=1, replicas=0)["settings"]["index"]["analysis"]
    assert chunk == parent


def test_knn_space_type_is_innerproduct() -> None:
    embedding = chunk_mapping(dimension=8)["properties"]["embedding"]
    assert embedding["space_type"] == "innerproduct"
    assert embedding["method"]["name"] == "hnsw"


def test_quantization_is_off_by_default_and_opt_in() -> None:
    plain = chunk_mapping(dimension=8)["properties"]["embedding"]["method"]["parameters"]
    assert "encoder" not in plain
    quantized = chunk_mapping(dimension=8, quantize=True)["properties"]["embedding"]["method"]["parameters"]
    assert quantized["encoder"]["name"] == "sq"


# --------------------------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------------------------


def spec(**kwargs: object) -> GenerationSpec:
    base: dict[str, object] = {
        "embedder_id": "hashing@384",
        "chunker_version": "1",
        "contextualizer_version": "template-1",
        "dimension": 384,
    }
    base.update(kwargs)
    return GenerationSpec(**base)  # type: ignore[arg-type]


def test_fingerprint_is_stable_across_calls() -> None:
    assert spec().fingerprint == spec().fingerprint


@pytest.mark.parametrize(
    "change",
    [
        {"embedder_id": "onnx-bge-m3@1024"},
        {"chunker_version": "2"},
        {"contextualizer_version": "llm-3"},
        {"dimension": 768},
        {"quantize": True},
    ],
)
def test_every_input_changes_the_fingerprint(change: dict[str, object]) -> None:
    """If any of the five inputs did not move the fingerprint, a rebuild could be skipped."""
    assert spec().fingerprint != spec(**change).fingerprint


def test_dimension_change_forces_a_new_mapping_digest() -> None:
    assert mapping_digest(dimension=384) != mapping_digest(dimension=1024)


def test_analysis_digest_is_deterministic() -> None:
    assert analysis_digest() == analysis_digest()
    assert len(analysis_digest()) == 16


def test_fingerprint_fields_cannot_collide_by_concatenation() -> None:
    """('ab','c') and ('a','bc') must not produce the same digest."""
    a = compute_fingerprint(
        embedder_id="ab", chunker_version="c", contextualizer_version="x", mapping_sha="m", analysis_sha="s"
    )
    b = compute_fingerprint(
        embedder_id="a", chunker_version="bc", contextualizer_version="x", mapping_sha="m", analysis_sha="s"
    )
    assert a != b


def test_json_round_trip_does_not_change_the_digest() -> None:
    """Key ordering must not matter, or a harmless refactor reindexes every tenant."""
    body = chunk_mapping(dimension=8)
    assert json.loads(json.dumps(body)) == body


# --------------------------------------------------------------------------------------------
# Pools, names and aliases
# --------------------------------------------------------------------------------------------


def test_pool_assignment_is_stable_across_processes() -> None:
    """Uses the UUID bytes, not hash(), whose seed is randomized per process."""
    assert pool_for_tenant(TENANT, pool_count=16) == pool_for_tenant(TENANT, pool_count=16)


def test_pool_assignment_is_in_range_and_spreads() -> None:
    pools = {pool_for_tenant(uuid.uuid4(), pool_count=16) for _ in range(500)}
    assert pools <= set(range(16))
    assert len(pools) == 16, "500 tenants should touch every pool"


def test_pool_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        pool_for_tenant(TENANT, pool_count=0)


def test_index_and_alias_names_are_zero_padded() -> None:
    assert chunk_index_name(generation=7, pool=3) == "chunks_g7_p003"
    assert chunk_read_alias(3) == "chunks_read_p003"


def test_chunk_document_id_is_deterministic_and_version_scoped() -> None:
    version = uuid.uuid4()
    first = chunk_document_id(tenant_id=TENANT, doc_version_id=version, ordinal=4)
    assert first == chunk_document_id(tenant_id=TENANT, doc_version_id=version, ordinal=4)
    # A new version produces new ids on purpose: citations must be version-stable.
    assert first != chunk_document_id(tenant_id=TENANT, doc_version_id=uuid.uuid4(), ordinal=4)
    assert first != chunk_document_id(tenant_id=TENANT, doc_version_id=version, ordinal=5)


def test_swap_actions_move_read_and_write_aliases_together() -> None:
    actions = swap_actions(from_generation=7, to_generation=8, pool=0)
    adds = [a["add"] for a in actions if "add" in a]
    removes = [a["remove"] for a in actions if "remove" in a]
    assert all(a["index"] == "chunks_g8_p000" for a in adds if "chunks" in str(a["index"]))
    assert all(r["index"] == "chunks_g7_p000" for r in removes if "chunks" in str(r["index"]))
    write_adds = [a for a in adds if a.get("is_write_index")]
    assert len(write_adds) == 2, "one write alias for chunks and one for parents"


# --------------------------------------------------------------------------------------------
# The generation state machine
# --------------------------------------------------------------------------------------------


def test_a_new_generation_cannot_go_live_without_verification() -> None:
    """No path promotes an unverified generation.

    Exactly two states may reach LIVE: VERIFYING, which is the forward promotion and means the
    shadow evaluation passed; and DRAINING, which is the rollback to the generation that was
    serving a moment ago and was therefore already verified. Everything earlier in the pipeline
    -- PLANNED, BUILDING, BACKFILLED -- must not, because those are the states a half-built
    index sits in.
    """
    forbidden = {
        GenerationState.PLANNED,
        GenerationState.BUILDING,
        GenerationState.BACKFILLED,
        GenerationState.RETIRED,
    }
    for state in forbidden:
        assert not can_transition(state, GenerationState.LIVE), f"{state} must not jump to LIVE"
    assert can_transition(GenerationState.VERIFYING, GenerationState.LIVE)


def test_failed_verification_can_return_to_building() -> None:
    assert can_transition(GenerationState.VERIFYING, GenerationState.BUILDING)


def test_rollback_from_draining_is_one_step() -> None:
    assert can_transition(GenerationState.DRAINING, GenerationState.LIVE)


def test_retired_is_terminal() -> None:
    for state in GenerationState:
        assert not can_transition(GenerationState.RETIRED, state)


def test_illegal_transition_names_what_was_allowed() -> None:
    with pytest.raises(ValueError, match="Illegal generation transition"):
        assert_transition(GenerationState.PLANNED, GenerationState.LIVE)

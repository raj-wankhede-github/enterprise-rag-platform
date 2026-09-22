"""Mappings against a real engine.

Unit tests prove the mapping dict has the shape we intend. Only OpenSearch can prove it is
*accepted*, and only OpenSearch can prove the analyzers actually produce the tokens the design
depends on. Analysis chains are exactly the kind of thing that looks right in JSON and silently
does something else -- a missing ``flatten_graph``, a filter in the wrong order, a normalizer
that quietly drops a character class.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

from app.search.mappings import chunk_index_body, parent_index_body

pytestmark = pytest.mark.integration

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    with httpx.Client(base_url=OPENSEARCH_URL, timeout=30.0) as session:
        try:
            session.get("/_cluster/health").raise_for_status()
        except (httpx.HTTPError, httpx.ConnectError) as exc:  # pragma: no cover - env dependent
            pytest.skip(f"OpenSearch not reachable at {OPENSEARCH_URL}: {exc}")
        yield session


@pytest.fixture
def chunk_index(client: httpx.Client) -> str:
    name = f"test-chunks-{uuid.uuid4().hex[:8]}"
    body = chunk_index_body(dimension=8, shards=1, replicas=0, refresh_interval="1s")
    response = client.put(f"/{name}", json=body)
    assert response.status_code in (200, 201), response.text
    yield name
    client.delete(f"/{name}")


def test_chunk_mapping_is_accepted(chunk_index: str, client: httpx.Client) -> None:
    mapping = client.get(f"/{chunk_index}/_mapping").json()[chunk_index]["mappings"]
    assert mapping["dynamic"] == "strict"
    assert mapping["properties"]["embedding"]["type"] == "knn_vector"
    assert mapping["properties"]["authority_rank"]["type"] == "rank_feature"


def test_parent_mapping_is_accepted(client: httpx.Client) -> None:
    name = f"test-parents-{uuid.uuid4().hex[:8]}"
    response = client.put(f"/{name}", json=parent_index_body(shards=1, replicas=0))
    assert response.status_code in (200, 201), response.text
    try:
        props = client.get(f"/{name}/_mapping").json()[name]["mappings"]["properties"]
        assert "embedding" not in props
    finally:
        client.delete(f"/{name}")


def test_strict_mapping_rejects_an_invented_field(chunk_index: str, client: httpx.Client) -> None:
    """A connector that adds a field must fail at ingest, not create per-pool mapping drift."""
    response = client.post(
        f"/{chunk_index}/_doc/x?refresh=true",
        json={"tenant_id": "A", "content": "hello", "invented_by_a_connector": "oops"},
    )
    assert response.status_code == 400
    assert "strict" in response.text


@pytest.mark.parametrize(
    ("indexed", "query"),
    [
        ("AB-1234/X", "AB-1234/X"),  # verbatim
        ("AB-1234/X", "ab1234x"),  # catenate_all collapses the separators
        ("AB-1234/X", "AB"),  # generate_word_parts
        ("AB-1234/X", "1234"),  # generate_number_parts
        ("SEC-4.2.1", "SEC-4.2.1"),
        ("TKT-99812", "TKT-99812"),
        ("TKT-99812", "99812"),
        ("SKU_88-A", "sku88a"),
    ],
)
def test_exact_analyzer_matches_identifier_variants(
    chunk_index: str, client: httpx.Client, indexed: str, query: str
) -> None:
    """The reason the exact leg exists. Dense vectors are unreliable on opaque identifiers."""
    client.post(
        f"/{chunk_index}/_doc/doc1?refresh=true",
        json={"tenant_id": "A", "content": f"the reference is {indexed} in section two"},
    )
    response = client.post(
        f"/{chunk_index}/_search",
        json={"query": {"match_phrase": {"content.exact": query}}},
    )
    hits = response.json()["hits"]["total"]["value"]
    assert hits == 1, f"indexed {indexed!r} should be findable as {query!r}, got {hits} hits"


def test_stemmed_analyzer_does_not_match_partial_identifiers(chunk_index: str, client: httpx.Client) -> None:
    """Sanity check that .exact is doing the work, not the stemmed field."""
    client.post(
        f"/{chunk_index}/_doc/doc1?refresh=true",
        json={"tenant_id": "A", "content": "the reference is AB-1234/X here"},
    )
    exact = client.post(
        f"/{chunk_index}/_search", json={"query": {"match_phrase": {"content.exact": "ab1234x"}}}
    ).json()["hits"]["total"]["value"]
    stemmed = client.post(f"/{chunk_index}/_search", json={"query": {"match_phrase": {"content": "ab1234x"}}}).json()[
        "hits"
    ]["total"]["value"]
    assert exact == 1
    assert stemmed == 0, "the stemmed field is not expected to collapse the separators"


EN_CHAIN = [
    "lowercase",
    "asciifolding",
    {"type": "stemmer", "language": "possessive_english"},
    {"type": "stemmer", "language": "light_english"},
]


def _tokens(client: httpx.Client, text: str) -> list[str]:
    response = client.post("/_analyze", json={"tokenizer": "standard", "filter": EN_CHAIN, "text": text})
    return [t["token"] for t in response.json()["tokens"]]


def test_english_analyzer_stems_and_folds(chunk_index: str, client: httpx.Client) -> None:
    client.post(
        f"/{chunk_index}/_doc/doc1?refresh=true",
        json={"tenant_id": "A", "content": "Employees are reimbursed for travelling invoices"},
    )
    for query in ("employee", "reimburse", "travel", "invoice"):
        hits = client.post(f"/{chunk_index}/_search", json={"query": {"match": {"content": query}}}).json()["hits"][
            "total"
        ]["value"]
        assert hits == 1, f"{query!r} should stem-match the indexed text"


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("report", "reports"),
        ("contract", "contracts"),
        ("payment", "payments"),
        ("document", "documents"),
        ("claim", "claims"),
        ("benefit", "benefits"),
        ("agreement", "agreements"),
        ("invoice", "invoices"),
        ("policy", "policies"),
        ("status", "statuses"),
        ("address", "addresses"),
        ("process", "processes"),
        ("employee", "employees"),
        ("company", "companies"),
    ],
)
def test_singular_and_plural_collapse_to_one_token(client: httpx.Client, singular: str, plural: str) -> None:
    """What actually matters for recall: both forms must reach the same postings."""
    assert _tokens(client, singular) == _tokens(client, plural)


@pytest.mark.parametrize("word", ["business", "organization", "reimbursement", "compliance", "analysis"])
def test_common_terms_are_not_over_stemmed(client: httpx.Client, word: str) -> None:
    """Over-stemming is the failure that costs BM25 its comparative advantage.

    porter2 scores 20/20 on singular/plural pairs but turns "organization" into "organ" and
    "business" into "busi", which collides unrelated vocabulary in the one leg whose job is
    precision. That is why it is not used despite the better pair score.
    """
    assert _tokens(client, word) == [word]


@pytest.mark.parametrize(("singular", "plural"), [("cost", "costs"), ("term", "terms"), ("expense", "expenses")])
def test_known_stemming_gaps_are_recorded_not_pretended_away(client: httpx.Client, singular: str, plural: str) -> None:
    """Three pairs in the benchmark do not collapse, and that is an accepted trade.

    Measured by ``bench/stemming_bench.py`` over 20 singular/plural pairs plus an over-stemming
    check: light_english and kstem both score 17/20 with zero mangling, minimal_english 16/20
    with 2 mangled, porter2 20/20 with 6 mangled. Buying these three pairs costs "organization"
    becoming "organ" across the whole corpus, which is a worse trade for a lexical leg.

    Morphological variation is what the **dense leg** is for -- this is the division of labour
    hybrid retrieval buys. Revisit with golden-set evidence at build step 6, not by argument.

    This test asserts the gap still exists so that an analyzer change cannot pass silently.
    """
    assert _tokens(client, singular) != _tokens(client, plural), (
        f"{plural!r} now collapses onto {singular!r}: the analyzer improved. "
        "Move this pair into test_singular_and_plural_collapse_to_one_token and re-run the bench."
    )

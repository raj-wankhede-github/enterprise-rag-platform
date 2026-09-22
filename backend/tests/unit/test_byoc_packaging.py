"""BYOC is a packaging exercise, not a second product.

That claim is easy to make and easy to let rot. These tests keep it honest without needing a
cluster: the BYOC compose file must reference the same image and the same entry points as the
SaaS one, must not turn anything off that would make it a different product, and must not let a
deployment silently fall back to infrastructure the customer did not provide.

The strongest check is the one in CI: the tenant-isolation suite runs against both compose files.
If BYOC ever needed a code branch, the isolation tests would be the first thing to diverge.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SAAS = ROOT / "docker-compose.yml"
BYOC = ROOT / "docker-compose.byoc.yml"


@pytest.fixture(scope="module")
def byoc() -> str:
    if not BYOC.exists():  # pragma: no cover - the file is committed
        pytest.skip("docker-compose.byoc.yml is not present")
    return BYOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def saas() -> str:
    return SAAS.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------
# Same product
# ------------------------------------------------------------------------------------------


def test_byoc_runs_the_same_entry_points_as_saas(byoc: str, saas: str) -> None:
    """A different command would mean a different product wearing the same name."""
    for command in ("alembic", "upgrade", "head"):
        assert command in byoc and command in saas
    assert "app.main:create_app" in byoc
    assert "app.workers.ingestion_worker" in byoc


def test_byoc_builds_nothing_of_its_own(byoc: str) -> None:
    """It consumes a published image. A `build:` here would let the two diverge silently, which
    is exactly the failure this file exists to prevent."""
    assert not re.search(r"^\s*build:", byoc, re.MULTILINE)


def test_byoc_does_not_disable_tenancy_or_acl_controls(byoc: str) -> None:
    """There is no BYOC mode in the code, and there must be no environment variable pretending
    there is. Single tenancy is a deployment fact, not a feature flag."""
    for forbidden in ("SINGLE_TENANT", "DISABLE_RLS", "SKIP_TENANT_CHECK", "BYOC_MODE", "DISABLE_ACL"):
        assert forbidden not in byoc


# ------------------------------------------------------------------------------------------
# Nothing falls back to infrastructure the customer did not provide
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variable", ["DATABASE_URL", "OPENSEARCH_URL", "REDIS_URL", "JWT_SECRET"])
def test_customer_infrastructure_has_no_default(byoc: str, variable: str) -> None:
    """A deployment that silently fell back to a bundled Postgres would be storing a customer's
    data somewhere neither party intended -- and the failure would be a working system rather
    than an error."""
    assert re.search(rf"\$\{{{variable}:\?", byoc), f"{variable} must be required, not defaulted"


def test_the_jwt_secret_has_no_development_default(byoc: str) -> None:
    """A deployment that started with the development secret would issue tokens anyone who has
    read this repository could forge."""
    assert "local-development-secret" not in byoc


def test_cookies_are_secure_and_not_configurable_downward(byoc: str) -> None:
    assert 'COOKIE_SECURE: "true"' in byoc
    assert "COOKIE_SECURE: ${" not in byoc


def test_certificate_verification_defaults_on(byoc: str) -> None:
    """The SaaS compose file turns it off for a local self-signed cluster. A production file
    inheriting that default would accept any certificate presented to it."""
    assert "OPENSEARCH_VERIFY_CERTS: ${OPENSEARCH_VERIFY_CERTS:-true}" in byoc


def test_redis_is_required_because_in_process_limits_cannot_span_replicas(byoc: str) -> None:
    """The file also declares two API replicas. Together those would silently double every rate
    limit if the in-memory limiter were reachable."""
    assert re.search(r"\$\{REDIS_URL:\?", byoc)
    assert "API_REPLICAS:-2" in byoc


# ------------------------------------------------------------------------------------------
# What is deliberately absent
# ------------------------------------------------------------------------------------------


def test_the_operator_plane_is_not_deployed(byoc: str) -> None:
    """A BYOC deployment has no vendor operators, so there is nobody for the plane to
    authenticate. Not deploying it is a stronger statement than deploying it empty -- it is the
    cleanest answer to "can the vendor read our data"."""
    assert "ops_api" not in byoc
    assert "create_ops_app" not in byoc


def test_content_capture_is_off_and_not_a_variable(byoc: str) -> None:
    """Here the answer to "does the vendor store our documents in a third-party tool" should be
    structural rather than configured."""
    assert 'TRACE_CONTENT_CAPTURE: "false"' in byoc
    assert "TRACE_CONTENT_CAPTURE: ${" not in byoc


def test_the_embedding_cache_is_tenant_scoped(byoc: str) -> None:
    """A global cache leaks nothing -- the key is a hash and the value a deterministic function
    of it -- but a single-tenant customer has no reason to accept even that argument."""
    assert "EMBEDDING_CACHE_SCOPE: tenant" in byoc


# ------------------------------------------------------------------------------------------
# Optional things are genuinely optional
# ------------------------------------------------------------------------------------------


def test_the_saml_broker_is_behind_a_profile(byoc: str) -> None:
    """Most customers do not need it, and a Keycloak nobody uses is a database, a certificate
    and a patching obligation."""
    assert "profiles: [saml]" in byoc


def test_the_brokers_variables_do_not_block_customers_who_do_not_want_saml(byoc: str) -> None:
    """Compose interpolates every variable in the file regardless of active profiles, so a
    `:?required` on a profiled service stops an unrelated customer starting the stack at all."""
    keycloak = byoc[byoc.index("keycloak:") :]
    assert ":?" not in keycloak, "a required variable on a profiled service blocks every deployment"


def test_the_model_containers_are_optional(byoc: str) -> None:
    """Left out, the product degrades to the hashing embedder and identity reranker and reports
    it in the ablation table. It does not fail to start."""
    assert "profiles: [models]" in byoc
    assert "profiles: [parse]" in byoc


def test_no_external_api_is_required(byoc: str) -> None:
    """A customer who wants nothing leaving their VPC sets LLM_PROVIDER=none and gets extractive
    answers with citations. That is the default here."""
    assert "LLM_PROVIDER: ${LLM_PROVIDER:-none}" in byoc
    assert "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}" in byoc


# ------------------------------------------------------------------------------------------
# The claim itself
# ------------------------------------------------------------------------------------------


def test_ci_runs_the_isolation_suite_against_the_byoc_file() -> None:
    """The check that actually keeps this honest.

    Everything above is a property of a YAML file. This is the one that would catch a real
    divergence -- and a CI job that quietly stopped running it would make every other test in
    this module a statement about nothing.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "docker-compose.byoc.yml" in workflow, "CI must exercise the BYOC topology"

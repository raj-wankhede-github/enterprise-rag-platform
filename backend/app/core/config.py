"""Typed settings.

Every tunable lives here with a comment explaining its default, and is mirrored in
``.env.example``. Production sizing is the design; the local compose file overrides a handful of
these downward, which is the only place the development machine is allowed to matter.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "staging", "production"]
LLMProvider = Literal["none", "anthropic", "openai"]
EmbeddingProvider = Literal["hashing", "onnx", "openai", "bedrock"]
RerankerProvider = Literal["identity", "lexical", "onnx", "cohere", "voyage", "jina"]
ContextualizerKind = Literal["template", "llm"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = "local"
    service_name: str = "enterprise-rag-platform"
    log_level: str = "INFO"
    log_json: bool = True

    # --- datastores -------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://erp:erp@localhost:5433/erp"
    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_statement_timeout_ms: int = 30_000

    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str | None = None
    opensearch_password: SecretStr | None = None
    opensearch_verify_certs: bool = True
    opensearch_request_timeout_s: float = 10.0
    # 16 pools balances "a tenant reindex is not a full-index operation" against cluster-state
    # size. Changing this after go-live requires rehashing every tenant binding, so it is fixed
    # at deployment time and recorded in tenant_index_bindings.
    opensearch_pool_count: int = 16
    opensearch_shards_per_pool: int = 3
    opensearch_replicas: int = 1

    redis_url: str | None = None  # None => in-process fallbacks (single replica only)

    # --- auth -------------------------------------------------------------------------
    jwt_secret: SecretStr = SecretStr("local-development-secret-change-me-32chars")
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 600  # 10 min; the principal is reloaded from PG each request
    refresh_token_ttl_seconds: int = 604_800  # 7 days absolute
    session_idle_seconds: int = 28_800  # 8 hours
    auth_cookie_name: str = "erp_at"
    refresh_cookie_name: str = "erp_rt"
    csrf_cookie_name: str = "erp_csrf"
    cookie_secure: bool = True
    cors_origins: list[str] = Field(default_factory=list)
    allowed_origins_strict: bool = True

    #: The externally visible origin, used to build the OIDC redirect URI.
    #:
    #: Configured rather than derived from the request, because the provider compares the
    #: redirect URI byte for byte against its registration -- and behind a proxy the request's
    #: own scheme and host are whatever the proxy chose to forward. Deriving it is how a working
    #: development setup becomes a "redirect_uri_mismatch" the first time it is deployed.
    public_base_url: str = "http://localhost:8001"

    # --- providers --------------------------------------------------------------------
    # The offline triple. CI runs on these: no paid API, no model download, deterministic output,
    # which is what makes the ablation table runnable on every pull request.
    llm_provider: LLMProvider = "none"
    embedding_provider: EmbeddingProvider = "hashing"
    reranker_provider: RerankerProvider = "identity"
    contextualizer: ContextualizerKind = "template"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5"
    openai_api_key: SecretStr | None = None

    models_service_url: str | None = None  # the ONNX embedder + reranker container
    parser_service_url: str | None = None  # the Docling container

    # --- retrieval --------------------------------------------------------------------
    # k=60 is the Cormack et al. default: it damps any single leg's top ranks, so a document
    # ranked ~5th by two legs beats one ranked 1st by a single leg. That is the behaviour we
    # want when legs disagree because the query is ambiguous.
    rrf_k: int = 60
    leg_size_bm25: int = 200
    leg_size_exact: int = 50
    leg_size_dense: int = 150
    leg_size_parent: int = 50
    fusion_top_k: int = 50
    knn_ef_search: int = 128

    # Top-24 rather than top-50: recall@50 is fusion's metric, and RRF rarely buries a true
    # positive below rank 24, so this costs <1 nDCG@10 point and buys ~2x the latency budget.
    rerank_top_n: int = 24
    rerank_max_tokens: int = 288  # attention is quadratic; chunks are 200-400 tokens anyway
    rerank_timeout_ms: int = 400  # on timeout we fall back to fusion order, never to an error

    # Bounded priors applied *after* fusion so each leg stays independently measurable.
    prior_authority_weight: float = 0.15
    prior_recency_weight: float = 0.10
    superseded_penalty: float = 0.25

    # --- ingestion --------------------------------------------------------------------
    chunk_target_tokens: int = 300
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 80
    parent_max_tokens: int = 1800
    # Zero overlap is deliberate: parent expansion recovers boundary context properly, and no
    # overlap keeps near-duplicate detection and token accounting honest.
    chunk_overlap_tokens: int = 0
    max_upload_bytes: int = 200 * 1024 * 1024
    # fp16 vectors in PG cost ~2 KB/chunk and turn a mapping-change rebuild from "three days and
    # your embedding quota" into "tonight".
    persist_vectors: bool = True
    embedding_cache_scope: Literal["global", "tenant"] = "global"
    contextualize_batch_size: int = 20
    contextualize_max_tokens_per_doc: int = 200_000

    # --- answering --------------------------------------------------------------------
    context_token_budget: int = 6000
    max_parents_in_context: int = 8
    max_parents_per_document: int = 3
    dedup_simhash_hamming: int = 6
    dedup_jaccard: float = 0.85

    # --- workers ----------------------------------------------------------------------
    worker_poll_interval_s: float = 2.0
    worker_batch_size: int = 4
    worker_max_attempts: int = 5
    # One tenant's 200k-document backfill must not starve everyone else. This is the single most
    # common multi-tenant ingestion failure, so the cap ships in v1 rather than being added later.
    max_inflight_jobs_per_tenant: int = 8

    @property
    def is_production(self) -> bool:
        return self.environment in ("staging", "production")

    @model_validator(mode="after")
    def _validate_production(self) -> Self:
        if not self.is_production:
            return self
        problems: list[str] = []
        if len(self.jwt_secret.get_secret_value()) < 32:
            problems.append("JWT_SECRET must be at least 32 characters")
        if not self.cookie_secure:
            problems.append("COOKIE_SECURE must be true")
        if "*" in self.cors_origins:
            problems.append("CORS_ORIGINS must not contain a wildcard")
        if self.opensearch_url.startswith("http://"):
            problems.append("OPENSEARCH_URL must use TLS")
        if self.redis_url is None:
            problems.append("REDIS_URL is required (in-process rate limiting cannot span replicas)")
        if problems:
            raise ValueError("Invalid production configuration: " + "; ".join(problems))
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Tests only."""
    global _settings
    _settings = None

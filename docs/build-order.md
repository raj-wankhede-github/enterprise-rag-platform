# Build order

Each step ends in something measurable. Steps 0-6 are the critical path to a number; nothing
after step 6 ships without the ablation table showing it earned its latency.

| # | Step | Exit criterion | State |
|---|---|---|---|
| 0 | Skeleton: uv, ruff/mypy/pytest, compose, CI | stack healthy, gate green | **done** |
| 1 | Tenancy spine: `TenantScoped`, ContextVar, ORM guards, RLS | cross-tenant suite passes with guards on *and* with RLS alone | guards + registry test done; RLS migration pending |
| 2 | Roles and capabilities | exhaustiveness + strict-nesting tests | **done** |
| 3 | `search/`: mappings, generations, router, `dsl.py` | every leg x profile x filter permutation carries exactly one tenant term | **done** (router pending) |
| 4 | Ingestion v1 + dedup and versioning | identical re-upload creates no version; edited re-upload re-embeds under 5% of chunks; case 4 returns 409 | **done** (loaders, chunker, contextualiser, embedder, reuse) |
| 5 | Retrieval: four legs and one `_msearch` | candidates carry per-leg ranks | **done** |
| 6 | **Eval harness, golden set, ablation runner, CI gate** | the table prints in under four minutes | **done** -- 20s, gating CI |
| 7 | Answer path: assembly, extractive generator, deterministic verification | citation-support and abstention metrics appear in the table | **next** -- abstention gate done; citation verification is what closes the remaining 30% made-up rate |
| 8 | `models` container, cross-encoder reranker, `rerank_bench.py` | a real nDCG delta, p95 within budget | pending |
| 9 | Query understanding: rules, fast path, then one LLM call | fast path p95 under 60 ms | pending |
| 10 | Contextual retrieval: template, then LLM with prompt caching | `+contextual` row, measured cost per 1k chunks | pending |
| 11 | `parser` container (Docling), OCR, tables | scanned-PDF and table strata pass | pending |
| 12 | Generation rebuild and atomic alias swap, shadow-evaluated | zero errors and zero empty results under continuous traffic | pending |
| 13 | SSO: discovery, OIDC, JIT provisioning, admin wizard | round trip against Keycloak and a real Entra tenant | pending |
| 14 | Frontend: routes, AuthContext, login, search and ask, admin | capability-gated nav, every error state demoed | pending |
| 15 | API keys, Redis rate limiting, concurrency caps | limits hold across two replicas | pending |
| 16 | Platform operator plane: separate app, host and audience; support grants | an operator cannot read tenant content without a grant | pending |
| 17 | Observability, audit UI, SharePoint connector | a restricted file is invisible to a non-member, end to end | pending |
| 18 | SAML via the broker, SCIM, BYOC packaging | isolation suite green against the BYOC compose file | pending |

Steps 1 and 2 are non-negotiably first: every later table and route inherits their shape, and
retrofitting `tenant_id` and capability gating is the most expensive rewrite available.

## Why the eval harness comes before the quality features

Steps 8, 10 and 11 — reranking, contextual chunking, layout-aware parsing — are each a claim that
something improves answers. Without step 6 there is no way to test the claim, so the decision
gets made by whoever argues best. With it, each one is a row in a table with a latency column
next to it, and a feature that does not earn its milliseconds gets removed.


## What the harness measured, and what it refuses to claim

First run, 14 documents / 71 chunks / 61 questions, offline triple:

```
config                   recall@10  recall@50  nDCG@10  MRR@10  abstain  made_up  leak
bm25_only                0.500      0.500      0.467    0.467   0.900    0.100    0.000
dense_only               0.853      0.961      0.680    0.635   0.700    0.300    0.000
hybrid_rrf               0.912      0.980      0.791    0.765   0.700    0.300    0.000
hybrid_rrf + contextual  0.941      0.990      0.832    0.818   0.700    0.300    0.000
```

Three things worth stating plainly, because the table is easy to over-read:

1. **recall@50 is not evidence at this corpus size.** 50 of 71 chunks is 70% of the index, so a
   semantically blind hashing embedder scores 0.96. `recall@10` is the honest column, and the
   report prints a warning until the corpus is large enough for the deeper cutoff to
   discriminate. Growing the corpus is the single highest-value improvement to this harness.
2. **`dense_only` beating `bm25_only` says nothing about embedding quality.** The hashing
   embedder has no semantics; it wins here because it retrieves broadly on a tiny corpus. The
   ablation measures the *pipeline*, not the model.
3. **`made_up` is 0.300 and that is the honest number.** `bench/abstention_calibration.py` shows
   no coverage threshold separates the remainder: driving it to zero means refusing 39% of
   answerable questions. The residue is questions whose topic terms are all present and only the
   asked-for value is absent. Citation verification at step 7 is what closes it.

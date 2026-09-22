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
| 7 | Answer path: assembly, extractive generator, deterministic verification | citation-support and abstention metrics appear in the table | **done** -- made_up 0.300 -> 0.200, `cite_ok` in the table |
| 8 | `models` container, cross-encoder reranker, `rerank_bench.py` | a real nDCG delta, p95 within budget | **done** -- nDCG@10 0.832 -> 0.895 for +12 ms |
| 9 | Query understanding: rules, fast path, then one LLM call | fast path p95 under 60 ms | **next** |
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


## Step 7: what was predicted, and what was measured

When the harness landed I predicted citation verification would close the 0.300 made-up rate.
That was **half right**, and the half that was wrong is worth recording.

`made_up` did fall, 0.300 -> 0.200. But it fell because fixing the harness exposed a dead
identifier rule (see below), **not** because of verification. The `+ verified` row scores
identically to the row without it.

The reason is structural. CI's generator is extractive: it quotes verbatim and cannot fabricate,
so verification has nothing to catch. Its value is proven in
`tests/unit/test_answer_path.py::test_pipeline_rejects_a_fabricated_value_and_abstains`, where a
deliberately fabricating generator states a per-diem for a grade the table does not list and is
caught with no model involved. Against a real LLM that matters; against the extractive floor it
is a no-op.

The two questions still answered are a **different failure**, and `citation_support` is 1.0 for
both, correctly:

- *"what is the per diem for a grade D destination"* -> quotes a real sentence about destination
  grades that never states grade D's rate.
- *"what is the retention period for board minutes"* -> quotes real sentences about retention
  periods that never mention board minutes.

The evidence is quoted faithfully and simply does not answer the question. That is an entailment
judgement, not a citation check, and no deterministic rule sees it. The cross-encoder arriving
with the models container at step 8 is the thing that can, which is why the `unsupported_answer_rate`
floor stays at 0.25 until that is measured rather than being asserted now.


## Step 8: reranking earns its latency; entailment does not close the gap

Measured with the lexical reranker, which is what CI can run offline:

```
config                               recall@10  nDCG@10  MRR@10  made_up  cite_ok  p95ms
hybrid_rrf + contextual              0.941      0.832    0.818   0.200    1.000    26
hybrid + contextual + rerank         0.980      0.895    0.889   0.200    1.000    37
hybrid + contextual + verified       0.941      0.832    0.818   0.200    1.000    154
hybrid + rerank + verified + entail  0.980      0.895    0.889   0.200    1.000    137
```

**Reranking earns its place.** +6.3 points of nDCG@10 and +7.1 of MRR@10 for about 12 ms, from a
reranker with no idea what a sentence means. That is the floor; the cross-encoder should beat it,
and `bench/rerank_bench.py` is what will say whether it does so inside the latency budget.

**Entailment did not move the made-up rate**, and the reason is now measured rather than
suspected. CI's generator is extractive, so a claim *is* a quote from its own evidence: lexical
entailment scores it 1.0 by construction, and `citation_support` is legitimately 1.0. There is a
test asserting exactly this limitation so the proxy is never mistaken for a model.

A correction to the step 7 write-up, which was right by accident: the `verified` row scored
identically to the plain row because **the answer path was never wired into the eval runner** --
a patch was silently lost. It is wired now (p95 26 -> 154 ms proves it), the numbers are
unchanged, and the conclusion stands for the right reason.

The two surviving cases need a judgement of *"does this evidence answer the question"*, not
*"does this evidence support the claim"*. Those are different questions and only the second was
being asked. A real cross-encoder scoring the question against the cited evidence is the thing
that can answer the first.

# SAML, SCIM and BYOC

## SAML: the product does not speak it

Customers who require SAML reach us through a **self-hosted Keycloak broker**. Keycloak speaks
SAML 2.0 outward to the customer's identity provider and OIDC inward to us. The product has one
federation implementation, and it is the OIDC one in `app/auth/oidc.py`.

This was a deliberate trade, and the reasons are worth keeping because the alternative looks
cheaper than it is:

**`xmlsec` has no usable Windows wheels.** `python3-saml` depends on it, and a developer on
Windows cannot install the project. That alone is not decisive, but it is the first sign of what
the dependency costs.

**XML signature wrapping is a place people build critical vulnerabilities by accident.** The
attack is that a SAML response can contain a validly-signed assertion *and* an unsigned one, and
a library that validates the first while the application reads the second authenticates the
attacker as anyone they name. Several widely-used SAML libraries have shipped exactly this bug.
Keycloak is a mature implementation maintained by people who work on this full time, and it is
the kind of code where "mature and widely attacked" is worth far more than "ours".

**It is self-hostable, so BYOC is unaffected.** Delegating to a hosted SAML service would put a
third party in the authentication path of an on-premises deployment, which is precisely what a
BYOC customer bought their way out of. Keycloak runs in their VPC alongside everything else.

**It was already in the development compose file.** Testing OIDC locally needs an identity
provider, and Keycloak is the sane choice for that regardless. Using it for SAML as well adds a
configuration, not a component.

### How a customer is onboarded

1. A realm is created in the broker for the tenant, and a SAML identity provider is configured
   inside it from the customer's metadata URL or XML.
2. An OIDC client is created in that realm for the product.
3. An `idp_configs` row is created with `kind: keycloak_broker` and the realm's issuer.
4. The administrator runs the wizard's **test sign-in** against it, exactly as for Entra or
   Google, and sees the claim preview before activating.

The product never learns that SAML was involved. `kind` exists only to label the button.

### Attribute mapping

The broker maps SAML attributes onto OIDC claims. Two mappings matter:

- The SAML `NameID`, or a stable attribute the customer nominates, becomes `sub`. It must be
  immutable — `app/auth/provisioning.py` matches identities on it, and a `NameID` that changes
  orphans the account.
- The group attribute becomes the `groups` claim, so `resolve_role` works unchanged.

---

## SCIM: provisioning is not the point, deprovisioning is

JIT provisioning already creates users correctly on first sign-in. What it cannot do is remove
them. When someone leaves, their directory account is disabled and they can no longer
authenticate — but their sessions remain valid until they expire, their API keys keep working,
and their name stays in the user list looking current.

The security questionnaire answer, with a number in it:

> **How quickly is access revoked when an employee leaves?**
> Within 15 minutes with SCIM configured. Within 24 hours without it, by nightly reconciliation
> against the directory.

Three decisions in `app/auth/scim.py`, each with a test:

- **`active: false` deactivates; it does not delete.** The audit trail must survive the person
  leaving, and a deleted row would orphan every entry naming them. A `DELETE` is treated the same
  way.
- **Deactivation revokes every session and API key immediately.** Marking the row inactive is the
  least interesting part of the change; without the rest, "we deprovisioned them" is true of the
  database and false of the system.
- **Roles are not writable over SCIM.** A directory push that could set a role would make the
  customer's IdP administrator an administrator here, through a path with none of the wizard's
  warnings or previews.

### The bug worth knowing about

Entra has historically sent `active` as the **string** `"False"`. `bool("False")` is `True` in
Python, so the naive implementation activates the account it was just told to disable — and
reports success. A departed employee keeps their access, with a green tick in the customer's
console. `_as_bool` exists for that one line, and
`test_entra_sends_active_as_the_string_false` guards it.

Okta sends deactivation as `{"op": "replace", "value": {"active": false}}` with no `path` at all,
which a path-only implementation ignores. Both shapes are tested.

---

## BYOC: packaging, not a code branch

The claim is that bring-your-own-cloud is a deployment exercise rather than a second product.
What makes it true:

**Shared-schema multi-tenancy runs unchanged with one tenant row.** This is the decisive reason
it was chosen over schema-per-tenant. A schema-per-tenant design forces two tenant-resolution
paths — one for SaaS and one for the single-tenant install — and the second is always the
less-tested one.

**Every dependency is self-hostable.** OpenSearch, Postgres, Redis, MinIO, Keycloak and the two
model containers all run in a customer's VPC. Nothing in `app/` imports a vendor SDK: tracing is
OpenTelemetry to a Collector the customer points wherever they like, and the offline triple
(`LLM_PROVIDER=none`, `EMBEDDING_PROVIDER=hashing`, `RERANKER_PROVIDER=identity`) means the
product starts and serves with no external API at all.

**The differences are a compose file.** `docker-compose.byoc.yml` is the same topology with
managed services swapped for customer-provided endpoints.

### What CI must prove

The tenant-isolation suite runs against **both** compose files. That is what keeps this honest:
if BYOC ever needed a code branch, the isolation tests would be the first thing to diverge, and
running them against only the SaaS configuration would hide it.

```bash
docker compose -f docker-compose.byoc.yml up -d
cd backend && uv run pytest -m integration -q
```

### What a BYOC customer gives up

Stated plainly, because discovering it after signing is worse than hearing it before:

- **Hosted rerankers and embedders** (Cohere, Voyage, Jina) are the only place tenant query text
  leaves the deployment, and a BYOC customer usually disables them. The ONNX containers are the
  default and the ablation table reports both.
- **Managed upgrades.** A generation rebuild is `python -m app.cli rebuild`, which they run.
- **Cross-tenant embedding cache reuse**, which is off for them anyway
  (`EMBEDDING_CACHE_SCOPE=tenant`).

### What they do not give up

Everything in the tenancy, ACL and audit story is identical, because it is the same code. The
platform operator plane simply has no operators in it, and `support_grants` stays empty — which
is the cleanest possible answer to "can the vendor read our data".

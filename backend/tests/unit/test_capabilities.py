"""Invariants of the four-role chain.

The product owner fixed a flat model: ADMIN > DEV > TEST > PROD, one role per user, descending
rights. These tests are the tripwire that keeps it that way -- particularly the nesting one,
which fails the moment someone adds a capability that only makes sense for a fifth role.
"""

from __future__ import annotations

import itertools

import pytest

from app.security.capabilities import (
    RANK,
    ROLE_CAPABILITIES,
    ROLE_CHAIN,
    Capability,
    Role,
    may_see,
)


def test_every_role_has_a_capability_set() -> None:
    assert set(ROLE_CAPABILITIES) == set(Role)


def test_no_orphan_capabilities() -> None:
    granted: set[Capability] = set()
    for caps in ROLE_CAPABILITIES.values():
        granted |= caps
    assert granted == set(Capability), f"never granted to any role: {set(Capability) - granted}"


def test_roles_form_a_strict_chain() -> None:
    """Each rank must be a proper superset of the one below it."""
    for lower, higher in itertools.pairwise(ROLE_CHAIN):
        low, high = ROLE_CAPABILITIES[lower], ROLE_CAPABILITIES[higher]
        assert low < high, f"{higher} must strictly contain {lower}; missing {low - high}"


def test_ranks_are_ordered_and_spaced() -> None:
    ranks = [RANK[role] for role in ROLE_CHAIN]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)
    # Spaced so an intermediate level is a data change, not a migration of stored integers.
    assert all(b - a >= 10 for a, b in itertools.pairwise(ranks))


@pytest.mark.parametrize(
    ("role", "capability", "expected"),
    [
        (Role.PROD, Capability.SEARCH, True),
        (Role.PROD, Capability.DOC_UPLOAD, False),
        (Role.PROD, Capability.TRACE_VIEW_ALL, False),
        (Role.TEST, Capability.EVAL_RUN, True),
        (Role.TEST, Capability.DOC_UPLOAD, False),
        (Role.TEST, Capability.EVAL_MANAGE_DATASET, False),
        (Role.TEST, Capability.EXPORT_DOCUMENTS, False),
        (Role.DEV, Capability.DOC_UPLOAD, True),
        (Role.DEV, Capability.DOC_SET_VISIBILITY, True),
        (Role.DEV, Capability.DOC_SET_VISIBILITY_ANY, False),
        (Role.DEV, Capability.USER_MANAGE, False),
        (Role.DEV, Capability.DOC_DELETE_ANY, False),
        (Role.ADMIN, Capability.USER_MANAGE, True),
        (Role.ADMIN, Capability.SSO_CONFIGURE, True),
        (Role.ADMIN, Capability.AUDIT_VIEW, True),
    ],
)
def test_matrix_spot_checks(role: Role, capability: Capability, expected: bool) -> None:
    assert (capability in ROLE_CAPABILITIES[role]) is expected


def test_admin_only_capabilities_are_admin_only() -> None:
    admin_only = ROLE_CAPABILITIES[Role.ADMIN] - ROLE_CAPABILITIES[Role.DEV]
    for role in (Role.PROD, Role.TEST, Role.DEV):
        assert not (ROLE_CAPABILITIES[role] & admin_only)


def test_visibility_is_at_or_below_rank() -> None:
    prod, test, dev, admin = (RANK[r] for r in ROLE_CHAIN)
    # A PROD user sees only PROD-level documents.
    assert may_see(prod, prod)
    assert not may_see(prod, test)
    assert not may_see(prod, admin)
    # An ADMIN sees everything.
    assert all(may_see(admin, level) for level in (prod, test, dev, admin))
    # DEV sees up to its own level, not above.
    assert may_see(dev, test)
    assert not may_see(dev, admin)


def test_platform_role_is_not_representable() -> None:
    """The vendor's identity must never be expressible as a tenant role."""
    assert "PLATFORM" not in {role.value for role in Role}

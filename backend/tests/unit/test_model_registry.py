"""Architecture tests: tenancy by default, not by remembering.

A new table must be tenant-scoped unless someone deliberately adds it to ``GLOBAL_TABLES``. That
inversion is the point -- a developer who forgets fails the build rather than shipping a table
that any tenant can read.
"""

from __future__ import annotations

import app.models  # noqa: F401  -- import for the side effect of registering every mapper
from app.db.base import GLOBAL_TABLES, Base, TenantScoped


def _mapped_classes() -> dict[str, type]:
    return {
        mapper.class_.__tablename__: mapper.class_
        for mapper in Base.registry.mappers
        if hasattr(mapper.class_, "__tablename__")
    }


def test_every_table_is_tenant_scoped_or_explicitly_global() -> None:
    offenders = [
        table
        for table, cls in _mapped_classes().items()
        if table not in GLOBAL_TABLES and not issubclass(cls, TenantScoped)
    ]
    assert not offenders, (
        f"These tables are neither TenantScoped nor listed in GLOBAL_TABLES: {sorted(offenders)}. "
        "Add the mixin, or add the table to GLOBAL_TABLES with a comment saying why."
    )


def test_tenant_scoped_tables_index_tenant_id_first() -> None:
    """A tenant filter that cannot use an index is a filter that will be dropped under load."""
    missing: list[str] = []
    for table, cls in _mapped_classes().items():
        if not issubclass(cls, TenantScoped):
            continue
        column = cls.__table__.c["tenant_id"]
        indexed = column.index or any(
            index.columns.values() and index.columns.values()[0].name == "tenant_id" for index in cls.__table__.indexes
        )
        if not indexed:
            missing.append(table)
    assert not missing, f"tenant_id is not the leading indexed column on: {sorted(missing)}"


def test_global_tables_really_have_no_tenant_column() -> None:
    """Catches a table added to GLOBAL_TABLES by mistake."""
    classes = _mapped_classes()
    for table in GLOBAL_TABLES:
        cls = classes.get(table)
        if cls is None:  # alembic_version and not-yet-written tables
            continue
        assert not issubclass(cls, TenantScoped), f"{table} is listed as global but is TenantScoped"


def test_user_role_is_constrained_to_the_four_tenant_roles() -> None:
    """The platform operator must not be representable as a tenant role."""
    user = _mapped_classes()["users"]
    checks = [str(c.sqltext) for c in user.__table__.constraints if c.__class__.__name__ == "CheckConstraint"]
    role_check = next(c for c in checks if "role" in c)
    for role in ("ADMIN", "DEV", "TEST", "PROD"):
        assert role in role_check
    assert "PLATFORM" not in role_check


def test_users_are_unique_per_tenant_not_globally() -> None:
    """Consultants hold accounts in several tenants; a global unique email is unmigratable."""
    user = _mapped_classes()["users"]
    uniques = [
        {col.name for col in c.columns}
        for c in user.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    assert {"tenant_id", "email"} in uniques
    assert {"email"} not in uniques


def test_documents_are_identified_by_source_and_external_id_not_by_content() -> None:
    """The same bytes may legitimately be two documents with different ACLs."""
    document = _mapped_classes()["documents"]
    uniques = [
        {col.name for col in c.columns}
        for c in document.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    assert {"tenant_id", "source_system", "external_id"} in uniques
    assert not any("blob_sha256" in u for u in uniques)


def test_no_source_file_contains_a_stray_control_character() -> None:
    """Guards a corruption that has already happened twice in this repo.

    Writing source through a shell heredoc turns an intended ``\b`` inside a raw string into a
    literal backspace (0x08). The file still parses, the regex still compiles, and it silently
    matches nothing -- so an identifier rule and a number-normalisation rule both shipped dead
    and were only found by a failing behaviour test much later.
    """
    import pathlib

    forbidden = {0x07: "BELL", 0x08: "BACKSPACE", 0x0B: "VTAB", 0x0C: "FORMFEED", 0x1B: "ESC"}
    offenders: list[str] = []
    roots = [pathlib.Path("app"), pathlib.Path("tests"), pathlib.Path("bench")]
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            raw = path.read_bytes()
            for code, name in forbidden.items():
                if bytes([code]) in raw:
                    line = raw[: raw.index(bytes([code]))].count(b"\n") + 1
                    offenders.append(f"{path}:{line} contains {name} (0x{code:02X})")
    assert not offenders, "stray control characters in source:\n  " + "\n  ".join(offenders)

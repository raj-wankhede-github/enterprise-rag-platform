"""Emit the frontend's capability table from the backend enum.

The frontend gates navigation and buttons on capabilities, and it must not maintain its own
list. Drift shows up as buttons that render and then 403 -- which reads to a user as the product
being broken rather than as them lacking permission.

A test asserts the checked-in file equals this output, so drift fails the build rather than
reaching a user. The frontend copy is a *rendering* convenience; authorisation is always the
server's decision, re-made on every request.
"""

from __future__ import annotations

from app.security.capabilities import RANK, ROLE_CAPABILITIES, Capability, Role

HEADER = """/**
 * Generated from backend/app/security/capabilities.py. Do not edit by hand.
 *
 * Regenerate with: uv run python -m app.cli capabilities
 *
 * A test asserts this file equals the backend enum. Drift would otherwise show up as buttons
 * that render and then 403 -- which reads to a user as the product being broken rather than as
 * them lacking permission.
 */
"""


def render() -> str:
    lines = [HEADER, "export const CAPABILITIES = ["]
    lines.extend(f'  "{capability.value}",' for capability in sorted(Capability))
    lines += [
        "] as const;",
        "",
        "export type Capability = (typeof CAPABILITIES)[number];",
        "",
        "export const ROLES = [",
    ]

    ordered = sorted(Role, key=lambda role: RANK[role])
    lines.extend(f'  "{role.value}",' for role in ordered)
    lines += [
        "] as const;",
        "",
        "export type Role = (typeof ROLES)[number];",
        "",
        "/** Spaced by 10 so an intermediate rank could be added without renumbering documents. */",
        "export const RANK: Record<Role, number> = {",
    ]
    lines.extend(f"  {role.value}: {RANK[role]}," for role in ordered)
    lines += [
        "};",
        "",
        "/** The full matrix, for the admin UI. Authorisation is always the server's decision. */",
        "export const ROLE_CAPABILITIES: Record<Role, readonly Capability[]> = {",
    ]
    for role in ordered:
        rendered = ", ".join(f'"{capability}"' for capability in sorted(str(c) for c in ROLE_CAPABILITIES[role]))
        lines.append(f"  {role.value}: [{rendered}],")
    lines += ["};", ""]
    return "\n".join(lines)

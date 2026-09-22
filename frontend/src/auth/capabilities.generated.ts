/**
 * Generated from backend/app/security/capabilities.py. Do not edit by hand.
 *
 * Regenerate with: uv run python -m app.cli capabilities
 *
 * A test asserts this file equals the backend enum. Drift would otherwise show up as buttons
 * that render and then 403 -- which reads to a user as the product being broken rather than as
 * them lacking permission.
 */

export const CAPABILITIES = [
  "apikey:manage_any",
  "apikey:manage_own",
  "ask",
  "audit:view",
  "collection:manage",
  "connector:manage",
  "doc:delete_any",
  "doc:delete_own",
  "doc:edit_metadata",
  "doc:read",
  "doc:reindex",
  "doc:set_visibility",
  "doc:set_visibility_any",
  "doc:upload",
  "eval:manage_dataset",
  "eval:run",
  "export:documents",
  "export:eval",
  "retrieval:read_config",
  "retrieval:tune",
  "search",
  "sso:configure",
  "tenant:settings",
  "trace:view_all",
  "trace:view_own",
  "user:manage",
] as const;

export type Capability = (typeof CAPABILITIES)[number];

export const ROLES = [
  "PROD",
  "TEST",
  "DEV",
  "ADMIN",
] as const;

export type Role = (typeof ROLES)[number];

/** Spaced by 10 so an intermediate rank could be added without renumbering documents. */
export const RANK: Record<Role, number> = {
  PROD: 10,
  TEST: 20,
  DEV: 30,
  ADMIN: 40,
};

/** The full matrix, for the admin UI. Authorisation is always the server's decision. */
export const ROLE_CAPABILITIES: Record<Role, readonly Capability[]> = {
  PROD: ["ask", "doc:read", "search", "trace:view_own"],
  TEST: ["ask", "doc:read", "eval:run", "export:eval", "retrieval:read_config", "search", "trace:view_all", "trace:view_own"],
  DEV: ["apikey:manage_own", "ask", "doc:delete_own", "doc:edit_metadata", "doc:read", "doc:reindex", "doc:set_visibility", "doc:upload", "eval:manage_dataset", "eval:run", "export:eval", "retrieval:read_config", "retrieval:tune", "search", "trace:view_all", "trace:view_own"],
  ADMIN: ["apikey:manage_any", "apikey:manage_own", "ask", "audit:view", "collection:manage", "connector:manage", "doc:delete_any", "doc:delete_own", "doc:edit_metadata", "doc:read", "doc:reindex", "doc:set_visibility", "doc:set_visibility_any", "doc:upload", "eval:manage_dataset", "eval:run", "export:documents", "export:eval", "retrieval:read_config", "retrieval:tune", "search", "sso:configure", "tenant:settings", "trace:view_all", "trace:view_own", "user:manage"],
};

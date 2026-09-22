/**
 * The sign-in configuration wizard: draft -> test -> activate.
 *
 * The order is the product, not the UI. Activating an untested configuration on a tenant with
 * password login disabled locks out everyone including the administrator who did it, and
 * recovery then needs a platform operator. So a configuration cannot be activated until a real
 * sign-in has been completed against it, and the claim preview from that sign-in is shown before
 * the activate button is enabled.
 *
 * The preview is what makes it more than ceremony. It names which directory groups arrived,
 * which of them matched a mapping, which were ignored, and what role the person signing in would
 * have received. An administrator who sees that "All Company" maps to ADMIN will not activate it
 * -- and that is a mistake that otherwise hands every employee full administrative rights
 * silently, at their next login.
 */

import { useEffect, useState } from "react";
import { api, ApiError } from "@/api/client";
import type { Role } from "@/auth/capabilities.generated";

type IdpState = "draft" | "tested" | "active" | "disabled";

type IdpConfig = {
  id: string;
  display_name: string;
  kind: string;
  state: IdpState;
  issuer: string;
  client_id: string;
  default_role: Role;
  jit_provisioning: boolean;
  sync_role_on_login: boolean;
  role_mappings: Record<string, Role>;
  tested_at: string | null;
  last_test_claims: Record<string, unknown>;
};

type ClaimPreview = {
  subject: string;
  email: string | null;
  email_verified: boolean;
  groups_seen: string[];
  groups_matched: Record<string, string>;
  groups_ignored: string[];
  resolved_role: string;
  would_elevate_to_admin: boolean;
  default_role_for_new_users: string;
  warnings: string[];
};

export default function Sso() {
  const [configs, setConfigs] = useState<IdpConfig[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<Record<string, ClaimPreview>>({});

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ configs: IdpConfig[] }>("/api/admin/sso", controller.signal)
      .then((payload) => setConfigs(payload.configs))
      .catch((caught) => {
        if (!controller.signal.aborted) {
          setError(caught instanceof ApiError ? caught.message : "We could not load sign-in settings.");
        }
      });
    return () => controller.abort();
  }, []);

  function runTest(config: IdpConfig) {
    // A real sign-in in a popup, against the same callback the live flow uses. A separate test
    // endpoint would mean the thing being validated is not the thing users will use, and that
    // difference is exactly where a configuration bug hides.
    const popup = window.open(`/api/auth/oidc/${config.id}/test`, "sso-test", "width=520,height=680");
    if (!popup) {
      setError("Your browser blocked the sign-in window. Allow pop-ups for this site and try again.");
      return;
    }

    const poll = window.setInterval(async () => {
      if (!popup.closed) return;
      window.clearInterval(poll);
      try {
        const result = await api.get<ClaimPreview>(`/api/admin/sso/${config.id}/last-test`);
        setPreview((current) => ({ ...current, [config.id]: result }));
        setConfigs((current) =>
          current?.map((row) => (row.id === config.id ? { ...row, state: "tested" as IdpState } : row)) ?? null,
        );
      } catch {
        setError("The test sign-in did not complete. Nothing has been changed.");
      }
    }, 500);
  }

  async function activate(config: IdpConfig) {
    setError(null);
    try {
      const updated = await api.post<IdpConfig>(`/api/admin/sso/${config.id}/activate`);
      setConfigs((current) => current?.map((row) => (row.id === config.id ? updated : row)) ?? null);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "We could not activate that configuration.");
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-semibold">Sign-in</h1>

      {error && (
        <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900">
          {error}
        </div>
      )}

      {configs?.length === 0 && (
        <p className="text-sm text-slate-600">
          No identity provider is configured. Users sign in with a password.
        </p>
      )}

      {configs?.map((config) => (
        <section key={config.id} className="rounded-lg border border-slate-200 bg-white p-5">
          <div className="mb-3 flex items-center gap-3">
            <h2 className="font-medium">{config.display_name || config.kind}</h2>
            <StateBadge state={config.state} />
            <div className="ml-auto flex gap-2">
              <button
                type="button"
                onClick={() => runTest(config)}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium hover:bg-slate-50"
              >
                Test sign-in
              </button>
              <button
                type="button"
                disabled={config.state === "draft" || config.state === "active"}
                onClick={() => void activate(config)}
                title={
                  config.state === "draft"
                    ? "Complete a test sign-in first. Activating an untested configuration can lock everyone out."
                    : undefined
                }
                className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-40"
              >
                {config.state === "active" ? "Active" : "Activate"}
              </button>
            </div>
          </div>

          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-slate-500">Issuer</dt>
            <dd className="font-mono text-xs break-all">{config.issuer}</dd>
            <dt className="text-slate-500">Client ID</dt>
            <dd className="font-mono text-xs break-all">{config.client_id}</dd>
            <dt className="text-slate-500">New users get</dt>
            <dd>
              {config.jit_provisioning ? config.default_role : "no account — an administrator must add them first"}
            </dd>
          </dl>

          {preview[config.id] && <Preview preview={preview[config.id]!} />}
        </section>
      ))}
    </div>
  );
}

function StateBadge({ state }: { state: IdpState }) {
  const styles: Record<IdpState, string> = {
    draft: "bg-slate-100 text-slate-600",
    tested: "bg-blue-100 text-blue-800",
    active: "bg-green-100 text-green-800",
    disabled: "bg-slate-100 text-slate-400",
  };
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${styles[state]}`}>{state}</span>
  );
}

/**
 * What the last test sign-in actually produced.
 *
 * The elevation warning is the single most valuable thing on this page, so it is rendered first
 * and loudly. Everything else here exists to answer "why did this person get that role", which
 * is otherwise guesswork against a directory the product cannot see.
 */
function Preview({ preview }: { preview: ClaimPreview }) {
  return (
    <div className="mt-4 border-t border-slate-100 pt-4">
      <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
        Last test sign-in
      </h3>

      {preview.would_elevate_to_admin && (
        <div
          role="alert"
          className="mb-3 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900"
        >
          <strong className="font-semibold">This grants ADMIN through a directory group.</strong>{" "}
          Everyone in that group will hold full administrative rights, including user management
          and these sign-in settings. Check the group is the one you intend before activating.
        </div>
      )}

      {preview.warnings
        .filter((warning) => !warning.includes("ADMIN"))
        .map((warning) => (
          <p key={warning} className="mb-2 rounded-md bg-amber-50 p-2.5 text-sm text-amber-900">
            {warning}
          </p>
        ))}

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-slate-500">Signed in as</dt>
        <dd>
          {preview.email ?? "(no email claim)"}
          {!preview.email_verified && (
            <span className="ml-2 text-xs text-amber-700">address not marked verified</span>
          )}
        </dd>
        <dt className="text-slate-500">Would receive</dt>
        <dd className="font-medium">{preview.resolved_role}</dd>
        <dt className="text-slate-500">Groups matched</dt>
        <dd>
          {Object.keys(preview.groups_matched).length === 0 ? (
            <span className="text-slate-400">none</span>
          ) : (
            Object.entries(preview.groups_matched).map(([group, role]) => (
              <div key={group} className="font-mono text-xs">
                {group} &rarr; {role}
              </div>
            ))
          )}
        </dd>
        {preview.groups_ignored.length > 0 && (
          <>
            <dt className="text-slate-500">Ignored</dt>
            <dd className="font-mono text-xs text-slate-500">
              {preview.groups_ignored.join(", ")}
            </dd>
          </>
        )}
      </dl>
    </div>
  );
}

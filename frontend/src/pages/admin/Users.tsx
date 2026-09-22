/**
 * User administration.
 *
 * Two things the UI is responsible for making legible, because both are easy to get wrong
 * quietly:
 *
 * **What a role actually grants.** An administrator choosing between DEV and TEST is choosing a
 * capability set, and the matrix is right there rather than in documentation nobody opens.
 *
 * **What "pinned" means.** A role pinned here overrules the directory on every subsequent SSO
 * login. Without saying so, an administrator who pins a role and then wonders why the nightly
 * group sync stopped working has no way to connect the two.
 */

import { useEffect, useState } from "react";
import { api, ApiError } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { RANK, ROLE_CAPABILITIES, type Role } from "@/auth/capabilities.generated";

type UserRow = {
  id: string;
  email: string;
  name: string;
  role: Role;
  role_locked: boolean;
  is_active: boolean;
  last_login_at: string | null;
  identity_providers: string[];
};

export default function Users() {
  const { session } = useAuth();
  const [rows, setRows] = useState<UserRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ users: UserRow[] }>("/api/admin/users", controller.signal)
      .then((payload) => setRows(payload.users))
      .catch((caught) => {
        if (!controller.signal.aborted) {
          setError(caught instanceof ApiError ? caught.message : "We could not load the user list.");
        }
      });
    return () => controller.abort();
  }, []);

  async function changeRole(user: UserRow, role: Role) {
    setSaving(user.id);
    setError(null);
    try {
      // Optimistic, then corrected from the server's answer. A role change that silently did
      // not apply is worse than one that briefly shows the wrong value.
      const updated = await api.patch<UserRow>(`/api/admin/users/${user.id}`, { role });
      setRows((current) => current?.map((row) => (row.id === user.id ? updated : row)) ?? null);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "We could not change that role.");
    } finally {
      setSaving(null);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-semibold">Users</h1>

      {error && (
        <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900">
          {error}
        </div>
      )}

      {rows && (
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2 font-medium">Person</th>
              <th className="py-2 font-medium">Signs in with</th>
              <th className="py-2 font-medium">Role</th>
              <th className="py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className="border-b border-slate-100">
                <td className="py-2.5">
                  <div className="font-medium">{row.name || row.email}</div>
                  <div className="text-xs text-slate-500">{row.email}</div>
                </td>
                <td className="py-2.5 text-slate-600">
                  {row.identity_providers.length > 0 ? row.identity_providers.join(", ") : "Password"}
                </td>
                <td className="py-2.5">
                  <select
                    value={row.role}
                    disabled={saving === row.id || row.id === session?.user_id}
                    onChange={(event) => void changeRole(row, event.target.value as Role)}
                    aria-label={`Role for ${row.email}`}
                    className="rounded-md border border-slate-300 px-2 py-1 text-sm disabled:opacity-60"
                  >
                    {(Object.keys(RANK) as Role[])
                      .sort((a, b) => RANK[b] - RANK[a])
                      .map((role) => (
                        <option key={role} value={role}>
                          {role}
                        </option>
                      ))}
                  </select>
                  {row.role_locked && (
                    <span
                      title="This role is pinned here and will not be changed by directory group membership at the next sign-in."
                      className="ml-2 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600"
                    >
                      pinned
                    </span>
                  )}
                  {row.id === session?.user_id && (
                    /* Changing your own role is how an administrator locks themselves out of
                       the product with no way back except a platform operator. */
                    <span className="ml-2 text-[10px] text-slate-400">your account</span>
                  )}
                </td>
                <td className="py-2.5 text-slate-600">{row.is_active ? "Active" : "Deactivated"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <RoleMatrix />
    </div>
  );
}

/**
 * What each role can do, rendered from the generated table.
 *
 * Generated rather than written out, so it cannot drift from the backend. The four roles are a
 * chain -- PROD subset TEST subset DEV subset ADMIN -- and a test on the backend asserts that
 * strictly, so this table can be read as cumulative.
 */
function RoleMatrix() {
  const roles = (Object.keys(ROLE_CAPABILITIES) as Role[]).sort((a, b) => RANK[a] - RANK[b]);

  return (
    <details className="rounded-md border border-slate-200 bg-white p-4 text-sm">
      <summary className="cursor-pointer font-medium">What each role can do</summary>
      <p className="mt-2 mb-3 text-xs text-slate-500">
        Each role includes everything the one below it can do. A lower role sees more documents,
        not fewer: PROD-level documents are the most widely visible and ADMIN-level the most
        restricted.
      </p>
      <div className="grid gap-4 sm:grid-cols-4">
        {roles.map((role, index) => {
          const previous = index > 0 ? new Set(ROLE_CAPABILITIES[roles[index - 1]!]) : new Set<string>();
          const added = ROLE_CAPABILITIES[role].filter((capability) => !previous.has(capability));
          return (
            <div key={role}>
              <div className="mb-1 font-medium">{role}</div>
              <ul className="space-y-0.5 text-xs text-slate-600">
                {index > 0 && <li className="text-slate-400">everything {roles[index - 1]} can do</li>}
                {added.map((capability) => (
                  <li key={capability}>{capability}</li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </details>
  );
}

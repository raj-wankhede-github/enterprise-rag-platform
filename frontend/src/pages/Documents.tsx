/**
 * The document list.
 *
 * Every row shows who can see it, because in a four-role product the commonest question is "why
 * can my colleague not see this", and the answer belongs here rather than in a support ticket.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "@/api/client";
import { useCan } from "@/auth/AuthContext";
import { RANK, type Role } from "@/auth/capabilities.generated";

type DocumentRow = {
  id: string;
  title: string;
  source_system: string;
  visibility_rank: number;
  version_no: number;
  chunk_count: number;
  is_superseded: boolean;
};

export default function Documents() {
  const [rows, setRows] = useState<DocumentRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canUpload = useCan("doc:upload");

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ documents: DocumentRow[] }>("/api/documents", controller.signal)
      .then((payload) => setRows(payload.documents))
      .catch((caught) => {
        if (!controller.signal.aborted) {
          setError(caught instanceof ApiError ? caught.message : "We could not load your documents.");
        }
      });
    return () => controller.abort();
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center">
        <h1 className="text-lg font-semibold">Documents</h1>
        {canUpload && (
          <Link
            to="/documents/upload"
            className="ml-auto rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800"
          >
            Upload
          </Link>
        )}
      </div>

      {error && (
        <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900">
          {error}
        </div>
      )}

      {rows?.length === 0 && (
        <div className="rounded-md border border-slate-200 bg-white p-5 text-sm text-slate-700">
          No documents yet.
          {canUpload ? " Upload one to get started." : " Ask an administrator to add some."}
        </div>
      )}

      {rows && rows.length > 0 && (
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2 font-medium">Title</th>
              <th className="py-2 font-medium">Source</th>
              <th className="py-2 font-medium">Visible to</th>
              <th className="py-2 font-medium">Version</th>
              <th className="py-2 text-right font-medium">Passages</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className="border-b border-slate-100">
                <td className="py-2.5">
                  <Link to={`/documents/${row.id}`} className="font-medium hover:underline">
                    {row.title}
                  </Link>
                  {row.is_superseded && (
                    <span className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800">
                      superseded
                    </span>
                  )}
                </td>
                <td className="py-2.5 text-slate-600">{row.source_system}</td>
                <td className="py-2.5 text-slate-600">{describeRank(row.visibility_rank)}</td>
                <td className="py-2.5 text-slate-600">v{row.version_no}</td>
                <td className="py-2.5 text-right text-slate-600">{row.chunk_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/**
 * Rank as a sentence, not a number.
 *
 * "visibility_rank: 30" means nothing to an administrator deciding who should see a policy.
 * Naming the lowest role that can see it is the fact they need, and it reads the right way
 * round: a *lower* rank is more widely visible, which is the opposite of most people's first
 * guess.
 */
export function describeRank(rank: number): string {
  const roles = (Object.keys(RANK) as Role[]).filter((role) => RANK[role] >= rank);
  if (roles.length === 0) return "Nobody";
  if (roles.length === Object.keys(RANK).length) return "Everyone";
  const lowest = roles.reduce((a, b) => (RANK[a] <= RANK[b] ? a : b));
  return `${lowest} and above`;
}

/**
 * Upload, with the duplicate outcome made visible.
 *
 * Re-uploading an identical file is the common case -- a retry, a double-click, a connector
 * re-sync -- and it is not an error: nothing is re-parsed, re-embedded or re-indexed. Saying so
 * plainly is what stops the user uploading it a third time to be sure.
 *
 * The 409 is the interesting one. It means the content already exists somewhere the uploader
 * cannot see, and the message deliberately names nothing: not the title, not the owner, not the
 * id. Creating a duplicate instead would be an ACL bypass, because the copy would be visible at
 * the uploader's own, lower rank.
 */

import { useState, type FormEvent } from "react";
import { ApiError, CSRF_COOKIE, CSRF_HEADER, readCookie } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { RANK, type Role } from "@/auth/capabilities.generated";

type UploadResult = {
  document_id: string;
  document_version_id: string;
  deduplicated: boolean;
  version_no: number;
  chunks_embedded: number;
  chunks_reused: number;
};

export default function Upload() {
  const { session } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [visibility, setVisibility] = useState<number>(RANK.PROD);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // A DEV cannot file a document above their own rank. The server enforces that; this keeps the
  // options honest rather than offering a choice that will be refused.
  const ceiling = session ? RANK[session.role] : RANK.PROD;
  const options = (Object.keys(RANK) as Role[]).filter((role) => RANK[role] <= ceiling);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;

    setBusy(true);
    setError(null);
    setResult(null);

    const body = new FormData();
    body.append("file", file);
    body.append("visibility_rank", String(visibility));

    try {
      // FormData, so no Content-Type header: the browser sets the multipart boundary, and
      // overriding it produces a body the server cannot parse.
      const token = readCookie(CSRF_COOKIE);
      const response = await fetch("/api/documents", {
        method: "POST",
        credentials: "same-origin",
        headers: token ? { [CSRF_HEADER]: token } : {},
        body,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new ApiError(
          response.status,
          String(payload.code ?? "error"),
          String(payload.message ?? "We could not upload that file."),
        );
      }
      setResult(payload as UploadResult);
      setFile(null);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "We could not upload that file.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-lg space-y-6">
      <h1 className="text-lg font-semibold">Upload a document</h1>

      <form onSubmit={submit} className="space-y-4">
        <div>
          <label htmlFor="file" className="mb-1.5 block text-sm font-medium text-slate-700">
            File
          </label>
          <input
            id="file"
            type="file"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
            required
          />
          <p className="mt-1 text-xs text-slate-500">
            PDF, Word, PowerPoint, Excel, Markdown, HTML or plain text. Scanned pages and tables
            are read with layout recognition.
          </p>
        </div>

        <div>
          <label htmlFor="visibility" className="mb-1.5 block text-sm font-medium text-slate-700">
            Who can see it
          </label>
          <select
            id="visibility"
            value={visibility}
            onChange={(event) => setVisibility(Number(event.target.value))}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
          >
            {options.map((role) => (
              <option key={role} value={RANK[role]}>
                {role} and above
              </option>
            ))}
          </select>
        </div>

        <button
          type="submit"
          disabled={busy || !file}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-60"
        >
          {busy ? "Uploading…" : "Upload"}
        </button>
      </form>

      {error && (
        <div role="alert" className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          {error}
        </div>
      )}

      {result && (
        <div role="status" className="rounded-md border border-slate-200 bg-white p-4 text-sm">
          {result.deduplicated ? (
            <>
              <p className="mb-1 font-medium text-slate-900">Already uploaded</p>
              <p className="text-slate-700">
                This file is identical to version {result.version_no}, so nothing was
                re-processed. The existing document is unchanged.
              </p>
            </>
          ) : (
            <>
              <p className="mb-1 font-medium text-slate-900">
                Uploaded as version {result.version_no}
              </p>
              <p className="text-slate-700">
                {result.chunks_embedded} passage{result.chunks_embedded === 1 ? "" : "s"} newly
                indexed
                {result.chunks_reused > 0 && `, ${result.chunks_reused} unchanged and reused`}. It
                will appear in search within a few moments.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Keyword and hybrid search over the corpus.
 *
 * The per-leg badges are not decoration: they are what makes a bad result explainable. When
 * someone says "why did this come back", the answer is almost always that one leg matched
 * strongly and the others did not, and showing which leg found a hit turns a complaint into a
 * diagnosis. They are visible to every role because the information is about the *query*, not
 * about anything the user cannot already see.
 */

import { useCallback, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, ApiError, type SearchResponse } from "@/api/client";

export default function Search() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  const run = useCallback(
    async (event: FormEvent) => {
      event.preventDefault();
      const trimmed = query.trim();
      if (!trimmed) return;

      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;

      setBusy(true);
      setError(null);
      try {
        setResult(await api.post<SearchResponse>("/api/search", { query: trimmed }, controller.signal));
      } catch (caught) {
        if (controller.signal.aborted) return;
        setError(caught instanceof ApiError ? caught.message : "We could not reach the server.");
      } finally {
        if (!controller.signal.aborted) setBusy(false);
      }
    },
    [query],
  );

  return (
    <div className="space-y-6">
      <form onSubmit={run} className="flex gap-2">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search documents, policies, tickets…"
          aria-label="Search"
          autoFocus
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-900 focus:ring-1 focus:ring-slate-900"
        />
        <button
          type="submit"
          disabled={busy || !query.trim()}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-60"
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      {error && (
        <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900">
          {error}
        </div>
      )}

      {result && (
        <>
          <p className="text-xs text-slate-500">
            {result.total === 0
              ? "No documents matched."
              : `${result.total} result${result.total === 1 ? "" : "s"} in ${result.took_ms} ms`}
          </p>

          {result.total === 0 && (
            /* Not an error state. An empty result for a permission-scoped corpus is ordinary,
               and the likeliest explanation is worth stating rather than leaving to guesswork. */
            <div className="rounded-md border border-slate-200 bg-white p-5 text-sm text-slate-700">
              <p className="mb-2 font-medium text-slate-900">Nothing matched that search.</p>
              <p>
                Try fewer or broader words. Documents you do not have permission to see are never
                included, so a colleague may get different results for the same search.
              </p>
            </div>
          )}

          <ol className="space-y-3">
            {result.hits.map((hit) => (
              <li key={hit.chunk_id} className="rounded-md border border-slate-200 bg-white p-4">
                <div className="mb-1 flex items-baseline gap-2">
                  <Link to={`/documents/${hit.doc_id}`} className="text-sm font-medium hover:underline">
                    {hit.title}
                  </Link>
                  <span className="ml-auto flex gap-1">
                    {Object.entries(hit.legs).map(([leg, detail]) => (
                      <span
                        key={leg}
                        title={`rank ${detail.rank}, score ${detail.score.toFixed(3)}`}
                        className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600"
                      >
                        {leg}
                      </span>
                    ))}
                  </span>
                </div>
                {hit.heading_path && <p className="mb-2 text-xs text-slate-500">{hit.heading_path}</p>}
                <p className="text-sm leading-relaxed text-slate-700">{hit.snippet}</p>
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}

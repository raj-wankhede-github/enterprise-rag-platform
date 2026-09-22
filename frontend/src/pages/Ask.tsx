/**
 * Ask a question, get a cited answer — or an honest refusal.
 *
 * The refusal is the part that matters here. `answerable: false` is a successful outcome, not an
 * error, and rendering it in a red error box would teach users that the one honest thing the
 * system does is a malfunction. It gets the same visual weight as an answer, with the same
 * "here is what we searched" context, and the sub-questions that *were* answerable are shown
 * rather than discarded.
 *
 * Citations are the other half. Every claim carries chunk ids; the UI makes them clickable back
 * to the passage, because an unverifiable citation is decoration.
 */

import { useCallback, useRef, useState, type FormEvent } from "react";
import { api, ApiError, type AnswerResponse, type Citation } from "@/api/client";

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  const ask = useCallback(async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    // A second question supersedes the first rather than racing it. Without this, a slow answer
    // can arrive after a fast one and overwrite it.
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;

    setBusy(true);
    setError(null);
    try {
      const answer = await api.post<AnswerResponse>("/api/ask", { question: trimmed }, controller.signal);
      setResult(answer);
    } catch (caught) {
      if (controller.signal.aborted) return;
      setError(caught instanceof ApiError ? caught.message : "We could not reach the server.");
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }, [question]);

  return (
    <div className="space-y-6">
      <form onSubmit={ask} className="flex gap-2">
        <input
          type="search"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about your documents…"
          aria-label="Question"
          autoFocus
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-900 focus:ring-1 focus:ring-slate-900"
        />
        <button
          type="submit"
          disabled={busy || !question.trim()}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-60"
        >
          {busy ? "Thinking…" : "Ask"}
        </button>
      </form>

      {error && (
        <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900">
          {error}
        </div>
      )}

      {result && (result.answerable ? <Answer result={result} /> : <Abstention result={result} />)}
    </div>
  );
}

function Answer({ result }: { result: AnswerResponse }) {
  const byId = new Map(result.citations.map((citation) => [citation.chunk_id, citation]));

  return (
    <article className="space-y-6">
      <div className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="space-y-3 text-sm leading-relaxed text-slate-800">
          {result.claims.map((claim, index) => (
            <p key={index}>
              {claim.text}{" "}
              {claim.evidence_ids.map((id) => (
                <a
                  key={id}
                  href={`#source-${id}`}
                  title={byId.get(id)?.title ?? "Source"}
                  className="ml-0.5 rounded bg-slate-100 px-1.5 py-0.5 align-super text-[10px] font-medium text-slate-600 no-underline hover:bg-slate-200"
                >
                  {shortId(id)}
                </a>
              ))}
            </p>
          ))}
        </div>
      </div>

      <Sources citations={result.citations} />
    </article>
  );
}

/**
 * "I don't know", rendered as an answer rather than as a failure.
 *
 * Neutral styling on purpose. A red box would tell the user something went wrong, when what
 * actually happened is the system declining to invent something — the behaviour we most want it
 * to keep.
 */
function Abstention({ result }: { result: AnswerResponse }) {
  return (
    <article className="space-y-6">
      <div className="rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="mb-2 text-sm font-semibold text-slate-900">No answer in your documents</h2>
        <p className="text-sm leading-relaxed text-slate-700">{result.answer}</p>

        {result.sub_questions_answered.length > 0 && (
          <div className="mt-4 border-t border-slate-100 pt-4">
            <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-slate-500">
              Partly answered
            </p>
            <ul className="list-inside list-disc space-y-1 text-sm text-slate-700">
              {result.sub_questions_answered.map((sub) => (
                <li key={sub}>{sub}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {result.citations.length > 0 && (
        <div>
          <p className="mb-2 text-xs text-slate-500">
            The closest material we found. It did not support an answer, but it may still be
            useful.
          </p>
          <Sources citations={result.citations} />
        </div>
      )}
    </article>
  );
}

function Sources({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  return (
    <section aria-label="Sources" className="space-y-2">
      <h2 className="text-xs font-medium uppercase tracking-wide text-slate-500">Sources</h2>
      {citations.map((citation) => (
        <div
          key={citation.chunk_id}
          id={`source-${citation.chunk_id}`}
          className="rounded-md border border-slate-200 bg-white p-4 target:border-slate-900"
        >
          <div className="mb-1 flex items-baseline gap-2">
            <a href={`/documents/${citation.doc_id}`} className="text-sm font-medium hover:underline">
              {citation.title}
            </a>
            {citation.page_from !== null && (
              <span className="text-xs text-slate-400">p.&nbsp;{citation.page_from}</span>
            )}
            <span className="ml-auto font-mono text-[10px] text-slate-400">
              {shortId(citation.chunk_id)}
            </span>
          </div>
          {citation.heading_path && (
            <p className="mb-2 text-xs text-slate-500">{citation.heading_path}</p>
          )}
          {/* Deliberately plain text, never dangerouslySetInnerHTML. This is document content
              from an untrusted corpus, and a markdown image in it is the classic exfiltration
              channel -- the renderer must not fetch a remote URL on its behalf. */}
          <p className="text-sm leading-relaxed text-slate-700">{citation.snippet}</p>
        </div>
      ))}
    </section>
  );
}

function shortId(id: string): string {
  return id.slice(0, 6);
}

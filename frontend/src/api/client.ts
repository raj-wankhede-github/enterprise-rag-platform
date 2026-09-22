/**
 * The one place a request leaves the browser.
 *
 * Everything the backend's auth design requires on this side lives here, so no page has to
 * remember it:
 *
 * - **Cookies, never a stored token.** `credentials: "same-origin"` and nothing in
 *   localStorage. A token in storage is a token an XSS can read, which is the entire reason the
 *   backend sets HttpOnly cookies rather than returning one in a body.
 * - **The CSRF header on every write.** Read from the deliberately script-readable `erp_csrf`
 *   cookie and echoed back. The backend also checks Origin, so this is the second of three
 *   overlapping controls rather than the only one.
 * - **One refresh at a time, with the failed requests queued behind it.** A page that fires six
 *   requests on mount would otherwise start six refreshes; five of them present a token the
 *   first has already rotated, the backend sees replay, and the session family is revoked. The
 *   user is signed out of everything by the act of loading a page. This single-flight is not an
 *   optimisation -- without it, refresh-token reuse detection makes the app unusable.
 */

import type { Capability, Role } from "@/auth/capabilities.generated";

export const CSRF_COOKIE = "erp_csrf";
export const CSRF_HEADER = "X-CSRF-Token";

const UNSAFE = new Set(["POST", "PUT", "PATCH", "DELETE"]);

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly extra: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** The backend returns 404 rather than 403 for another tenant's resources, deliberately. */
  get isNotFound(): boolean {
    return this.status === 404;
  }

  get isUnauthenticated(): boolean {
    return this.status === 401;
  }

  get isForbidden(): boolean {
    return this.status === 403;
  }
}

export function readCookie(name: string): string | null {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return match?.[1] ? decodeURIComponent(match[1]) : null;
}

type RequestOptions = {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  /** Set on the refresh call itself, so a failed refresh cannot recurse into another refresh. */
  skipRefresh?: boolean;
};

/** Called when a session ends for good. Set once by AuthProvider. */
let onSessionLost: (() => void) | null = null;

export function setSessionLostHandler(handler: (() => void) | null): void {
  onSessionLost = handler;
}

let refreshing: Promise<boolean> | null = null;

/**
 * Refresh once, however many callers ask.
 *
 * Every concurrent 401 awaits the same promise. The alternative -- each retrying on its own --
 * presents an already-rotated token, which the backend correctly treats as theft and answers by
 * revoking the whole session family.
 */
async function refreshOnce(): Promise<boolean> {
  refreshing ??= (async () => {
    try {
      const response = await fetch("/api/auth/refresh", {
        method: "POST",
        credentials: "same-origin",
        headers: csrfHeaders("POST"),
      });
      return response.ok;
    } catch {
      return false;
    } finally {
      // Cleared in a microtask rather than immediately, so callers that arrive during the
      // settle still join this attempt instead of starting another.
      queueMicrotask(() => {
        refreshing = null;
      });
    }
  })();
  return refreshing;
}

function csrfHeaders(method: string): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (UNSAFE.has(method.toUpperCase())) {
    const token = readCookie(CSRF_COOKIE);
    if (token) headers[CSRF_HEADER] = token;
  }
  return headers;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();

  const send = (): Promise<Response> =>
    fetch(path, {
      method,
      credentials: "same-origin",
      headers: csrfHeaders(method),
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options.signal,
    });

  let response = await send();

  if (response.status === 401 && !options.skipRefresh) {
    const refreshed = await refreshOnce();
    if (refreshed) {
      response = await send();
    } else {
      // The refresh failed, so the session is gone for good. Tell the app once rather than
      // letting every in-flight request independently decide to redirect.
      onSessionLost?.();
    }
  }

  if (response.status === 204) return undefined as T;

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new ApiError(
      response.status,
      String((payload as { code?: string }).code ?? "error"),
      // The backend's `message` is always safe to show; `detail` never crosses the wire.
      String((payload as { message?: string }).message ?? "Something went wrong."),
      (payload as { extra?: Record<string, unknown> }).extra ?? {},
    );
  }

  return payload as T;
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => request<T>(path, { signal }),
  post: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    request<T>(path, { method: "POST", body, signal }),
  patch: <T>(path: string, body?: unknown) => request<T>(path, { method: "PATCH", body }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

// ---------------------------------------------------------------------------------------------
// Shapes the backend returns. Kept here so a response change breaks compilation in one place.
// ---------------------------------------------------------------------------------------------

export type LoginMethod = {
  kind: string;
  display_name: string;
  start_url: string;
};

export type DiscoverResponse = {
  tenant_slug: string | null;
  tenant_name: string | null;
  password_login: boolean;
  methods: LoginMethod[];
  requires_tenant_choice: boolean;
  tenants: { slug: string; name: string }[];
  emergency_access: boolean;
};

export type SessionResponse = {
  user_id: string;
  tenant_slug: string;
  email: string;
  display_name: string;
  role: Role;
  capabilities: Capability[];
  expires_at: string;
};

export type Citation = {
  chunk_id: string;
  doc_id: string;
  title: string;
  heading_path: string;
  page_from: number | null;
  snippet: string;
};

export type AnswerResponse = {
  /**
   * False is a successful outcome, not an error. The backend abstains when the evidence does not
   * support an answer, and the UI must render that as an answer in its own right rather than as
   * a failure -- otherwise the one honest thing the system does looks like a bug.
   */
  answerable: boolean;
  answer: string;
  claims: { text: string; evidence_ids: string[] }[];
  citations: Citation[];
  abstention_reason: string | null;
  sub_questions_answered: string[];
  diagnostics: Record<string, unknown>;
};

export type SearchHit = {
  chunk_id: string;
  doc_id: string;
  title: string;
  heading_path: string;
  snippet: string;
  score: number;
  legs: Record<string, { rank: number; score: number }>;
};

export type SearchResponse = {
  hits: SearchHit[];
  total: number;
  took_ms: number;
  diagnostics: Record<string, unknown>;
};

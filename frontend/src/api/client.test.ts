/**
 * The API client.
 *
 * The single-flight refresh test is the one that matters. Without it, a page that fires six
 * requests on mount starts six refreshes; five present a token the first has already rotated,
 * the backend correctly reads that as replay, and the whole session family is revoked. The user
 * is signed out of every device by the act of loading a page — and the backend is behaving
 * exactly as designed, so the bug is invisible from that side.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, CSRF_COOKIE, CSRF_HEADER, readCookie, request, setSessionLostHandler } from "./client";

function jsonResponse(status: number, body: unknown = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
  setSessionLostHandler(null);
});

describe("cookies and CSRF", () => {
  it("reads a cookie by name", () => {
    document.cookie = `${CSRF_COOKIE}=abc123; path=/`;
    expect(readCookie(CSRF_COOKIE)).toBe("abc123");
  });

  it("returns null for a cookie that is not set", () => {
    expect(readCookie("nothing_here")).toBeNull();
  });

  it("sends the CSRF header on a write", async () => {
    document.cookie = `${CSRF_COOKIE}=token-value; path=/`;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { ok: true }));

    await api.post("/api/documents", { title: "x" });

    const headers = fetchMock.mock.calls[0]![1]!.headers as Record<string, string>;
    expect(headers[CSRF_HEADER]).toBe("token-value");
  });

  it("does not send the CSRF header on a read", async () => {
    document.cookie = `${CSRF_COOKIE}=token-value; path=/`;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, {}));

    await api.get("/api/documents");

    const headers = fetchMock.mock.calls[0]![1]!.headers as Record<string, string>;
    expect(headers[CSRF_HEADER]).toBeUndefined();
  });

  it("always sends cookies same-origin and never a stored token", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, {}));
    await api.get("/api/documents");

    const init = fetchMock.mock.calls[0]![1]!;
    expect(init.credentials).toBe("same-origin");
    expect((init.headers as Record<string, string>).Authorization).toBeUndefined();
    expect(localStorage.length).toBe(0);
  });
});

describe("errors", () => {
  it("throws an ApiError carrying the server's safe message", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(409, { code: "conflict", message: "A document with this content already exists." }),
    );

    await expect(api.post("/api/documents")).rejects.toMatchObject({
      status: 409,
      code: "conflict",
      message: "A document with this content already exists.",
    });
  });

  it("never surfaces a detail field, because the server never sends one", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(403, { code: "forbidden", message: "You do not have permission." }),
    );

    const error = await api.get("/api/admin/users").catch((caught) => caught as ApiError);
    expect(error).toBeInstanceOf(ApiError);
    expect(JSON.stringify(error)).not.toContain("detail");
  });

  it("treats 404 as not-found rather than forbidden", async () => {
    // The backend returns 404 for another tenant's resource on purpose: a 403 would confirm it
    // exists. The client must not undo that by rendering "forbidden".
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(404, { code: "not_found", message: "Not found." }));

    const error = (await api.get("/api/documents/other-tenant-doc").catch((c) => c)) as ApiError;
    expect(error.isNotFound).toBe(true);
    expect(error.isForbidden).toBe(false);
  });

  it("handles a 204 with no body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 204 }));
    await expect(api.post("/api/auth/logout")).resolves.toBeUndefined();
  });

  it("handles a non-JSON error body without throwing a parse error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("<html>502</html>", { status: 502 }));
    await expect(api.get("/api/search")).rejects.toBeInstanceOf(ApiError);
  });
});

describe("refresh", () => {
  it("retries once after a successful refresh", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(401, { code: "unauthenticated" }))
      .mockResolvedValueOnce(jsonResponse(200, {})) // the refresh
      .mockResolvedValueOnce(jsonResponse(200, { hits: [] })); // the retry

    await expect(api.get("/api/search")).resolves.toEqual({ hits: [] });
    expect(fetchMock.mock.calls[1]![0]).toBe("/api/auth/refresh");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("refreshes ONCE for many concurrent 401s", async () => {
    // The test this file exists for. Six parallel refreshes means five replays of an
    // already-rotated token, which the backend reads as theft and answers by revoking the
    // family — signing the user out everywhere, because they loaded a page.
    let refreshes = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url === "/api/auth/refresh") {
        refreshes += 1;
        await new Promise((resolve) => setTimeout(resolve, 5));
        return jsonResponse(200, {});
      }
      return refreshes === 0 ? jsonResponse(401, {}) : jsonResponse(200, { ok: true });
    });

    await Promise.all([
      api.get("/api/a"),
      api.get("/api/b"),
      api.get("/api/c"),
      api.get("/api/d"),
      api.get("/api/e"),
      api.get("/api/f"),
    ]);

    expect(refreshes).toBe(1);
  });

  it("does not try to refresh the refresh call itself", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(401, {}));
    await expect(request("/api/auth/refresh", { method: "POST", skipRefresh: true })).rejects.toBeInstanceOf(
      ApiError,
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("tells the app once when the session is gone for good", async () => {
    const lost = vi.fn();
    setSessionLostHandler(lost);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(401, {}));

    await api.get("/api/search").catch(() => undefined);

    expect(lost).toHaveBeenCalledTimes(1);
  });

  it("does not treat a network failure during refresh as a successful one", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(401, {}))
      .mockRejectedValueOnce(new TypeError("network"));

    await expect(api.get("/api/search")).rejects.toBeInstanceOf(ApiError);
  });
});

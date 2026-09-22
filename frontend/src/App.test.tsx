/**
 * The route guards.
 *
 * Both are about rendering, never about enforcement — the server re-checks every request against
 * a principal it reloads from Postgres. What they buy is that an unauthenticated visitor does
 * not see a flash of the signed-in shell, and that someone without a capability gets an
 * explanation instead of a page made entirely of errors.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { RequireAuth, RequireCapability } from "./App";
import { AuthProvider } from "@/auth/AuthContext";
import type { SessionResponse } from "@/api/client";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const SESSION: SessionResponse = {
  user_id: "11111111-1111-1111-1111-111111111111",
  tenant_slug: "acme",
  email: "a@acme.com",
  display_name: "A Person",
  role: "DEV",
  capabilities: ["ask", "search", "doc:upload"],
  expires_at: new Date(Date.now() + 600_000).toISOString(),
};

function mockSession(session: SessionResponse | null) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    session ? jsonResponse(200, session) : jsonResponse(401, { code: "unauthenticated" }),
  );
}

function renderGuarded(children: React.ReactNode) {
  return render(
    <MemoryRouter initialEntries={["/private"]}>
      <AuthProvider>
        <Routes>
          <Route path="/private" element={<RequireAuth>{children}</RequireAuth>} />
          <Route path="/login" element={<div>Sign in page</div>} />
        </Routes>
      </AuthProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("RequireAuth", () => {
  it("shows nothing but a loading state until the session is known", () => {
    // Redirecting during load would bounce a signed-in user to the login page on every cold
    // load; rendering the app would flash the signed-in shell to everyone else.
    vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise(() => {}));
    renderGuarded(<div>Private content</div>);

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.queryByText("Private content")).not.toBeInTheDocument();
    expect(screen.queryByText("Sign in page")).not.toBeInTheDocument();
  });

  it("renders the page for a signed-in user", async () => {
    mockSession(SESSION);
    renderGuarded(<div>Private content</div>);
    expect(await screen.findByText("Private content")).toBeInTheDocument();
  });

  it("sends an anonymous visitor to the login page", async () => {
    mockSession(null);
    renderGuarded(<div>Private content</div>);

    expect(await screen.findByText("Sign in page")).toBeInTheDocument();
    expect(screen.queryByText("Private content")).not.toBeInTheDocument();
  });

  it("never renders the private content even briefly for an anonymous visitor", async () => {
    mockSession(null);
    renderGuarded(<div>Private content</div>);
    // Asserted after the redirect has settled: if it had rendered and then unmounted, the flash
    // would still have happened, so the loading-state test above covers the other half.
    await screen.findByText("Sign in page");
    expect(screen.queryByText("Private content")).not.toBeInTheDocument();
  });
});

describe("RequireCapability", () => {
  it("renders the page when the principal holds the capability", async () => {
    mockSession(SESSION);
    renderGuarded(
      <RequireCapability capability="doc:upload">
        <div>Upload form</div>
      </RequireCapability>,
    );
    expect(await screen.findByText("Upload form")).toBeInTheDocument();
  });

  it("explains rather than showing a page made of errors", async () => {
    mockSession(SESSION);
    renderGuarded(
      <RequireCapability capability="sso:configure">
        <div>SSO settings</div>
      </RequireCapability>,
    );

    expect(await screen.findByText(/do not have access to this page/i)).toBeInTheDocument();
    expect(screen.queryByText("SSO settings")).not.toBeInTheDocument();
  });

  it("does not name the capability to the user", async () => {
    // "You are missing sso:configure" is a message for a developer. A user needs to know who to
    // ask, which is what the wording says instead.
    mockSession(SESSION);
    renderGuarded(
      <RequireCapability capability="sso:configure">
        <div>SSO settings</div>
      </RequireCapability>,
    );

    const page = await screen.findByText(/do not have access to this page/i);
    expect(page.parentElement?.textContent).not.toContain("sso:configure");
    expect(page.parentElement?.textContent).toMatch(/administrator/i);
  });
});

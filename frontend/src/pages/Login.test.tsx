/**
 * The login state machine.
 *
 * Two properties are worth a test at this level: that the page never renders a method the tenant
 * has not enabled, and that it does not helpfully re-distinguish failures the backend
 * deliberately made identical.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import Login from "./Login";
import { AuthProvider } from "@/auth/AuthContext";
import type { DiscoverResponse } from "@/api/client";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const PASSWORD_ONLY: DiscoverResponse = {
  tenant_slug: "acme",
  tenant_name: "Acme GmbH",
  password_login: true,
  methods: [],
  requires_tenant_choice: false,
  tenants: [],
  emergency_access: false,
};

const SSO_ONLY: DiscoverResponse = {
  ...PASSWORD_ONLY,
  password_login: false,
  methods: [
    { kind: "entra", display_name: "Continue with Microsoft", start_url: "/api/auth/oidc/abc/start" },
  ],
};

/** Route every call except the named discovery response to a 401, so nothing leaks between tests. */
function mockBackend(handlers: Record<string, () => Response>) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    const handler = Object.entries(handlers).find(([path]) => url.startsWith(path))?.[1];
    return handler ? handler() : jsonResponse(401, { code: "unauthenticated" });
  });
}

/**
 * Render, then wait for AuthProvider's initial session probe to settle.
 *
 * Without the wait, that probe resolves after the test's synchronous render and React reports an
 * unwrapped state update. The warning is noise here, but noise in a test suite is how a real
 * warning later goes unread.
 */
async function renderLogin(initialEntry = "/login") {
  const result = render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <AuthProvider>
        <Login />
      </AuthProvider>
    </MemoryRouter>,
  );
  await screen.findByLabelText("Work email").catch(() => undefined);
  return result;
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("email-first discovery", () => {
  it("asks for an email before showing any sign-in method", async () => {
    mockBackend({});
    await renderLogin();

    expect(screen.getByLabelText("Work email")).toBeInTheDocument();
    expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
  });

  it("shows a password field for a tenant that allows passwords", async () => {
    mockBackend({ "/api/auth/discover": () => jsonResponse(200, PASSWORD_ONLY) });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByLabelText("Password")).toBeInTheDocument();
  });

  it("shows no password field for an SSO-only tenant", async () => {
    mockBackend({ "/api/auth/discover": () => jsonResponse(200, SSO_ONLY) });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByRole("link", { name: "Continue with Microsoft" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
  });

  it("names the organisation once it is known, so the user knows where they are signing in", async () => {
    mockBackend({ "/api/auth/discover": () => jsonResponse(200, PASSWORD_ONLY) });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByText("Acme GmbH")).toBeInTheDocument();
  });

  it("lets the user go back and try a different address", async () => {
    mockBackend({ "/api/auth/discover": () => jsonResponse(200, PASSWORD_ONLY) });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));
    await userEvent.click(await screen.findByText("Use a different email address"));

    expect(screen.getByLabelText("Work email")).toBeInTheDocument();
    expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
  });

  it("asks which organisation when a domain belongs to several", async () => {
    mockBackend({
      "/api/auth/discover": () =>
        jsonResponse(200, {
          ...PASSWORD_ONLY,
          requires_tenant_choice: true,
          tenants: [
            { slug: "acme", name: "Acme GmbH" },
            { slug: "beta", name: "Beta AG" },
          ],
        }),
    });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@shared.example");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByRole("button", { name: "Acme GmbH" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Beta AG" })).toBeInTheDocument();
  });

  it("says plainly when a tenant has no way to sign in at all", async () => {
    mockBackend({
      "/api/auth/discover": () => jsonResponse(200, { ...PASSWORD_ONLY, password_login: false, methods: [] }),
    });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByText(/no sign-in method enabled/i)).toBeInTheDocument();
  });

  it("warns when password sign-in is only on because of the break-glass window", async () => {
    // A tenant that believes SSO is enforced needs to be told when it temporarily is not.
    mockBackend({
      "/api/auth/discover": () => jsonResponse(200, { ...SSO_ONLY, password_login: true, emergency_access: true }),
    });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByText(/temporarily enabled/i)).toBeInTheDocument();
  });
});

describe("failure messages", () => {
  it("shows the server's single message and does not add detail of its own", async () => {
    const message = "That email address and password combination was not recognised.";
    mockBackend({
      "/api/auth/discover": () => jsonResponse(200, PASSWORD_ONLY),
      "/api/auth/login": () => jsonResponse(401, { code: "unauthenticated", message }),
    });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));
    await userEvent.type(await screen.findByLabelText("Password"), "wrong-password");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(message);
    // Nothing that would distinguish a wrong password from an unknown account.
    expect(alert.textContent).not.toMatch(/no such|unknown|deactivated|sso/i);
  });

  it("clears the password field after a failure", async () => {
    mockBackend({
      "/api/auth/discover": () => jsonResponse(200, PASSWORD_ONLY),
      "/api/auth/login": () => jsonResponse(401, { code: "unauthenticated", message: "Not recognised." }),
    });
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));
    const password = await screen.findByLabelText("Password");
    await userEvent.type(password, "wrong-password");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => expect(password).toHaveValue(""));
  });

  it("turns an OIDC callback code into wording rather than showing the code", async () => {
    mockBackend({});
    await renderLogin("/login?error=account_deactivated");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/deactivated/i);
    expect(alert.textContent).not.toContain("account_deactivated");
  });

  it("falls back to generic wording for a code it does not recognise", async () => {
    mockBackend({});
    await renderLogin("/login?error=something_new_from_the_backend");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/could not sign you in/i);
    expect(alert.textContent).not.toContain("something_new_from_the_backend");
  });

  it("does not show a network failure as an authentication failure", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("network"));
    await renderLogin();

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/could not reach the server/i);
  });
});

describe("the redirect after sign-in", () => {
  it("carries a same-origin next path through to the SSO button", async () => {
    mockBackend({ "/api/auth/discover": () => jsonResponse(200, SSO_ONLY) });
    await renderLogin("/login?next=%2Fdocuments");

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    const link = await screen.findByRole("link", { name: "Continue with Microsoft" });
    expect(link).toHaveAttribute("href", expect.stringContaining("next=%2Fdocuments"));
  });

  it("discards a next path that would leave the origin", async () => {
    // The standard way a login page becomes a phishing relay. The server checks this too; the
    // client checking as well means a bug on either side alone is not enough.
    mockBackend({ "/api/auth/discover": () => jsonResponse(200, SSO_ONLY) });
    await renderLogin("/login?next=https%3A%2F%2Fattacker.example.com");

    await userEvent.type(screen.getByLabelText("Work email"), "a@acme.com");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    const link = await screen.findByRole("link", { name: "Continue with Microsoft" });
    expect(link.getAttribute("href")).not.toContain("attacker.example.com");
    expect(link).toHaveAttribute("href", expect.stringContaining("next=%2Fsearch"));
  });
});

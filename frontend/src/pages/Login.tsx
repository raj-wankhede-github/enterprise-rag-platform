/**
 * Email-first login.
 *
 * The user types an address, the server says which tenant that is and how that tenant allows
 * signing in, and only then does the page render a password field or SSO buttons. The
 * alternative -- showing one password field and three provider buttons to everyone -- means most
 * of what is on screen will fail for any given person.
 *
 * Two things this page is careful about:
 *
 * **It never treats the discovery response as authorisation.** `password_login: false` hides the
 * field; the server independently refuses a password login for that tenant. If the two ever
 * disagree, the server wins and the user sees its message.
 *
 * **It shows one failure message for every cause.** The backend returns one message for an
 * unknown address, a wrong password, a deactivated account and an SSO-only account, and this
 * page must not helpfully distinguish them again in the UI.
 */

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, ApiError, type DiscoverResponse, type SessionResponse } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";

type Stage = "email" | "choose-tenant" | "methods";

/** Codes the OIDC callback redirects here with. Each maps to wording, never to a raw code. */
const CALLBACK_ERRORS: Record<string, string> = {
  provider_declined:
    "Your identity provider did not complete the sign-in. This usually means consent was declined, or your account is not assigned to this application.",
  account_deactivated:
    "Your account has been deactivated. Please contact an administrator at your organisation.",
  not_provisioned:
    "You signed in successfully, but you do not yet have an account here. Please ask an administrator to add you.",
  expired: "That sign-in request expired. Please try again.",
  incomplete: "The sign-in did not complete. Please try again.",
  unavailable: "That sign-in method is no longer available.",
  verification_failed:
    "We could not verify the response from your identity provider. Please try again, or contact an administrator.",
  sign_in_failed: "We could not sign you in. Please try again.",
};

export default function Login() {
  const navigate = useNavigate();
  const { signIn } = useAuth();
  const [params] = useSearchParams();

  const [stage, setStage] = useState<Stage>("email");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [discovery, setDiscovery] = useState<DiscoverResponse | null>(null);
  const [tenantSlug, setTenantSlug] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const passwordRef = useRef<HTMLInputElement>(null);
  const callbackError = params.get("error");

  useEffect(() => {
    if (callbackError) setError(CALLBACK_ERRORS[callbackError] ?? CALLBACK_ERRORS.sign_in_failed!);
  }, [callbackError]);

  useEffect(() => {
    // Move focus to the password field once it appears, so the flow stays keyboard-only.
    if (stage === "methods" && discovery?.password_login) passwordRef.current?.focus();
  }, [stage, discovery]);

  const discover = useCallback(
    async (slug: string | null) => {
      setBusy(true);
      setError(null);
      try {
        const result = await api.post<DiscoverResponse>("/api/auth/discover", {
          email,
          tenant_slug: slug,
        });
        setDiscovery(result);
        setStage(result.requires_tenant_choice ? "choose-tenant" : "methods");
      } catch (caught) {
        setError(messageOf(caught));
      } finally {
        setBusy(false);
      }
    },
    [email],
  );

  async function onSubmitEmail(event: FormEvent) {
    event.preventDefault();
    await discover(tenantSlug);
  }

  async function onSubmitPassword(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const session = await api.post<SessionResponse>("/api/auth/login", {
        email,
        password,
        tenant_slug: tenantSlug ?? discovery?.tenant_slug ?? null,
      });
      signIn(session);
      navigate(nextPath(params.get("next")), { replace: true });
    } catch (caught) {
      setError(messageOf(caught));
      setPassword("");
      passwordRef.current?.focus();
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <h1 className="mb-1 text-2xl font-semibold tracking-tight">Knowledge Search</h1>
        <p className="mb-8 text-sm text-slate-500">
          {stage === "email" ? "Sign in to continue." : (discovery?.tenant_name ?? "Sign in to continue.")}
        </p>

        {error && (
          <div role="alert" className="mb-4 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            {error}
          </div>
        )}

        {discovery?.emergency_access && (
          <div
            role="status"
            className="mb-4 rounded-md border border-orange-300 bg-orange-50 p-3 text-sm text-orange-900"
          >
            Password sign-in is temporarily enabled for this organisation while its identity
            provider is unavailable. It will be disabled again automatically.
          </div>
        )}

        {stage === "email" && (
          <form onSubmit={onSubmitEmail} className="space-y-4">
            <Field
              id="email"
              label="Work email"
              type="email"
              value={email}
              onChange={setEmail}
              autoComplete="username"
              autoFocus
              required
            />
            <Submit busy={busy} label="Continue" />
          </form>
        )}

        {stage === "choose-tenant" && discovery && (
          <div className="space-y-3">
            <p className="text-sm text-slate-600">
              That address belongs to more than one organisation. Which one are you signing in to?
            </p>
            {discovery.tenants.map((tenant) => (
              <button
                key={tenant.slug}
                type="button"
                onClick={() => {
                  setTenantSlug(tenant.slug);
                  void discover(tenant.slug);
                }}
                className="w-full rounded-md border border-slate-300 bg-white px-4 py-2.5 text-left text-sm font-medium hover:border-slate-400 hover:bg-slate-50"
              >
                {tenant.name}
              </button>
            ))}
          </div>
        )}

        {stage === "methods" && discovery && (
          <div className="space-y-4">
            {discovery.methods.map((method) => (
              <a
                key={method.start_url}
                href={withNext(method.start_url, params.get("next"))}
                className="block w-full rounded-md border border-slate-300 bg-white px-4 py-2.5 text-center text-sm font-medium hover:border-slate-400 hover:bg-slate-50"
              >
                {method.display_name}
              </a>
            ))}

            {discovery.methods.length > 0 && discovery.password_login && (
              <div className="flex items-center gap-3 py-1 text-xs uppercase tracking-wide text-slate-400">
                <span className="h-px flex-1 bg-slate-200" />
                or
                <span className="h-px flex-1 bg-slate-200" />
              </div>
            )}

            {discovery.password_login && (
              <form onSubmit={onSubmitPassword} className="space-y-4">
                <input type="hidden" name="username" value={email} autoComplete="username" />
                <Field
                  id="password"
                  label="Password"
                  type="password"
                  value={password}
                  onChange={setPassword}
                  autoComplete="current-password"
                  inputRef={passwordRef}
                  required
                />
                <Submit busy={busy} label="Sign in" />
              </form>
            )}

            {!discovery.password_login && discovery.methods.length === 0 && (
              <p className="text-sm text-slate-600">
                This organisation has no sign-in method enabled. Please contact an administrator.
              </p>
            )}

            <button
              type="button"
              onClick={() => {
                setStage("email");
                setDiscovery(null);
                setPassword("");
                setError(null);
              }}
              className="w-full text-center text-sm text-slate-500 underline-offset-2 hover:underline"
            >
              Use a different email address
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

function Field({
  id,
  label,
  type,
  value,
  onChange,
  inputRef,
  ...rest
}: {
  id: string;
  label: string;
  type: string;
  value: string;
  onChange: (value: string) => void;
  inputRef?: React.Ref<HTMLInputElement>;
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange" | "value" | "type" | "id">) {
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-sm font-medium text-slate-700">
        {label}
      </label>
      <input
        id={id}
        ref={inputRef}
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-900 focus:ring-1 focus:ring-slate-900"
        {...rest}
      />
    </div>
  );
}

function Submit({ busy, label }: { busy: boolean; label: string }) {
  return (
    <button
      type="submit"
      disabled={busy}
      className="w-full rounded-md bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-60"
    >
      {busy ? "Working…" : label}
    </button>
  );
}

function messageOf(caught: unknown): string {
  // The backend's `message` is always safe to show; anything else gets a generic line rather
  // than a stack trace or a network error string.
  if (caught instanceof ApiError) return caught.message;
  return "We could not reach the server. Please check your connection and try again.";
}

/** Same-origin paths only. Mirrors the server's own check rather than trusting it. */
function nextPath(candidate: string | null): string {
  if (!candidate || !candidate.startsWith("/") || candidate.startsWith("//")) return "/search";
  return candidate;
}

function withNext(startUrl: string, next: string | null): string {
  const safe = nextPath(next);
  const separator = startUrl.includes("?") ? "&" : "?";
  return `${startUrl}${separator}next=${encodeURIComponent(safe)}`;
}

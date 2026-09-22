/**
 * The route tree and the two guards every authenticated route passes through.
 *
 * `RequireAuth` holds rendering until the session is known. Rendering the app first and
 * redirecting on a 401 produces a visible flash of the signed-in shell for anyone who is not
 * signed in, and worse, fires the page's data requests before anyone knows whether they should.
 *
 * `RequireCapability` is a *rendering* guard. The server re-checks every request against a
 * principal it reloads from Postgres, so this exists to avoid showing someone a page made
 * entirely of errors -- not to enforce anything.
 */

import { lazy, Suspense, type ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import type { Capability } from "@/auth/capabilities.generated";
import Shell from "@/components/Shell";

const Login = lazy(() => import("@/pages/Login"));
const Search = lazy(() => import("@/pages/Search"));
const Ask = lazy(() => import("@/pages/Ask"));
const Documents = lazy(() => import("@/pages/Documents"));
const Upload = lazy(() => import("@/pages/Upload"));
const Users = lazy(() => import("@/pages/admin/Users"));
const Sso = lazy(() => import("@/pages/admin/Sso"));
const NotFound = lazy(() => import("@/pages/NotFound"));

export default function App() {
  return (
    <Suspense fallback={<Splash />}>
      <Routes>
        <Route path="/login" element={<Login />} />

        <Route
          element={
            <RequireAuth>
              <Shell />
            </RequireAuth>
          }
        >
          <Route index element={<Navigate to="/search" replace />} />
          <Route path="/search" element={<Search />} />
          <Route
            path="/ask"
            element={
              <RequireCapability capability="ask">
                <Ask />
              </RequireCapability>
            }
          />
          <Route path="/documents" element={<Documents />} />
          <Route
            path="/documents/upload"
            element={
              <RequireCapability capability="doc:upload">
                <Upload />
              </RequireCapability>
            }
          />
          <Route
            path="/admin/users"
            element={
              <RequireCapability capability="user:manage">
                <Users />
              </RequireCapability>
            }
          />
          <Route
            path="/admin/sso"
            element={
              <RequireCapability capability="sso:configure">
                <Sso />
              </RequireCapability>
            }
          />
        </Route>

        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
}

export function RequireAuth({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const location = useLocation();

  // Not a redirect: until the session is known, redirecting would bounce a signed-in user to
  // the login page on every cold load.
  if (status === "loading") return <Splash />;

  if (status === "anonymous") {
    const next = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?next=${next}`} replace />;
  }

  return <>{children}</>;
}

export function RequireCapability({
  capability,
  children,
}: {
  capability: Capability;
  children: ReactNode;
}) {
  const { can } = useAuth();
  if (!can(capability)) return <Forbidden />;
  return <>{children}</>;
}

function Splash() {
  return (
    <div className="flex min-h-screen items-center justify-center" role="status" aria-live="polite">
      <span className="text-sm text-slate-400">Loading…</span>
    </div>
  );
}

function Forbidden() {
  return (
    <div className="mx-auto max-w-lg px-4 py-16">
      <h1 className="mb-2 text-lg font-semibold">You do not have access to this page</h1>
      <p className="text-sm text-slate-600">
        Your role does not include this area of the product. If you think that is wrong, ask an
        administrator at your organisation to review your access.
      </p>
    </div>
  );
}

/**
 * The session, and the capability check every gated control uses.
 *
 * Three things this deliberately does not do:
 *
 * **It does not store the principal anywhere durable.** No localStorage, no sessionStorage. The
 * session lives in HttpOnly cookies the browser holds, and this is a render-time copy that is
 * re-fetched on load. Persisting it would mean a demoted or deactivated user keeps their old
 * role in the UI until they happen to reload.
 *
 * **`useCan` is a rendering decision, never an authorisation one.** The server re-checks every
 * request against a principal it reloads from Postgres. Hiding a button the user cannot use is
 * courtesy; the button being hidden is not what stops them.
 *
 * **It does not poll.** The access token is short-lived and the API client refreshes it on
 * demand. A background timer would either be wrong about the expiry or fire constantly, and the
 * failure it guards against -- a request arriving after expiry -- is already handled where it
 * happens.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, ApiError, setSessionLostHandler, type SessionResponse } from "@/api/client";
import { RANK, type Capability, type Role } from "@/auth/capabilities.generated";

export type AuthStatus = "loading" | "authenticated" | "anonymous";

export type AuthValue = {
  status: AuthStatus;
  session: SessionResponse | null;
  can: (capability: Capability) => boolean;
  /** Whether the principal may see documents at this visibility rank. */
  maySeeRank: (rank: number) => boolean;
  signIn: (session: SessionResponse) => void;
  signOut: (allDevices?: boolean) => Promise<void>;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [session, setSession] = useState<SessionResponse | null>(null);

  const load = useCallback(async () => {
    try {
      // Not /api/auth/refresh: this must work for an already-valid session without rotating its
      // refresh token, or every page load would burn one rotation and make reuse detection
      // fire on any two tabs opened at once.
      const me = await api.get<SessionResponse>("/api/auth/me");
      setSession(me);
      setStatus("authenticated");
    } catch (error) {
      if (error instanceof ApiError && error.isUnauthenticated) {
        setSession(null);
        setStatus("anonymous");
        return;
      }
      // A network failure is not a sign-out. Treating it as one would drop a working session
      // every time the laptop lid closes.
      setStatus((current) => (current === "loading" ? "anonymous" : current));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    // One place decides what a lost session means, rather than each in-flight request racing to
    // redirect.
    setSessionLostHandler(() => {
      setSession(null);
      setStatus("anonymous");
    });
    return () => setSessionLostHandler(null);
  }, []);

  const value = useMemo<AuthValue>(() => {
    const granted = new Set(session?.capabilities ?? []);
    const rank = session ? (RANK[session.role] ?? 0) : 0;

    return {
      status,
      session,
      can: (capability) => granted.has(capability),
      maySeeRank: (visibilityRank) => visibilityRank <= rank,
      signIn: (next) => {
        setSession(next);
        setStatus("authenticated");
      },
      signOut: async (allDevices = false) => {
        try {
          await api.post(`/api/auth/logout${allDevices ? "?all_devices=true" : ""}`);
        } finally {
          // Clear locally whatever the server said. A logout that leaves the UI signed in
          // because a request failed is worse than one that leaves a row behind.
          setSession(null);
          setStatus("anonymous");
        }
      },
      refresh: load,
    };
  }, [status, session, load]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside an AuthProvider");
  return value;
}

/**
 * Whether to render a control.
 *
 * Named so it reads as a question about the UI. `useCan("doc:upload")` sitting next to an upload
 * button is obviously about showing the button; a function called `authorize` next to the same
 * button invites someone to believe it is doing the enforcing.
 */
export function useCan(capability: Capability): boolean {
  return useAuth().can(capability);
}

export function useRole(): Role | null {
  return useAuth().session?.role ?? null;
}

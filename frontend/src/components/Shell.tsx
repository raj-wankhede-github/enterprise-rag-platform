/**
 * The signed-in frame: navigation, tenant and account.
 *
 * Every nav item declares the capability it needs, and items the principal lacks are not
 * rendered. That is a courtesy rather than a control -- the server decides -- but it is the
 * difference between a product that fits the person using it and one that is mostly locked
 * doors.
 */

import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import type { Capability } from "@/auth/capabilities.generated";

type NavItem = { to: string; label: string; capability?: Capability };

const NAV: NavItem[] = [
  { to: "/search", label: "Search" },
  { to: "/ask", label: "Ask", capability: "ask" },
  { to: "/documents", label: "Documents" },
  { to: "/admin/users", label: "Users", capability: "user:manage" },
  { to: "/admin/sso", label: "Sign-in", capability: "sso:configure" },
];

export default function Shell() {
  const { session, can, signOut } = useAuth();
  const navigate = useNavigate();

  const visible = NAV.filter((item) => !item.capability || can(item.capability));

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          <span className="text-sm font-semibold tracking-tight">Knowledge Search</span>

          <nav className="flex items-center gap-1" aria-label="Main">
            {visible.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `rounded-md px-3 py-1.5 text-sm ${
                    isActive ? "bg-slate-100 font-medium text-slate-900" : "text-slate-600 hover:text-slate-900"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className="text-slate-500">
              {session?.display_name || session?.email}
              {/* The role is shown because in a four-role product "why can't I see that
                  document" is the commonest question, and the answer usually starts here. */}
              <span className="ml-2 rounded bg-slate-100 px-1.5 py-0.5 text-xs font-medium text-slate-600">
                {session?.role}
              </span>
            </span>
            <button
              type="button"
              onClick={async () => {
                await signOut();
                navigate("/login", { replace: true });
              }}
              className="text-slate-500 underline-offset-2 hover:text-slate-900 hover:underline"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8">
        <Outlet />
      </main>
    </div>
  );
}

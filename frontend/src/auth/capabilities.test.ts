/**
 * The generated capability table must equal the backend enum.
 *
 * Drift here does not fail loudly on its own -- it renders a button that then 403s, which reads
 * to a user as the product being broken rather than as them lacking permission. Worse in the
 * other direction: a capability the backend added but the frontend does not know about hides a
 * feature the customer is paying for, and nothing anywhere reports it.
 *
 * This test shells out to the backend rather than comparing against a second checked-in copy,
 * because two copies drift together and prove nothing. It skips when Python is unavailable, so
 * a frontend-only checkout still runs its suite; CI has both.
 */

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { CAPABILITIES, RANK, ROLE_CAPABILITIES, ROLES } from "./capabilities.generated";

const here = dirname(fileURLToPath(import.meta.url));
const backend = resolve(here, "../../../backend");
const generatedPath = resolve(here, "capabilities.generated.ts");

function regenerate(): string | null {
  if (!existsSync(backend)) return null;
  try {
    return execFileSync("uv", ["run", "python", "-m", "app.cli", "capabilities"], {
      cwd: backend,
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch {
    return null;
  }
}

describe("capability parity with the backend", () => {
  it("the checked-in file is exactly what the backend emits", () => {
    const fresh = regenerate();
    if (fresh === null) {
      // A frontend-only checkout, or no uv. CI has both, so the check still runs there.
      return;
    }
    const committed = readFileSync(generatedPath, "utf-8");
    expect(normalize(committed)).toBe(normalize(fresh));
  });
});

describe("the shape the UI relies on", () => {
  it("has exactly the four roles, and no fifth", () => {
    // The product owner fixed this. A fifth role appearing here means someone is smuggling one
    // in through the frontend, where there is no nesting test to catch it.
    expect([...ROLES].sort()).toEqual(["ADMIN", "DEV", "PROD", "TEST"]);
  });

  it("ranks ADMIN highest and PROD lowest", () => {
    expect(RANK.ADMIN).toBeGreaterThan(RANK.DEV);
    expect(RANK.DEV).toBeGreaterThan(RANK.TEST);
    expect(RANK.TEST).toBeGreaterThan(RANK.PROD);
  });

  it("spaces ranks so an intermediate one could be added without renumbering documents", () => {
    const ordered = [...ROLES].sort((a, b) => RANK[a] - RANK[b]);
    for (let i = 1; i < ordered.length; i += 1) {
      expect(RANK[ordered[i]!] - RANK[ordered[i - 1]!]).toBeGreaterThanOrEqual(10);
    }
  });

  it("nests strictly: each role is a superset of the one below", () => {
    // Mirrors the backend's own assertion. A capability that breaks nesting is a fifth role
    // wearing one of the four names.
    const ordered = [...ROLES].sort((a, b) => RANK[a] - RANK[b]);
    for (let i = 1; i < ordered.length; i += 1) {
      const lower = new Set(ROLE_CAPABILITIES[ordered[i - 1]!]);
      const higher = new Set(ROLE_CAPABILITIES[ordered[i]!]);
      for (const capability of lower) {
        expect(higher.has(capability)).toBe(true);
      }
    }
  });

  it("grants every capability to somebody", () => {
    const granted = new Set(ROLES.flatMap((role) => [...ROLE_CAPABILITIES[role]]));
    expect([...CAPABILITIES].filter((capability) => !granted.has(capability))).toEqual([]);
  });

  it("keeps bulk content export away from TEST", () => {
    // Reading in the UI and bulk content egress are different risk classes. TEST reads
    // everything and still must not be able to walk out with the corpus.
    expect(ROLE_CAPABILITIES.TEST).not.toContain("export:documents");
    expect(ROLE_CAPABILITIES.ADMIN).toContain("export:documents");
  });

  it("keeps sign-in configuration to ADMIN", () => {
    expect(ROLE_CAPABILITIES.DEV).not.toContain("sso:configure");
    expect(ROLE_CAPABILITIES.ADMIN).toContain("sso:configure");
  });
});

function normalize(source: string): string {
  return source.replace(/\r\n/g, "\n").trimEnd();
}

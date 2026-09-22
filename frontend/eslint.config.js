import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules", "src/auth/capabilities.generated.ts"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
    languageOptions: { ecmaVersion: 2022, globals: globals.browser },
    plugins: { "react-hooks": reactHooks, "react-refresh": reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "@typescript-eslint/no-explicit-any": "error",
      // Document text is rendered as text, never as markup: a remote image in a retrieved
      // passage is the classic exfiltration channel.
      "no-restricted-syntax": [
        "error",
        {
          selector: "JSXAttribute[name.name='dangerouslySetInnerHTML']",
          message: "Retrieved document content is untrusted. Render it as text.",
        },
      ],
      "no-restricted-globals": [
        "error",
        { name: "localStorage", message: "Session state lives in HttpOnly cookies, never in storage script can read." },
        { name: "sessionStorage", message: "Session state lives in HttpOnly cookies, never in storage script can read." },
      ],
    },
  },
  {
    // Tests may name the storage APIs in order to assert they are unused. The rule exists to
    // stop production code putting a session there, not to stop a test proving it did not.
    files: ["**/*.test.{ts,tsx}", "src/test/**"],
    rules: { "no-restricted-globals": "off" },
  },
);

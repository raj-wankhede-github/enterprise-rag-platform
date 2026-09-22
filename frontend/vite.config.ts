/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    // The API is proxied rather than called cross-origin so that the browser treats it as
    // same-origin in development too. Without this, __Host- cookies and the CSRF Origin check
    // behave differently locally than in production -- which is exactly the class of bug that
    // only shows up after deployment.
    proxy: {
      "/api": { target: "http://localhost:8001", changeOrigin: false },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});

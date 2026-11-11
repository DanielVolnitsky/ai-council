// Vite is two tools in one:
//   * `vite` (dev): an HTTP server that serves src/ as native ES modules and
//     transforms .tsx/.ts/.css on request. Edits are pushed to the browser over
//     a websocket (Hot Module Replacement), so React components update in place
//     without a full reload and without losing their state.
//   * `vite build` (prod): bundles everything into static files under dist/.
// This file configures both. Vitest also reads it, so tests share the same setup.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // The React plugin compiles JSX and wires React's Fast Refresh into HMR.
  plugins: [react()],
  server: {
    // The browser only ever talks to the Vite dev server (localhost:5173).
    // Any request whose path starts with /api is forwarded to FastAPI, so the
    // frontend calls a same-origin URL like `/api/council/ask/stream` and no
    // CORS configuration is needed on the backend. The proxy passes the SSE
    // response through chunk by chunk rather than buffering it.
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});

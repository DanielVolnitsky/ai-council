// The JavaScript entry point that index.html loads. Its only job is to mount the
// root React component into the page: the frontend's equivalent of `main()`.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
// Importing a CSS file from TS is a Vite feature: in dev it injects a <style>
// tag (and hot-swaps it on edit), and in a build it is extracted into a .css file.
import "./styles.css";

// `!` is TypeScript's non-null assertion: getElementById returns
// `HTMLElement | null`, and we know index.html always contains #root.
const rootElement: HTMLElement = document.getElementById("root")!;

createRoot(rootElement).render(
  // StrictMode is development-only scaffolding with no effect on a build. It
  // deliberately mounts, unmounts and remounts every component once, and calls
  // render functions twice, to surface effects that don't clean up after
  // themselves (see the useEffect cleanup in useCouncilRound.ts).
  <StrictMode>
    <App />
  </StrictMode>,
);

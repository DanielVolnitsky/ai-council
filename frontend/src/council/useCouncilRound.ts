// All the state of one council round, and the function that starts a round.
//
// React's mental model in one paragraph: a component is a function from state
// to UI. You never mutate the DOM yourself. You change state, React calls the
// component function again ("re-renders"), compares the returned UI with the
// previous one, and patches only what differs. So the job here is to keep the
// round as one immutable state value, replaced on every SSE event, and let the
// components below render whatever that value currently is.
import { useEffect, useReducer, useRef } from "react";

import { streamCouncil } from "./streamCouncil";
import type { CouncilStreamEvent, CouncilSynthesis } from "./types";

export type MemberStatus = "streaming" | "done" | "failed";

export interface MemberAnswer {
  modelId: string;
  text: string;
  status: MemberStatus;
  error: string | null;
}

// "idle" until the first question is asked. "running" covers the members
// streaming and the synthesis that follows. A round ends as "done" (a synthesis
// arrived) or "failed" (an error event, a network failure, or a dropped stream).
export type RoundStatus = "idle" | "running" | "done" | "failed";

export interface RoundState {
  status: RoundStatus;
  sessionId: string | null;
  // Seeded from session_start in config order, so every panel shows before its first token.
  members: MemberAnswer[];
  synthesis: CouncilSynthesis | null;
  error: string | null;
}

// Every way the state can change. The server's stream events are wrapped
// rather than mixed in, so the reducer's two concerns stay visibly separate:
// lifecycle actions from the hook itself, and events from the backend.
type RoundAction =
  | { type: "round_started" }
  | { type: "stream_event"; event: CouncilStreamEvent }
  | { type: "stream_ended" }
  | { type: "request_failed"; message: string };

const IDLE_ROUND: RoundState = {
  status: "idle",
  sessionId: null,
  members: [],
  synthesis: null,
  error: null,
};

// A reducer is a pure function (state, action) → new state, the same pattern
// as Redux or an event-sourced aggregate's apply(). It must never mutate
// `state`: React detects a change by reference (===), so every change returns a
// new object. `{ ...state, x }` copies the object with one field replaced; it is
// the idiomatic substitute for Java's withX() methods on an immutable value.
function roundReducer(state: RoundState, action: RoundAction): RoundState {
  switch (action.type) {
    case "round_started":
      return { ...IDLE_ROUND, status: "running" };
    case "stream_event":
      return applyStreamEvent(state, action.event);
    case "stream_ended":
      // The server closes the stream right after synth_done or error. If neither
      // arrived, the connection dropped mid-round (for example, the backend restarted).
      return state.status === "running"
        ? { ...state, status: "failed", error: "The stream ended before the synthesis arrived." }
        : state;
    case "request_failed":
      return { ...state, status: "failed", error: action.message };
  }
}

function applyStreamEvent(state: RoundState, event: CouncilStreamEvent): RoundState {
  switch (event.type) {
    case "session_start":
      return {
        ...state,
        sessionId: event.session_id,
        members: event.model_ids.map(pendingMember),
      };
    case "model_token":
      return {
        ...state,
        members: upsertMember(state.members, event.model_id, (member) => ({
          ...member,
          text: member.text + event.token,
        })),
      };
    case "model_done":
      // The final `response` is authoritative, so it replaces the accumulated
      // tokens rather than being appended to them.
      return {
        ...state,
        members: upsertMember(state.members, event.model_id, (member) => ({
          ...member,
          text: event.response,
          status: event.error === null ? "done" : "failed",
          error: event.error,
        })),
      };
    case "synth_done":
      return { ...state, status: "done", synthesis: event.synthesis };
    case "error":
      return { ...state, status: "failed", error: event.message };
    // A default branch keeps the app working if the backend adds an event type
    // before the frontend learns about it.
    default:
      return state;
  }
}

function pendingMember(modelId: string): MemberAnswer {
  return { modelId, text: "", status: "streaming", error: null };
}

// Applies `update` to the member with `modelId`, creating it first if session_start
// did not list it, so an event for an unannounced member still gets a panel.
function upsertMember(
  members: MemberAnswer[],
  modelId: string,
  update: (member: MemberAnswer) => MemberAnswer,
): MemberAnswer[] {
  const exists: boolean = members.some((member) => member.modelId === modelId);
  const withMember: MemberAnswer[] = exists
    ? members
    : [...members, pendingMember(modelId)];
  return withMember.map((member) => (member.modelId === modelId ? update(member) : member));
}

// A custom hook: any function whose name starts with `use` and that calls
// React's built-in hooks. It lets a component reuse stateful logic, here the
// round state plus the network loop that feeds it, without the component
// knowing how SSE works. Hooks must be called unconditionally and in the same
// order on every render, because React identifies them by call order.
export function useCouncilRound(): { round: RoundState; ask: (question: string) => void } {
  // useReducer holds state that survives re-renders. `round` is the current
  // value, and `dispatch(action)` runs the reducer and schedules a re-render.
  const [round, dispatch] = useReducer(roundReducer, IDLE_ROUND);

  // useRef is a mutable box that survives re-renders, but changing it does NOT
  // trigger one. That fits the in-flight request's controller: plumbing the UI
  // never displays.
  const controllerRef = useRef<AbortController | null>(null);

  // useEffect runs code after rendering, to sync with things outside React. With
  // an empty dependency list it runs once on mount. The function it returns is
  // the cleanup, which React runs on unmount, like a @PreDestroy hook. Here it
  // cancels an in-flight round so the stream doesn't outlive the component.
  useEffect(() => () => controllerRef.current?.abort(), []);

  function ask(question: string): void {
    // The form is disabled while a round runs, but it re-enables on synth_done or
    // error, a moment before the server closes the stream. A question asked in that
    // gap must cancel the old request so its final events can't leak into the new round.
    controllerRef.current?.abort();
    const controller: AbortController = new AbortController();
    controllerRef.current = controller;
    dispatch({ type: "round_started" });
    // ask() is called from an event handler, which cannot itself be awaited,
    // so the async loop is started and deliberately not awaited ("fire and forget").
    void consumeStream(question, controller.signal, dispatch);
  }

  return { round, ask };
}

async function consumeStream(
  question: string,
  signal: AbortSignal,
  dispatch: (action: RoundAction) => void,
): Promise<void> {
  // No abort check is needed inside the loop. An abort comes from a click,
  // key press or unmount: a separate task on the browser's event loop. It can
  // only run once the current chain of promise callbacks has drained, and that
  // chain covers every frame from one network chunk. After the abort, the
  // generator's next reader.read() rejects, so the loop exits through `catch`.
  try {
    for await (const event of streamCouncil(question, signal)) {
      dispatch({ type: "stream_event", event });
    }
    // Normally redundant for the same reason. It's kept because the Streams spec
    // lets a browser deliver "body closed" in a later task: if the user asks
    // again in that window, this superseded round must not mark the new one failed.
    if (!signal.aborted) dispatch({ type: "stream_ended" });
  } catch (error) {
    // Aborting is how we cancel on purpose, so its AbortError is not a failure.
    if (signal.aborted) return;
    // In TS a caught value is typed `unknown`, since JS can throw anything, not only Errors.
    const message: string = error instanceof Error ? error.message : String(error);
    dispatch({ type: "request_failed", message });
  }
}

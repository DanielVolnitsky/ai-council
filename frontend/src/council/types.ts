// TypeScript mirrors of the backend's wire shapes in core/core/domain.py and
// core/core/events.py. They exist only at compile time: TS erases every type,
// so nothing here checks the JSON at runtime. If the Python models change,
// these must be updated by hand.
//
// `interface` reads much like a Java interface, but the typing is structural:
// any object with matching fields satisfies it, with no `implements` needed.

export interface Disagreement {
  point: string;
  models_for: string[];
  models_against: string[];
}

export interface ModelInsights {
  model_id: string;
  insights: string[];
}

export interface Verdict {
  strongest: string;
  weakest: string;
  justification: string;
}

export interface CouncilSynthesis {
  summary: string;
  consensus: string[];
  disagreements: Disagreement[];
  verdict: Verdict;
  unique_insights: ModelInsights[];
  blind_spots: string[];
  takeaways: string[];
}

// One event of the /api/council/ask/stream response. Each variant carries the
// SSE `event:` name in `type` plus the fields of its JSON `data:` payload.
//
// This is a "discriminated union": a closed set of shapes told apart by the
// literal `type` field, like a Java sealed interface with records. After
// `switch (event.type)` matches "model_token", the compiler narrows `event` to
// that variant, so `event.token` type-checks only inside that case.
export type CouncilStreamEvent =
  | { type: "session_start"; session_id: string; model_ids: string[] }
  | { type: "model_token"; model_id: string; token: string }
  | { type: "model_done"; model_id: string; response: string; error: string | null }
  | { type: "synth_done"; synthesis: CouncilSynthesis }
  | { type: "error"; message: string };

// The network client for one council round. It POSTs the question and turns
// the streamed response body into typed CouncilStreamEvents.
import { parseSseFrames, type SseFrame } from "./sse";
import type { CouncilStreamEvent } from "./types";

// `async function*` declares an async generator: a function that can `yield`
// many values over time and `await` between them. The caller consumes it with
// `for await (const event of streamCouncil(...))`. The closest Java analogue
// is a lazily pulled Iterator, or a Flux that only produces values on demand.
// Nothing here runs until the caller asks for the first event.
export async function* streamCouncil(
  question: string,
  // An AbortSignal is a cancellation token. When its AbortController is
  // aborted, fetch stops the request and any pending reader.read() rejects
  // with an AbortError, which ends the `for await` loop in the caller.
  signal: AbortSignal,
): AsyncGenerator<CouncilStreamEvent> {
  // The path is relative, so it goes to the Vite dev server, which proxies /api
  // to FastAPI (see vite.config.ts). The body is `{"question": ...}` because the
  // endpoint declares `Body(embed=True)`.
  const response: Response = await fetch("/api/council/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  // fetch resolves as soon as the response headers arrive, before any of the
  // body. Non-2xx statuses don't throw (e.g. a 422 for a blank question), so check
  // explicitly, and include FastAPI's error body, since it says what went wrong.
  if (!response.ok || response.body === null) {
    const detail: string = await response.text().catch(() => "");
    throw new Error(`The council request failed with HTTP ${response.status}. ${detail}`.trim());
  }

  // response.body is a ReadableStream of raw bytes. Piping it through a
  // TextDecoderStream yields text instead, and it correctly handles a
  // multi-byte UTF-8 character that is split across two chunks.
  const reader: ReadableStreamDefaultReader<string> = response.body
    .pipeThrough(new TextDecoderStream())
    .getReader();

  let buffer: string = "";
  // try/finally works in generators as it does in Java: `finally` runs however
  // the generator stops, whether it finishes, throws (e.g. JSON.parse on a
  // malformed frame), or the caller abandons the loop. Cancelling the reader
  // there closes the HTTP connection instead of downloading a stream nobody
  // reads. The cancel's own failure is irrelevant, so it is swallowed.
  try {
    while (true) {
      // read() resolves with the next chunk; `done` is true once the server closes the stream.
      const { value, done } = await reader.read();
      if (done) return;
      buffer += value;
      const { frames, rest } = parseSseFrames(buffer);
      buffer = rest;
      for (const frame of frames) {
        yield toCouncilEvent(frame);
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

function toCouncilEvent(frame: SseFrame): CouncilStreamEvent {
  // Spreads the JSON payload's fields next to the event name, so
  // `{"model_id":..,"token":..}` becomes `{type:"model_token", model_id, token}`.
  // `as` is an unchecked cast, like a Java downcast with no runtime check:
  // we trust the backend's contract. An unknown event type would slip through
  // here and is ignored by the reducer's switch.
  return { type: frame.event, ...JSON.parse(frame.data) } as CouncilStreamEvent;
}

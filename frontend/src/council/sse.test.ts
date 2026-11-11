// Vitest is a Jest-compatible test runner built on Vite: `npm test` finds every
// *.test.ts file and runs it in Node. `describe`/`it` group tests much like
// JUnit's @Nested and @Test, and `expect(...).toEqual(...)` compares deeply,
// like assertEquals on records.
import { describe, expect, it } from "vitest";

import { parseSseFrames, type SseParseResult } from "./sse";

describe("parseSseFrames", () => {
  it("parses complete frames and keeps the unterminated tail as rest", () => {
    const buffer: string =
      'event: session_start\ndata: {"session_id":"s1"}\n\n' +
      'event: model_token\ndata: {"model_id":"m","token":"Hi"}\n\n' +
      "event: model_do";

    expect(parseSseFrames(buffer)).toEqual({
      frames: [
        { event: "session_start", data: '{"session_id":"s1"}' },
        { event: "model_token", data: '{"model_id":"m","token":"Hi"}' },
      ],
      rest: "event: model_do",
    });
  });

  it("reassembles a frame split across two network chunks", () => {
    const first: SseParseResult = parseSseFrames("event: error\nda");
    const second: SseParseResult = parseSseFrames(first.rest + 'ta: {"message":"boom"}\n\n');

    expect([first, second]).toEqual([
      { frames: [], rest: "event: error\nda" },
      { frames: [{ event: "error", data: '{"message":"boom"}' }], rest: "" },
    ]);
  });

  it("joins multi-line data, skips comments, and accepts CRLF line endings", () => {
    const buffer: string = ": keep-alive\r\nevent: x\r\ndata: a\r\ndata:b\r\n\r\n";

    expect(parseSseFrames(buffer)).toEqual({
      frames: [{ event: "x", data: "a\nb" }],
      rest: "",
    });
  });

  it("defaults the event name to message", () => {
    expect(parseSseFrames("data: 1\n\n")).toEqual({
      frames: [{ event: "message", data: "1" }],
      rest: "",
    });
  });
});

// A minimal Server-Sent Events parser.
//
// Browsers ship EventSource for SSE, but it can only issue GET requests, and
// our stream endpoint is a POST with a JSON body. So we read the response
// body ourselves and parse the text format, which is simple:
//
//   event: model_token
//   data: {"model_id":"openai:gpt-4o","token":"Hel"}
//   <blank line>
//
// Frames are separated by a blank line. Each line is `field: value`, and
// several `data:` lines in one frame are joined with "\n".
//
// The network hands us arbitrary chunks, so one frame may be split across two
// chunks and one chunk may hold several frames. The caller therefore keeps a
// buffer, and this function returns the complete frames together with the
// unfinished remainder to carry into the next call. It is kept pure (string in,
// data out) so it can be unit tested without a network (see sse.test.ts).

export interface SseFrame {
  event: string;
  data: string;
}

export interface SseParseResult {
  frames: SseFrame[];
  rest: string;
}

export function parseSseFrames(buffer: string): SseParseResult {
  // The spec allows \r\n line endings, so normalise them before splitting on "\n\n".
  const normalized: string = buffer.replace(/\r\n/g, "\n");
  const blocks: string[] = normalized.split("\n\n");
  // The last block has not been terminated by a blank line yet: either it is
  // empty (the buffer ended exactly on a frame boundary) or it is a partial
  // frame whose remainder is still in flight.
  const rest: string = blocks.pop() ?? "";
  const frames: SseFrame[] = blocks
    .filter((block) => block.trim() !== "")
    .map(parseFrame);
  return { frames, rest };
}

function parseFrame(block: string): SseFrame {
  // "message" is the event name the SSE spec assigns when a frame has no `event:` line.
  let event: string = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    // A line starting with ":" is a comment, which servers use as a keep-alive.
    if (line.startsWith(":")) continue;
    const colon: number = line.indexOf(":");
    const field: string = colon === -1 ? line : line.slice(0, colon);
    let value: string = colon === -1 ? "" : line.slice(colon + 1);
    // The spec drops exactly one space after the colon, if present.
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  return { event, data: dataLines.join("\n") };
}

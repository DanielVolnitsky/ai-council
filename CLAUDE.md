# Contribution Rules

## Python Code Quality Preferences

### Naming
- No synonyms for the same concept across the codebase — pick one word and use it everywhere (e.g. "discussion", not "discussion" in some places and "thread" in others).
- Prefer descriptive domain names over generic ones: OutputDiscussion / RawDiscussion over Thread / dict; build_output_discussion over build_thread.
- Use the Output / Raw prefix convention to distinguish between incoming API payloads and outgoing records.
- Name local variables after their type prefix when it adds clarity: raw_discussions, output_discussions, output_notes.
- Use the domain vocabulary consistently: in this codebase "agent" not "bot".

### Types
- Annotate every local variable binding with its type, even when the type is already inferable: `chat_model: BaseChatModel = build_chat_model(model_config)`, not `chat_model = build_chat_model(model_config)`. The type is then readable at the use site without chasing the callee's signature. The only exclusions are bindings Python cannot annotate inline: `for` targets, comprehension variables, `with ... as`, and `except ... as`.
- No bare list[dict] or dict in function signatures or local annotations — every dict boundary gets a named TypedDict.
- Use NamedTuple for small value objects returned from internal helpers (e.g. subprocess results) — named fields beat positional tuple[int, str, str].
- Avoid redundant wrapper types: if a TypedDict would only name 1–2 fields of a large external payload and add no real signal, keep dict rather than creating noise.

### Functions
- When a function body is a simple if / else over two distinct paths, extract each path into its own private function and make the public function a one-line router.

### Tests
- When asserting the shape of a dict or list of dicts, use a single equality assertion against the full expected structure rather than multiple field-by-field assertions — it catches all fields at once (including ones the piecemeal version silently ignores) and reads as a clear input → output example.

## Git Commit Pattern

When committing changes, use the following format:

**Format:** `<one-sentence change description describing what, why, and how>`
 
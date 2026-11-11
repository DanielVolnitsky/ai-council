// The question card from the reference screenshot: a borderless textarea and a
// round send button. Enter sends, and Shift+Enter inserts a newline.
import { useState, type FormEvent, type KeyboardEvent } from "react";

interface QuestionFormProps {
  busy: boolean;
  // Callback props are how a child talks to its parent: the parent passes a
  // function down, and the child calls it. Data flows down via props, and
  // events flow up via callbacks.
  onAsk: (question: string) => void;
}

export function QuestionForm({ busy, onAsk }: QuestionFormProps) {
  // useState is the single-value form of useReducer. This is a "controlled
  // input": React state is the source of truth for the textarea's text, the
  // textarea displays `question`, and every keystroke calls setQuestion,
  // which re-renders it. Reading the text later is then just reading state.
  const [question, setQuestion] = useState<string>("");

  // Mirrors the backend's Question type (a non-blank string), so an invalid
  // request is never sent and the 422 path is never hit from the UI.
  const canSubmit: boolean = !busy && question.trim() !== "";

  function submit(): void {
    if (canSubmit) onAsk(question);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    // A native <form> submit navigates the page; preventDefault keeps us in the SPA.
    event.preventDefault();
    submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    // isComposing is true while an IME (e.g. for Japanese input) is mid-word,
    // when Enter confirms the composed word rather than meaning "send".
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form className="question-card" onSubmit={handleSubmit}>
      <textarea
        className="question-input"
        placeholder="Ask anything…"
        // A placeholder vanishes once typing starts, so it can't serve as the field's accessible name.
        aria-label="Your question"
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={handleKeyDown}
        rows={5}
        autoFocus
      />
      <div className="question-toolbar">
        <span className="question-hint">Enter to send · Shift+Enter for a new line</span>
        <button
          type="submit"
          className="send-button"
          disabled={!canSubmit}
          aria-label={busy ? "The council is answering" : "Ask the council"}
        >
          {/* A ternary picks one of two elements: a spinner while busy, otherwise the arrow. */}
          {busy ? <span className="spinner" /> : <ArrowUpIcon />}
        </button>
      </div>
    </form>
  );
}

// Inline SVG instead of an icon library: one icon doesn't justify a dependency.
// `currentColor` makes the arrow take the button's text color.
function ArrowUpIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 19V5M5 12l7-7 7 7"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// The root component: the page layout, which wires the question form to the
// round state and renders the results beneath it.
import { MemberPanel } from "./components/MemberPanel";
import { QuestionForm } from "./components/QuestionForm";
import { SynthesisView } from "./components/SynthesisView";
import { useCouncilRound, type RoundState } from "./council/useCouncilRound";

// Matches LANGFUSE_INIT_PROJECT_ID in docker-compose.yml. The link only works
// while the local Langfuse stack is running.
const LANGFUSE_SESSIONS_URL: string = "http://localhost:3000/project/ai-council/sessions";

export function App() {
  // Every SSE event dispatched inside the hook re-runs this function with the
  // new `round`, and React updates only the DOM that actually changed.
  const { round, ask } = useCouncilRound();

  return (
    <main className="page">
      <header className="hero">
        <h1>
          AI Council <span className="badge">beta</span>
        </h1>
        <p>Ask several models at once, and let a synthesizer fuse their answers into one analysis.</p>
      </header>

      <QuestionForm busy={round.status === "running"} onAsk={ask} />
      <p className="disclaimer">
        Responses are AI-generated and can be inaccurate. Review all outputs before relying on them.
      </p>

      {round.status !== "idle" && <RoundView round={round} />}
    </main>
  );
}

function RoundView({ round }: { round: RoundState }) {
  return (
    <div className="round">
      {/* role="alert" makes screen readers announce the banner as soon as it appears. */}
      {round.error && (
        <div className="error-banner" role="alert">
          {round.error}
        </div>
      )}

      <section>
        <h2 className="section-title">Council</h2>
        {round.members.length === 0 ? (
          <p className="muted">
            {round.status === "running" ? "Waiting for the first answer…" : "No answers."}
          </p>
        ) : (
          <div className="member-grid">
            {/* The model id is the stable key, so each panel keeps its identity while it streams. */}
            {round.members.map((member) => (
              <MemberPanel key={member.modelId} member={member} />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="section-title">Synthesis</h2>
        <SynthesisSlot round={round} />
      </section>

      {round.sessionId && (
        <p className="session-id">
          Session{" "}
          <a href={`${LANGFUSE_SESSIONS_URL}/${round.sessionId}`} target="_blank" rel="noreferrer">
            {round.sessionId}
          </a>
        </p>
      )}
    </div>
  );
}

function SynthesisSlot({ round }: { round: RoundState }) {
  if (round.synthesis) return <SynthesisView synthesis={round.synthesis} />;
  if (round.status !== "running") return <p className="muted">No synthesis for this round.</p>;
  // The stream never says how many members a round has, so "synthesizing" is
  // inferred from every member seen so far having finished. A member that has
  // not sent its first token yet is invisible here, so the label can show a
  // moment early.
  const synthesizing: boolean =
    round.members.length > 0 && round.members.every((member) => member.status !== "streaming");
  // aria-live="polite" has screen readers read out the label when it changes,
  // once they finish what they are currently saying.
  return (
    <div className="panel synthesis-pending" aria-live="polite">
      <span className="spinner" />
      {synthesizing ? "Synthesizing the council's answers…" : "Waiting for every member to finish…"}
    </div>
  );
}

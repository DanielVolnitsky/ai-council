// One council member's answer. It fills in token by token while streaming and
// settles into "done" or "failed" when its model_done event arrives.
import Markdown from "react-markdown";

import type { MemberAnswer, MemberStatus } from "../council/useCouncilRound";
import { ModelName } from "./ModelName";

interface MemberPanelProps {
  member: MemberAnswer;
}

// A lookup table instead of a chain of ifs. `Record<K, V>` is TS's Map-like
// object type; keying it by the MemberStatus union makes the compiler reject
// the table if a status is ever added without a label.
const STATUS_LABELS: Record<MemberStatus, string> = {
  streaming: "Answering",
  done: "Done",
  failed: "Failed",
};

export function MemberPanel({ member }: MemberPanelProps) {
  return (
    <article className="panel member-panel">
      <header className="panel-header">
        <ModelName modelId={member.modelId} />
        <span className="status-pill" data-status={member.status}>
          {STATUS_LABELS[member.status]}
        </span>
      </header>
      <div className="panel-body">
        <MemberBody member={member} />
      </div>
    </article>
  );
}

// An early `return` per case keeps the three states readable. A component may
// return different elements on different renders, and React reconciles the change.
function MemberBody({ member }: MemberPanelProps) {
  if (member.status === "failed") {
    return <p className="member-error">{member.error}</p>;
  }
  // A model can succeed with an empty reply, so "empty" only means "thinking" while still streaming.
  if (member.text === "") {
    return <p className="muted">{member.status === "streaming" ? "Thinking…" : "No answer."}</p>;
  }
  // react-markdown parses the Markdown into React elements (it never uses
  // innerHTML, so model output cannot inject script tags). It re-parses the
  // whole text on every token, which is cheap at answer-sized text.
  return (
    <div className="markdown">
      <Markdown>{member.text}</Markdown>
    </div>
  );
}

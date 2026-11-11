// The synthesizer's structured document: the seven CouncilSynthesis sections.
import type { ReactNode } from "react";
import Markdown from "react-markdown";

import type { CouncilSynthesis } from "../council/types";
import { ModelName } from "./ModelName";

interface SynthesisViewProps {
  synthesis: CouncilSynthesis;
}

export function SynthesisView({ synthesis }: SynthesisViewProps) {
  return (
    <article className="panel synthesis">
      <Section title="Summary">
        <div className="markdown">
          <Markdown>{synthesis.summary}</Markdown>
        </div>
      </Section>

      <Section title="Verdict">
        <div className="verdict-grid">
          <div className="verdict-card" data-kind="strongest">
            <span className="verdict-label">Strongest</span>
            <ModelName modelId={synthesis.verdict.strongest} />
          </div>
          <div className="verdict-card" data-kind="weakest">
            <span className="verdict-label">Weakest</span>
            <ModelName modelId={synthesis.verdict.weakest} />
          </div>
        </div>
        <p className="verdict-justification">{synthesis.verdict.justification}</p>
      </Section>

      <BulletSection title="Consensus" items={synthesis.consensus} />

      <Section title="Disagreements">
        {synthesis.disagreements.length === 0 ? (
          <p className="muted">None noted.</p>
        ) : (
          <ul className="plain-list">
            {/*
              Rendering a list means mapping data to elements. React needs a
              `key` on each item to match items between renders; the array
              index works here because this list never reorders.
            */}
            {synthesis.disagreements.map((disagreement, index) => (
              <li key={index} className="disagreement">
                <p>{disagreement.point}</p>
                <ModelList label="For" modelIds={disagreement.models_for} />
                <ModelList label="Against" modelIds={disagreement.models_against} />
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Unique insights">
        {synthesis.unique_insights.length === 0 ? (
          <p className="muted">None noted.</p>
        ) : (
          synthesis.unique_insights.map((group, index) => (
            // The synthesizer is an LLM and nothing validates that it lists each
            // model once, so the index is mixed in to keep keys unique even if it repeats one.
            <div key={`${group.model_id}-${index}`} className="insight-group">
              <ModelName modelId={group.model_id} />
              <ul>
                {group.insights.map((insight, index) => (
                  <li key={index}>{insight}</li>
                ))}
              </ul>
            </div>
          ))
        )}
      </Section>

      <BulletSection title="Blind spots" items={synthesis.blind_spots} />

      {/* Takeaways are "when applicable" per the synthesis prompt, so an empty list hides the section. */}
      {synthesis.takeaways.length > 0 && (
        <Section title="Actionable takeaways">
          <ol className="takeaways">
            {synthesis.takeaways.map((takeaway, index) => (
              <li key={index}>{takeaway}</li>
            ))}
          </ol>
        </Section>
      )}
    </article>
  );
}

// `children` is the special prop holding whatever JSX is nested between a
// component's tags, like <Section title="x">…this…</Section>. ReactNode is the
// type of "anything renderable": elements, strings, arrays, null.
function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="synthesis-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function BulletSection({ title, items }: { title: string; items: string[] }) {
  return (
    <Section title={title}>
      {items.length === 0 ? (
        <p className="muted">None noted.</p>
      ) : (
        <ul>
          {items.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      )}
    </Section>
  );
}

function ModelList({ label, modelIds }: { label: string; modelIds: string[] }) {
  if (modelIds.length === 0) return null;
  return (
    <div className="model-list">
      <span className="model-list-label">{label}</span>
      {modelIds.map((modelId, index) => (
        <ModelName key={`${modelId}-${index}`} modelId={modelId} />
      ))}
    </div>
  );
}

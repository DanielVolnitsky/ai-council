// Renders a config.yaml model id like "openai:gpt-4o" as a small provider tag
// plus the model name. It's used wherever a model is named: panel headers, the
// verdict, and disagreement attributions.

// A component's inputs ("props") are one object, declared like a parameter
// type. Components are plain functions; the capitalised name is what tells
// JSX to treat <ModelName /> as a component rather than an HTML tag.
interface ModelNameProps {
  modelId: string;
}

export function ModelName({ modelId }: ModelNameProps) {
  // Splits on the first ":" only, since model names may contain colons themselves.
  const separator: number = modelId.indexOf(":");
  const provider: string = separator === -1 ? "" : modelId.slice(0, separator);
  const model: string = separator === -1 ? modelId : modelId.slice(separator + 1);

  // JSX is syntax sugar for function calls building a UI tree. `{...}` embeds a
  // JS expression, and `className` is used because `class` is reserved in JS.
  // `cond && <X />` renders X only when cond is truthy: React skips false/null.
  // The data-provider attribute lets styles.css color the tag per provider.
  return (
    <span className="model-name" title={modelId}>
      {provider && (
        <span className="provider-tag" data-provider={provider}>
          {provider}
        </span>
      )}
      <span>{model}</span>
    </span>
  );
}

import { useStore } from "./store";
import { Layout } from "./components/Layout";
import { TextDeck } from "./components/TextDeck";
import { OutputDeck } from "./components/OutputDeck";
import { EmotionDJPanel } from "./components/EmotionDJPanel";

export default function App() {
  const error = useStore((s) => s.error);
  return (
    <Layout
      header={
        <div className="flex items-baseline justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-booth-ink">
              Emotion DJ Booth
              <span className="ml-2 text-[10px] uppercase tracking-[0.25em] text-booth-accent">
                latent · 64-D
              </span>
            </h1>
            <p className="text-[11px] text-booth-muted">
              Mix subtle emotional directions before rewriting — not select labels.
            </p>
          </div>
          {error && (
            <div className="text-[11px] text-booth-bad bg-booth-bad/10 border border-booth-bad/30 rounded px-2 py-1 max-w-md truncate">
              {error}
            </div>
          )}
        </div>
      }
      source={<TextDeck />}
      output={<OutputDeck />}
      panel={<EmotionDJPanel />}
    />
  );
}

import { useStore } from "../store";
import { useDebouncedCallback } from "../hooks/useDebouncedCallback";

export function TextDeck() {
  const sourceText = useStore((s) => s.sourceText);
  const setSourceText = useStore((s) => s.setSourceText);
  const analyze = useStore((s) => s.analyze);
  const status = useStore((s) => s.status);

  // Debounced analyze fires only when the user actually edits the text.
  // (Don't drive this from useEffect on sourceText — every store update
  // re-renders this component and would re-arm the timer, causing an
  // infinite analyze→rewrite loop.)
  const debouncedAnalyze = useDebouncedCallback(() => {
    void analyze();
  }, 800);

  return (
    <div className="h-full rounded-2xl bg-booth-panel border border-booth-edge/60 shadow-booth flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-booth-edge/60 flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-widest text-booth-muted">
            Source deck
          </div>
          <div className="text-sm text-booth-ink">Draft text</div>
        </div>
        <button
          onClick={() => void analyze()}
          disabled={status !== "idle" || !sourceText.trim()}
          className="text-xs px-3 py-1.5 rounded-md bg-booth-accent/20 hover:bg-booth-accent/30 text-booth-ink border border-booth-accent/40 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          {status === "analyzing" ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      <textarea
        value={sourceText}
        onChange={(e) => {
          setSourceText(e.target.value);
          if (e.target.value.trim().length > 0) debouncedAnalyze();
        }}
        placeholder="Type or paste a draft. The system will estimate its 64-dim latent emotion code and rewrite it as you mix."
        className="flex-1 w-full p-4 bg-transparent text-booth-ink placeholder:text-booth-muted/70 outline-none resize-none leading-relaxed"
      />
      <div className="px-4 py-2 text-[11px] text-booth-muted border-t border-booth-edge/60">
        Auto-analyzes 800 ms after you stop typing.
      </div>
    </div>
  );
}

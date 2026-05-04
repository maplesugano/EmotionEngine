import { useState } from "react";
import { useStore } from "../store";
import { DiffViewer } from "./DiffViewer";

export function OutputDeck() {
  const rewritten = useStore((s) => s.rewrittenText);
  const diff = useStore((s) => s.diff);
  const status = useStore((s) => s.status);
  const [showDiff, setShowDiff] = useState(true);
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(rewritten);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="h-full rounded-2xl bg-booth-panel border border-booth-edge/60 shadow-booth flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-booth-edge/60 flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-widest text-booth-muted">
            Output deck
          </div>
          <div className="text-sm text-booth-ink">
            Rewritten via 64-D latent edit
          </div>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-[11px] text-booth-muted">
            <input
              type="checkbox"
              checked={showDiff}
              onChange={(e) => setShowDiff(e.target.checked)}
              className="accent-booth-accent"
            />
            diff
          </label>
          <button
            disabled={!rewritten}
            onClick={onCopy}
            className="text-xs px-3 py-1.5 rounded-md bg-booth-edge hover:bg-booth-panel2 text-booth-ink border border-booth-edge disabled:opacity-40 transition"
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
      </div>

      <div className="flex-1 p-4 overflow-auto scrollbar-thin">
        {status === "rewriting" && (
          <div className="absolute mt-1 ml-1 text-[11px] text-booth-accent2 animate-pulse">
            rewriting…
          </div>
        )}
        {!rewritten ? (
          <div className="text-booth-muted text-sm italic">
            Output will appear here after the first analyze.
          </div>
        ) : showDiff ? (
          <DiffViewer diff={diff} />
        ) : (
          <div className="text-sm leading-relaxed whitespace-pre-wrap text-booth-ink">
            {rewritten}
          </div>
        )}
      </div>
    </div>
  );
}

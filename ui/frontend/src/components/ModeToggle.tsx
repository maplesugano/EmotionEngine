import type { Mode } from "../types";

export type UIMode = "normal" | "research" | "discovery";

export function ModeToggle({
  uiMode,
  setUIMode,
  rewriteMode,
  setRewriteMode,
}: {
  uiMode: UIMode;
  setUIMode: (m: UIMode) => void;
  rewriteMode: Mode;
  setRewriteMode: (m: Mode) => void;
}) {
  return (
    <div className="flex flex-col sm:flex-row gap-3">
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-booth-muted">
          UI mode
        </div>
        <div className="flex rounded-lg overflow-hidden border border-booth-edge">
          {(["normal", "research", "discovery"] as UIMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setUIMode(m)}
              className={
                "px-3 py-1.5 text-xs capitalize transition " +
                (uiMode === m
                  ? "bg-booth-accent/30 text-booth-ink"
                  : "bg-booth-panel2 text-booth-muted hover:text-booth-ink")
              }
            >
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-booth-muted">
          Rewrite intensity
        </div>
        <div className="flex rounded-lg overflow-hidden border border-booth-edge">
          {(["subtle", "balanced", "strong"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setRewriteMode(m)}
              className={
                "px-3 py-1.5 text-xs capitalize transition " +
                (rewriteMode === m
                  ? "bg-booth-accent2/30 text-booth-ink"
                  : "bg-booth-panel2 text-booth-muted hover:text-booth-ink")
              }
            >
              {m}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

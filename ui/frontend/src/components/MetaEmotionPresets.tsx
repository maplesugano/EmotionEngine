import { useStore } from "../store";
import { PRESETS } from "../types";

export function MetaEmotionPresets() {
  const preset = useStore((s) => s.preset);
  const status = useStore((s) => s.status);

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-booth-ink">
        Meta-emotion presets
      </h3>
      <div className="grid grid-cols-2 gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            disabled={status !== "idle"}
            onClick={() => void preset(p.id)}
            className="text-left px-3 py-2 rounded-lg bg-booth-panel2 hover:bg-booth-edge border border-booth-edge/60 hover:border-booth-accent/60 transition disabled:opacity-50"
          >
            <div className="text-xs text-booth-ink font-medium">{p.label}</div>
            <div className="text-[10px] text-booth-muted">{p.hint}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

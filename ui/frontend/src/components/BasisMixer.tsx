import { useEffect, useState } from "react";
import { useStore } from "../store";
import { useDebouncedCallback } from "../hooks/useDebouncedCallback";

const KNOWN: Record<number, string> = {
  7: "addressivity / detachment",
  10: "self-observation",
  12: "indecision",
};

function labelFor(i: number): string {
  return KNOWN[i] ?? "latent component";
}

export function BasisMixer() {
  const basis = useStore((s) => s.basisVector);
  const setBasisComponent = useStore((s) => s.setBasisComponent);
  const excluded = useStore((s) => s.excludedComponents);
  const excludedSet = new Set(excluded);
  const [open, setOpen] = useState(false);

  // local mirror so dragging feels instant
  const [local, setLocal] = useState(basis);
  useEffect(() => setLocal(basis), [basis]);

  const debounced = useDebouncedCallback((i: number, v: number) => {
    void setBasisComponent(i, v);
  }, 500);

  return (
    <div className="space-y-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between text-left px-3 py-2 rounded-lg bg-booth-panel2 hover:bg-booth-edge border border-booth-edge/60 transition"
      >
        <div>
          <div className="text-sm font-semibold text-booth-ink">
            Advanced 64-basis mixer
          </div>
          <div className="text-[10px] text-booth-muted">
            Edit individual latent coefficients
          </div>
        </div>
        <span className="text-booth-muted">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="space-y-2">
          <div className="text-[11px] text-booth-warm bg-booth-warm/10 border border-booth-warm/30 rounded p-2 leading-snug">
            ⚠ Advanced: these basis components are distributed and not
            one-to-one emotions. Real emotions emerge from dense mixtures
            across all 64 dimensions.
            {excluded.length > 0 && (
              <span className="block mt-1 text-booth-muted">
                {excluded.length} axes greyed out: flagged as repetition /
                dirty-language drivers by
                <code className="mx-1">eval_basis_pathology</code>.
              </span>
            )}
          </div>

          <div className="grid grid-cols-8 sm:grid-cols-12 md:grid-cols-16 gap-1">
            {local.map((v, i) => {
              const isKnown = i in KNOWN;
              const isExcluded = excludedSet.has(i);
              return (
                <div
                  key={i}
                  className={
                    "flex flex-col items-center gap-0.5 group " +
                    (isExcluded ? "opacity-40" : "")
                  }
                  title={
                    `b${String(i + 1).padStart(2, "0")} — ${labelFor(i)}\n` +
                    `weight ${v.toFixed(3)}` +
                    (isExcluded
                      ? "\n⚠ disabled: pathological axis (loops / dirty text)"
                      : "")
                  }
                >
                  <input
                    type="range"
                    min={-1}
                    max={1}
                    step={0.01}
                    value={v}
                    disabled={isExcluded}
                    onChange={(e) => {
                      const nv = parseFloat(e.target.value);
                      const next = [...local];
                      next[i] = nv;
                      setLocal(next);
                      debounced(i, nv);
                    }}
                    className={
                      "w-full h-1 cursor-pointer " +
                      (isExcluded
                        ? "accent-booth-muted cursor-not-allowed"
                        : "accent-booth-accent")
                    }
                  />
                  <span
                    className={
                      "text-[9px] tabular-nums " +
                      (isExcluded
                        ? "text-booth-bad/70 line-through"
                        : isKnown
                          ? "text-booth-accent"
                          : "text-booth-muted")
                    }
                  >
                    b{String(i + 1).padStart(2, "0")}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

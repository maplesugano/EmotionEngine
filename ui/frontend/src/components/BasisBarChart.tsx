import { useStore } from "../store";

const KNOWN: Record<number, string> = {
  7: "addressivity / detachment",
  10: "self-observation",
  12: "indecision",
};

function labelFor(i: number): string {
  return KNOWN[i] ?? "latent component";
}

export function BasisBarChart() {
  const top = useStore((s) => s.topBasisComponents);
  const excluded = useStore((s) => s.excludedComponents);
  const excludedSet = new Set(excluded);
  if (!top.length) {
    return (
      <div className="text-[11px] text-booth-muted italic">
        Top basis weights will appear after analyze.
      </div>
    );
  }
  const max = Math.max(...top.map((c) => Math.abs(c.weight)), 0.001);
  return (
    <div className="space-y-1">
      <div className="text-[11px] uppercase tracking-wider text-booth-muted">
        Top 10 basis weights
      </div>
      <ul className="space-y-1">
        {top.map((c) => {
          const w = c.weight;
          const pct = (Math.abs(w) / max) * 50;
          const isPos = w >= 0;
          const isExcluded = excludedSet.has(c.index);
          return (
            <li
              key={c.index}
              className={
                "flex items-center gap-2 text-[11px] " +
                (isExcluded ? "opacity-40" : "")
              }
              title={
                isExcluded
                  ? "pathological axis — zeroed by backend"
                  : undefined
              }
            >
              <span
                className={
                  "w-10 tabular-nums " +
                  (isExcluded
                    ? "text-booth-bad/80 line-through"
                    : "text-booth-muted")
                }
              >
                b{String(c.index + 1).padStart(2, "0")}
              </span>
              <div className="relative flex-1 h-2 bg-booth-panel2 rounded">
                <div className="absolute inset-y-0 left-1/2 w-px bg-booth-edge" />
                <div
                  className={
                    "absolute inset-y-0 " +
                    (isPos
                      ? "left-1/2 bg-booth-good/70"
                      : "right-1/2 bg-booth-bad/70")
                  }
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="w-12 tabular-nums text-right text-booth-muted">
                {w.toFixed(2)}
              </span>
              <span className="w-40 truncate text-booth-muted">
                {labelFor(c.index)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import type { AxisLabelEntry, Projection } from "../types";

const SIZE = 240; // px

function describeAxis(entries: AxisLabelEntry[] | undefined, fallback: string): {
  short: string;
  detail: string;
} {
  if (!entries || entries.length === 0) {
    return { short: fallback, detail: fallback };
  }
  const top = entries[0];
  const short = top.phrase;
  const detail = entries
    .map((e) => `b${String(e.index).padStart(2, "0")} ${e.phrase} (${e.weight.toFixed(2)})`)
    .join("\n");
  return { short, detail };
}

export function LatentMap() {
  const projection = useStore((s) => s.projection);
  const setProjection = useStore((s) => s.setProjection);
  const axisLabels = useStore((s) => s.axisLabels);
  const [drag, setDrag] = useState<Projection | null>(null);
  const ref = useRef<SVGSVGElement | null>(null);

  const labels = useMemo(
    () => ({
      top: describeAxis(axisLabels?.pos_y, "activated"),
      bottom: describeAxis(axisLabels?.neg_y, "calm"),
      left: describeAxis(axisLabels?.neg_x, "inward"),
      right: describeAxis(axisLabels?.pos_x, "expressive"),
    }),
    [axisLabels],
  );

  const point = drag ?? projection;
  const cx = (point.x * 0.5 + 0.5) * SIZE;
  const cy = (1 - (point.y * 0.5 + 0.5)) * SIZE;

  // Pointer drag handling.
  useEffect(() => {
    if (!drag) return;
    const onMove = (e: PointerEvent) => {
      const rect = ref.current?.getBoundingClientRect();
      if (!rect) return;
      const px = (e.clientX - rect.left) / rect.width;
      const py = (e.clientY - rect.top) / rect.height;
      const x = Math.max(-1, Math.min(1, (px - 0.5) * 2));
      const y = Math.max(-1, Math.min(1, ((1 - py) - 0.5) * 2));
      setDrag({ x, y });
    };
    const onUp = () => {
      if (drag) void setProjection(drag);
      setDrag(null);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp, { once: true });
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [drag, setProjection]);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-booth-ink">
          Latent emotion map
        </h3>
        <span className="text-[11px] text-booth-muted">
          drag to move through emotion space
        </span>
      </div>
      <div
        className="relative mx-auto select-none"
        style={{ width: SIZE, height: SIZE }}
      >
        {/* axis labels — dynamic top contributors per direction */}
        <div
          className="absolute inset-x-0 -top-1 text-center text-[10px] text-booth-muted truncate"
          title={labels.top.detail}
        >
          {labels.top.short}
        </div>
        <div
          className="absolute inset-x-0 -bottom-4 text-center text-[10px] text-booth-muted truncate"
          title={labels.bottom.detail}
        >
          {labels.bottom.short}
        </div>
        <div
          className="absolute -left-2 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-booth-muted whitespace-nowrap origin-center"
          title={labels.left.detail}
        >
          {labels.left.short}
        </div>
        <div
          className="absolute -right-2 top-1/2 -translate-y-1/2 rotate-90 text-[10px] text-booth-muted whitespace-nowrap origin-center"
          title={labels.right.detail}
        >
          {labels.right.short}
        </div>

        <svg
          ref={ref}
          width={SIZE}
          height={SIZE}
          className="rounded-xl bg-booth-panel2 border border-booth-edge cursor-crosshair"
          onPointerDown={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const px = (e.clientX - rect.left) / rect.width;
            const py = (e.clientY - rect.top) / rect.height;
            const x = (px - 0.5) * 2;
            const y = ((1 - py) - 0.5) * 2;
            setDrag({ x, y });
          }}
        >
          {/* gridlines */}
          <line
            x1={SIZE / 2}
            x2={SIZE / 2}
            y1={0}
            y2={SIZE}
            stroke="currentColor"
            className="text-booth-edge"
            strokeDasharray="4 4"
          />
          <line
            x1={0}
            x2={SIZE}
            y1={SIZE / 2}
            y2={SIZE / 2}
            stroke="currentColor"
            className="text-booth-edge"
            strokeDasharray="4 4"
          />
          {/* halo */}
          <circle
            cx={cx}
            cy={cy}
            r={18}
            className="fill-booth-accent/10"
          />
          <circle
            cx={cx}
            cy={cy}
            r={7}
            className="fill-booth-accent stroke-booth-ink"
            strokeWidth={1.5}
            style={{
              transition: drag ? "none" : "cx 200ms ease, cy 200ms ease",
            }}
          />
        </svg>
        <div className="mt-1 text-center text-[10px] text-booth-muted tabular-nums">
          x {point.x.toFixed(2)} · y {point.y.toFixed(2)}
        </div>
      </div>
    </div>
  );
}

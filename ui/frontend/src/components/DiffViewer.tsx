import type { DiffSegment } from "../types";

export function DiffViewer({ diff }: { diff: DiffSegment[] }) {
  if (!diff.length) return null;
  return (
    <div className="text-sm leading-relaxed whitespace-pre-wrap">
      {diff.map((seg, i) => {
        if (seg.type === "same") {
          return (
            <span key={i} className="text-booth-ink">
              {seg.text}
            </span>
          );
        }
        if (seg.type === "added") {
          return (
            <span
              key={i}
              className="bg-booth-good/15 text-booth-good rounded px-0.5"
            >
              {seg.text}
            </span>
          );
        }
        return (
          <span
            key={i}
            className="text-booth-muted line-through decoration-booth-bad/60"
          >
            {seg.text}
          </span>
        );
      })}
    </div>
  );
}

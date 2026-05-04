import { useStore } from "../store";
import { MACRO_EMOTIONS, type MacroEmotion } from "../types";
import { useDebouncedCallback } from "../hooks/useDebouncedCallback";
import { useState, useEffect } from "react";

export function MacroSliders() {
  const macro = useStore((s) => s.macroEmotions);
  const setMacro = useStore((s) => s.setMacro);

  // local copy so the slider feels instant; flush to backend after 500ms idle
  const [local, setLocal] = useState(macro);
  useEffect(() => setLocal(macro), [macro]);

  const debouncedFlush = useDebouncedCallback(
    (name: MacroEmotion, value: number) => {
      void setMacro(name, value);
    },
    500,
  );

  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-booth-ink">
          Macro emotion projection
        </h3>
        <p className="text-[11px] text-booth-muted leading-snug max-w-md">
          These are projections from the 64-dimensional latent basis, not
          independent controls. Moving one slider nudges the latent code along
          a low-norm preimage, which shifts the others.
        </p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-2">
        {MACRO_EMOTIONS.map((name) => (
          <SliderRow
            key={name}
            name={name}
            value={local[name]}
            onChange={(v) => {
              setLocal((s) => ({ ...s, [name]: v }));
              debouncedFlush(name, v);
            }}
          />
        ))}
      </div>
    </div>
  );
}

function SliderRow({
  name,
  value,
  onChange,
}: {
  name: MacroEmotion;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <div className="flex justify-between text-[11px]">
        <span className="capitalize text-booth-ink">{name}</span>
        <span className="tabular-nums text-booth-muted">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        className="dj"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

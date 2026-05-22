import { create } from "zustand";
import {
  analyzeText,
  applyPreset,
  getMeta,
  rewriteText,
  updateFromMacro,
  updateFromProjection,
} from "./api/client";
import type {
  AxisLabels,
  DiffSegment,
  EmotionState,
  MacroEmotion,
  MacroEmotions,
  Mode,
  Preset,
  Projection,
  TopBasisComponent,
} from "./types";
import { ZERO_BASIS, ZERO_MACRO } from "./types";

type Status = "idle" | "analyzing" | "rewriting";

type Store = EmotionState & {
  diff: DiffSegment[];
  status: Status;
  error: string | null;
  axisLabels: AxisLabels | null;
  basisPhrases: string[];
  excludedComponents: number[];

  setSourceText: (t: string) => void;
  setStrength: (v: number) => void;
  setMode: (m: Mode) => void;

  loadMeta: () => Promise<void>;
  analyze: () => Promise<void>;
  rewrite: () => Promise<void>;
  preset: (p: Preset) => Promise<void>;
  setMacro: (name: MacroEmotion, v: number) => Promise<void>;
  setProjection: (p: Projection) => Promise<void>;
  setBasisComponent: (index: number, value: number) => Promise<void>;
  shuffleLatent: () => Promise<void>;
};

export const useStore = create<Store>()((set, get) => ({
  sourceText: "",
  rewrittenText: "",
  basisVector: ZERO_BASIS,
  macroEmotions: { ...ZERO_MACRO },
  projection: { x: 0, y: 0 },
  topBasisComponents: [],
  strength: 0.2,
  mode: "balanced",
  diff: [],
  status: "idle",
  error: null,
  axisLabels: null,
  basisPhrases: [],
  excludedComponents: [],

  loadMeta: async () => {
    try {
      const m = await getMeta();
      set({
        axisLabels: m.axis_labels,
        basisPhrases: m.basis_phrases,
        excludedComponents: m.excluded_components ?? [],
      });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  setSourceText: (t) => set({ sourceText: t }),
  setStrength: (v) => {
    const clamped = Math.max(0.01, Math.min(0.4, v));
    set({ strength: clamped });
  },
  setMode: (m) => set({ mode: m }),

  analyze: async () => {
    const text = get().sourceText;
    if (!text.trim()) return;
    set({ status: "analyzing", error: null });
    try {
      const r = await analyzeText(text);
      set({
        basisVector: r.basis_vector,
        macroEmotions: r.macro_emotions,
        projection: r.projection,
        topBasisComponents: r.top_basis_components,
      });
      // Auto-rewrite once after analysis so the right deck has content.
      await get().rewrite();
    } catch (e) {
      set({ error: (e as Error).message });
    } finally {
      set({ status: "idle" });
    }
  },

  rewrite: async () => {
    const s = get();
    if (!s.sourceText.trim()) return;
    set({ status: "rewriting", error: null });
    try {
      const r = await rewriteText({
        source_text: s.sourceText,
        basis_vector: s.basisVector,
        macro_emotions: s.macroEmotions,
        strength: s.strength,
        mode: s.mode,
      });
      set({
        rewrittenText: r.rewritten_text,
        macroEmotions: r.macro_emotions,
        projection: r.projection,
        diff: r.diff,
      });
    } catch (e) {
      set({ error: (e as Error).message });
    } finally {
      set({ status: "idle" });
    }
  },

  preset: async (p) => {
    const s = get();
    set({ status: "rewriting", error: null });
    try {
      const r = await applyPreset({
        current_basis_vector: s.basisVector,
        preset: p,
        strength: s.strength,
      });
      set({
        basisVector: r.basis_vector,
        macroEmotions: r.macro_emotions,
        projection: r.projection,
        topBasisComponents: r.top_basis_components,
      });
      await get().rewrite();
    } catch (e) {
      set({ error: (e as Error).message });
    } finally {
      set({ status: "idle" });
    }
  },

  setMacro: async (name, v) => {
    // optimistic local update so the slider feels alive
    const target: MacroEmotions = { ...get().macroEmotions, [name]: v };
    set({ macroEmotions: target });
    try {
      const r = await updateFromMacro({
        current_basis_vector: get().basisVector,
        target_macro_emotions: target,
        blend: 0.4,
      });
      set({
        basisVector: r.basis_vector,
        macroEmotions: r.macro_emotions,
        projection: r.projection,
        topBasisComponents: r.top_basis_components,
      });
      await get().rewrite();
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  setProjection: async (p) => {
    set({ projection: p });
    try {
      const r = await updateFromProjection({
        current_basis_vector: get().basisVector,
        x: p.x,
        y: p.y,
      });
      set({
        basisVector: r.basis_vector,
        macroEmotions: r.macro_emotions,
        projection: r.projection,
        topBasisComponents: r.top_basis_components,
      });
      await get().rewrite();
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  setBasisComponent: async (index, value) => {
    // Refuse edits on pathological axes (backend would zero them anyway).
    if (get().excludedComponents.includes(index)) return;
    const next = [...get().basisVector];
    next[index] = Math.max(-1, Math.min(1, value));
    // recompute top via local sort; macro/projection get refreshed by rewrite()
    const top: TopBasisComponent[] = next
      .map((w, i) => ({ index: i, weight: w, label: "latent component" }))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 10);
    set({ basisVector: next, topBasisComponents: top });
    await get().rewrite();
  },

  shuffleLatent: async () => {
    const next = get().basisVector.map((w) => {
      const noise = (Math.random() - 0.5) * 0.4;
      const v = Math.tanh(w + noise);
      return v;
    });
    set({ basisVector: next });
    await get().rewrite();
  },
}));

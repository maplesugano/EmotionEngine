export type MacroEmotion =
  | "joy"
  | "trust"
  | "fear"
  | "surprise"
  | "sadness"
  | "disgust"
  | "anger"
  | "anticipation";

export const MACRO_EMOTIONS: MacroEmotion[] = [
  "joy",
  "trust",
  "fear",
  "surprise",
  "sadness",
  "disgust",
  "anger",
  "anticipation",
];

export type Mode = "subtle" | "balanced" | "strong";

export type Preset =
  | "uncertainty"
  | "self_doubt"
  | "analytical_detachment"
  | "addressivity"
  | "warmth"
  | "urgency";

export const PRESETS: { id: Preset; label: string; hint: string }[] = [
  { id: "uncertainty", label: "Uncertainty", hint: "Hedge, soften assertions" },
  { id: "self_doubt", label: "Self-doubt", hint: "Inward, second-guessing" },
  {
    id: "analytical_detachment",
    label: "Analytical detachment",
    hint: "Impersonal, objective",
  },
  { id: "addressivity", label: "Addressivity", hint: "Direct, second-person" },
  { id: "warmth", label: "Warmth", hint: "Soft, friendly opening" },
  { id: "urgency", label: "Urgency", hint: "Tight, pressing" },
];

export type Projection = { x: number; y: number };

export type TopBasisComponent = {
  index: number;
  weight: number;
  label: string;
};

export type MacroEmotions = Record<MacroEmotion, number>;

export type DiffSegment = {
  type: "same" | "removed" | "added";
  text: string;
};

export type EmotionState = {
  sourceText: string;
  rewrittenText: string;
  basisVector: number[];
  macroEmotions: MacroEmotions;
  projection: Projection;
  topBasisComponents: TopBasisComponent[];
  strength: number;
  mode: Mode;
};

export type AnalyzeResponse = {
  source_text: string;
  basis_vector: number[];
  macro_emotions: MacroEmotions;
  projection: Projection;
  top_basis_components: TopBasisComponent[];
};

export type RewriteResponse = {
  rewritten_text: string;
  basis_vector: number[];
  macro_emotions: MacroEmotions;
  projection: Projection;
  diff: DiffSegment[];
};

export type PresetResponse = {
  basis_vector: number[];
  macro_emotions: MacroEmotions;
  projection: Projection;
  top_basis_components: TopBasisComponent[];
};

export type MetaResponse = {
  basis_dim: number;
  macro_emotions: MacroEmotion[];
  presets: Preset[];
  basis_labels: Record<string, string>;
};

export const ZERO_MACRO: MacroEmotions = {
  joy: 0,
  trust: 0,
  fear: 0,
  surprise: 0,
  sadness: 0,
  disgust: 0,
  anger: 0,
  anticipation: 0,
};

export const ZERO_BASIS: number[] = new Array(64).fill(0);

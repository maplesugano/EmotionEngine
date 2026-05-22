import type {
  AnalyzeResponse,
  MacroEmotions,
  MetaResponse,
  Mode,
  Preset,
  PresetResponse,
  RewriteResponse,
} from "../types";

const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://localhost:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`POST ${path} → ${res.status}: ${txt}`);
  }
  return (await res.json()) as T;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} → ${res.status}`);
  }
  return (await res.json()) as T;
}

export function getMeta(): Promise<MetaResponse> {
  return get<MetaResponse>("/api/meta");
}

export function analyzeText(text: string): Promise<AnalyzeResponse> {
  return post<AnalyzeResponse>("/api/analyze", { text });
}

export function rewriteText(payload: {
  source_text: string;
  basis_vector: number[];
  macro_emotions: MacroEmotions;
  strength: number;
  mode: Mode;
}): Promise<RewriteResponse> {
  return post<RewriteResponse>("/api/rewrite", payload);
}

export function applyPreset(payload: {
  current_basis_vector: number[];
  preset: Preset;
  strength?: number;
}): Promise<PresetResponse> {
  return post<PresetResponse>("/api/preset", {
    strength: 0.2,
    ...payload,
  });
}

export function updateFromMacro(payload: {
  current_basis_vector: number[];
  target_macro_emotions: MacroEmotions;
  blend?: number;
}): Promise<PresetResponse> {
  return post<PresetResponse>("/api/update_from_macro", {
    blend: 0.5,
    ...payload,
  });
}

export function updateFromProjection(payload: {
  current_basis_vector: number[];
  x: number;
  y: number;
}): Promise<PresetResponse> {
  return post<PresetResponse>("/api/update_from_projection", payload);
}

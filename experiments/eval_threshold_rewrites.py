"""Threshold-alpha rewrites: CODEBOOK_FIXED_PROMPTS を各 ICA 基底ベクトル 64 軸でリライト.

threshold_summary.csv に記録されたブレークポイント直前の閾値アルファを使い、
各プロンプトを negative / positive 方向にステアリングして生成したテキストと
元のプロンプトを対にして JSON として保存する。

既存の adaptive_generations.parquet からキャッシュを読み込み、未生成分のみ
実際にモデルを呼び出す（Generation Cache Policy に準拠）。

Usage
-----
    uv run python -m experiments.eval_threshold_rewrites \\
        --basis data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt \\
        --threshold-csv experiments/results/emotion_codebook/ica_k064_L22/threshold_summary.csv \\
        --output-dir experiments/results/threshold_rewrites/ica_k064_L22
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.activations._runtime import load_model, load_profile
from src.steering.generate import steered_generate
from experiments.eval_emotion_codebook import (
    CODEBOOK_FIXED_PROMPTS,
    GenerationCache,
    _compute_alpha_scale,
    _finalize_min_sentences,
    _load_basis_artifact,
)


def _extract_or_generate(
    gen_cache: GenerationCache,
    component: int,
    signed_au: float,
    prompts: list[str],
    *,
    model,
    basis_vec: torch.Tensor,
    alpha_phys: float,
    layer: int,
    apply_to: str,
    max_new_tokens: int,
    min_sentences: int,
) -> list[str]:
    """キャッシュから取得するか、モデルを呼び出して生成する。"""
    results = []
    for prompt in prompts:
        cached = gen_cache.get(component, signed_au, prompt)
        if cached is not None:
            results.append(cached)
        else:
            gen = steered_generate(
                model,
                prompt,
                vector=basis_vec,
                alpha=alpha_phys,
                layers=[layer],
                apply_to=apply_to,
                max_new_tokens=max_new_tokens + 24,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            )
            tail = gen[len(prompt):] if gen.startswith(prompt) else gen
            tail = _finalize_min_sentences(tail, min_sentences=min_sentences)
            gen_cache.put(component, signed_au, prompt, tail)
            results.append(tail)
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="各 ICA 基底軸の閾値アルファで CODEBOOK_FIXED_PROMPTS をリライトし JSON を出力する"
    )
    p.add_argument(
        "--basis",
        type=Path,
        default=Path("data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt"),
    )
    p.add_argument(
        "--threshold-csv",
        type=Path,
        default=Path("experiments/results/emotion_codebook/ica_k064_L22/threshold_summary.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/threshold_rewrites/ica_k064_L22"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model.yaml"),
    )
    p.add_argument(
        "--steering-config",
        type=Path,
        default=Path("configs/steering.yaml"),
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )
    p.add_argument(
        "--min-sentences",
        type=int,
        default=2,
    )
    p.add_argument(
        "--components",
        type=int,
        nargs="*",
        default=None,
        help="処理対象のコンポーネント番号（省略時: 全64軸）",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 基底ベクトルの読み込み ---
    print(f"[main] 基底ロード: {args.basis}")
    W, layer, k, decomposer = _load_basis_artifact(args.basis)
    alpha_scale = _compute_alpha_scale(W, "caa_match")
    print(f"[main] k={k}, layer={layer}, decomposer={decomposer}")

    # --- モデル設定 ---
    profile_cfg, profile_name = load_profile(args.config)
    sc = yaml.safe_load(args.steering_config.read_text())
    apply_to = sc["caa"].get("apply_to", "generation")

    # --- 閾値サマリの読み込み ---
    threshold_df = pd.read_csv(args.threshold_csv)
    threshold_lookup: dict[int, dict] = {
        int(row["component"]): row.to_dict()
        for _, row in threshold_df.iterrows()
    }

    # --- 生成キャッシュの読み込み ---
    gen_cache = GenerationCache.from_results_dir(Path("experiments/results"))

    # --- 対象コンポーネントの決定 ---
    components = sorted(set(args.components)) if args.components else list(range(k))
    print(f"[main] 処理対象: {len(components)} 軸")

    model = None
    prompts = CODEBOOK_FIXED_PROMPTS

    new_rows: list[dict] = []  # adaptive_generations.parquetへの追記用

    for component in tqdm(components, desc="軸ごとにリライト"):
        row = threshold_lookup.get(component)
        if row is None:
            print(f"  [skip] component {component}: threshold_summary に記録なし")
            continue

        alpha_neg = float(row["threshold_alpha_unit_negative"])
        alpha_pos = float(row["threshold_alpha_unit_positive"])
        b_vec = torch.from_numpy(W[component]).to(torch.float32)
        b_norm = float(np.linalg.norm(W[component]))

        signed_au_neg = -alpha_neg
        signed_au_pos = alpha_pos
        alpha_phys_neg = signed_au_neg * alpha_scale * b_norm
        alpha_phys_pos = signed_au_pos * alpha_scale * b_norm

        # 必要な場合のみモデルをロード
        need_neg = any(gen_cache.get(component, round(signed_au_neg, 6), pr) is None for pr in prompts)
        need_pos = any(gen_cache.get(component, round(signed_au_pos, 6), pr) is None for pr in prompts)
        if (need_neg or need_pos) and model is None:
            print("[main] モデルをロード中...")
            model, _, _ = load_model(profile_cfg)

        gens_neg = _extract_or_generate(
            gen_cache,
            component,
            signed_au_neg,
            prompts,
            model=model,
            basis_vec=b_vec,
            alpha_phys=alpha_phys_neg,
            layer=layer,
            apply_to=apply_to,
            max_new_tokens=args.max_new_tokens,
            min_sentences=args.min_sentences,
        )
        gens_pos = _extract_or_generate(
            gen_cache,
            component,
            signed_au_pos,
            prompts,
            model=model,
            basis_vec=b_vec,
            alpha_phys=alpha_phys_pos,
            layer=layer,
            apply_to=apply_to,
            max_new_tokens=args.max_new_tokens,
            min_sentences=args.min_sentences,
        )

        # --- JSON 出力 (1軸 = 1ファイル) ---
        pairs = [
            {
                "prompt_id": pid,
                "original": prompt,
                "generation_negative": gens_neg[pid],
                "generation_positive": gens_pos[pid],
            }
            for pid, prompt in enumerate(prompts)
        ]
        record = {
            "component": component,
            "basis": str(args.basis),
            "layer": layer,
            "decomposer": decomposer,
            "alpha_unit_negative": alpha_neg,
            "alpha_unit_positive": alpha_pos,
            "prompts": pairs,
        }
        out_path = args.output_dir / f"b{component:02d}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        # adaptive_generations.parquet への追記用行を収集（キャッシュになかった分）
        for pid, prompt in enumerate(prompts):
            for sign_val, au, alpha_phys, gens in [
                (-1, alpha_neg, alpha_phys_neg, gens_neg),
                (1, alpha_pos, alpha_phys_pos, gens_pos),
            ]:
                new_rows.append({
                    "component": component,
                    "sign": sign_val,
                    "alpha_unit": au,
                    "prompt_id": pid,
                    "prompt": prompt,
                    "generation": gens[pid],
                    "effective_alpha": alpha_phys,
                    "basis_path": str(args.basis),
                    "layer": layer,
                    "k": k,
                })

    print(f"\n[main] JSON 出力完了: {len(components)} ファイル → {args.output_dir}/b??.json")

    # --- 新規生成行を adaptive_generations.parquet に追記 ---
    adaptive_path = Path("experiments/results/emotion_codebook/ica_k064_L22/adaptive_generations.parquet")
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if adaptive_path.exists():
            old_df = pd.read_parquet(adaptive_path)
            # 重複除去してマージ
            merged = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
                subset=["component", "sign", "alpha_unit", "prompt_id"]
            )
            merged.to_parquet(adaptive_path, index=False)
            print(f"[main] adaptive_generations.parquet を更新: {len(old_df)} → {len(merged)} 行")
        else:
            new_df.to_parquet(adaptive_path, index=False)
            print(f"[main] adaptive_generations.parquet を新規作成: {len(new_df)} 行")

    print(f"\n[main] ✓ 完了. 出力先: {args.output_dir}")


if __name__ == "__main__":
    main()

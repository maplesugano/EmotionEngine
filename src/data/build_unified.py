"""Build the unified, normalised dataset by running every loader.

Usage
-----
    uv run python -m src.data.build_unified \
        --config configs/data.yaml \
        --output data/unified/examples.parquet

Outputs
-------
- data/unified/examples.parquet   one row per `EmotionExample`
- data/unified/stats.json         per-source / per-category counts
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from .loaders import LOADERS, LoaderError
from .schema import CATEGORIES


def _to_records(loader_fn, name: str, splits: list[str] | None) -> list[dict]:
    out: list[dict] = []
    try:
        for ex in loader_fn(splits=splits):
            out.append(ex.to_dict())
    except LoaderError as exc:
        print(f"[{name}] skipped: {exc}")
    except Exception as exc:                       # noqa: BLE001
        print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
        raise
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/unified/examples.parquet"))
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Optional subset of source names; default = all sources in config",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    source_specs = {s["name"]: s for s in cfg["source_datasets"]}
    selected = args.sources or list(source_specs.keys())

    records: list[dict] = []
    for name in selected:
        if name not in LOADERS:
            print(f"[skip] unknown loader: {name}")
            continue
        spec = source_specs[name]
        splits = spec.get("splits")
        print(f"[{name}] loading splits={splits} ...")
        rows = _to_records(LOADERS[name], name, splits)
        print(f"[{name}] -> {len(rows)} examples")
        records.extend(rows)

    if not records:
        raise SystemExit("No records produced; check loader errors above.")

    df = pd.DataFrame.from_records(records)
    # `source_labels` is a free-form dict per source; serialise to JSON so
    # Arrow doesn't try to infer a single struct schema across sources.
    if "source_labels" in df.columns:
        df["source_labels"] = df["source_labels"].apply(
            lambda v: json.dumps(v, ensure_ascii=False, default=str) if v else "{}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    # Stats
    by_source = Counter(df["source"])
    by_category = {
        c: Counter(df.loc[df["source"] == s, "label_primary"]).get(c, 0)
        for s in by_source
        for c in CATEGORIES
    }
    stats = {
        "total": int(len(df)),
        "by_source": dict(by_source),
        "category_counts_by_source": {
            s: {c: int(Counter(df.loc[df["source"] == s, "label_primary"]).get(c, 0))
                for c in CATEGORIES}
            for s in by_source
        },
    }
    stats_path = args.output.with_name("stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {len(df)} rows → {args.output}")
    print(f"Wrote stats         → {stats_path}")


if __name__ == "__main__":
    main()

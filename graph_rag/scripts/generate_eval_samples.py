"""Generate deterministic mixed-config evaluation index files from local RAGBench copies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from random import Random

from datasets import Dataset, DatasetDict, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RANDOM_SEED

PRIMARY_CONFIGS = ["hotpotqa", "msmarco", "hagrid"]
SECONDARY_CONFIGS = ["delucionqa", "cuad", "emanual"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sample index generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ragbench"),
        help="Root directory that contains one folder per RAGBench config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name used for metadata validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic index generation.",
    )
    parser.add_argument(
        "--primary-configs",
        nargs="+",
        default=PRIMARY_CONFIGS,
        help="Configs used for the primary mixed evaluation set.",
    )
    parser.add_argument(
        "--secondary-configs",
        nargs="+",
        default=SECONDARY_CONFIGS,
        help="Configs used for the secondary mixed evaluation set.",
    )
    parser.add_argument(
        "--primary-total",
        type=int,
        default=500,
        help="Total number of questions in the primary mixed set.",
    )
    parser.add_argument(
        "--secondary-total",
        type=int,
        default=200,
        help="Total number of questions in the secondary mixed set.",
    )
    parser.add_argument(
        "--primary-output",
        type=Path,
        default=Path("data/shared_eval_indices_primary_500.json"),
        help="Output JSON path for primary mixed indices.",
    )
    parser.add_argument(
        "--secondary-output",
        type=Path,
        default=Path("data/shared_eval_indices_secondary_200.json"),
        help="Output JSON path for secondary mixed indices.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path("data/shared_eval_indices.json"),
        help="Output JSON path containing both primary and secondary payloads.",
    )
    return parser.parse_args()


def _normalize_configs(configs: list[str]) -> list[str]:
    """Deduplicate config names while preserving order."""
    return list(dict.fromkeys(configs))


def _load_local_split(dataset_path: Path, split: str) -> Dataset:
    """Load one config from disk and return a Dataset object."""
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: '{dataset_path}'"
        raise FileNotFoundError(msg)

    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, Dataset):
        return loaded

    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            msg = f"Split '{split}' not found at '{dataset_path}'."
            raise KeyError(msg)
        return loaded[split]

    msg = f"Unsupported dataset object at '{dataset_path}': {type(loaded).__name__}"
    raise TypeError(msg)


def _allocation(total: int, configs: list[str]) -> dict[str, int]:
    """Allocate a total sample size as evenly as possible across configs."""
    if total <= 0:
        msg = "Total sample size must be greater than zero."
        raise ValueError(msg)

    if not configs:
        msg = "At least one config is required."
        raise ValueError(msg)

    base = total // len(configs)
    remainder = total % len(configs)
    return {
        config_name: base + (1 if index < remainder else 0)
        for index, config_name in enumerate(configs)
    }


def _build_group_payload(
    *,
    data_dir: Path,
    split: str,
    seed: int,
    group_name: str,
    configs: list[str],
    total: int,
) -> dict:
    """Build deterministic sample indices for one evaluation group."""
    normalized = _normalize_configs(configs)
    counts = _allocation(total=total, configs=normalized)
    rng = Random(seed)

    indices_by_config: dict[str, list[int]] = {}
    mixed_samples: list[dict[str, int | str]] = []
    source_sizes: dict[str, int] = {}

    for config_name in normalized:
        dataset_path = data_dir / config_name
        dataset = _load_local_split(dataset_path=dataset_path, split=split)

        source_sizes[config_name] = len(dataset)
        needed = counts[config_name]
        if needed > len(dataset):
            msg = (
                f"Config '{config_name}' has {len(dataset)} rows but {needed} are required "
                f"for {group_name} sample generation."
            )
            raise ValueError(msg)

        selected = sorted(rng.sample(range(len(dataset)), needed))
        indices_by_config[config_name] = selected
        for row_index in selected:
            mixed_samples.append({"config": config_name, "row_index": row_index})

    rng.shuffle(mixed_samples)
    return {
        "group_name": group_name,
        "split": split,
        "seed": seed,
        "total_requested": total,
        "total_selected": len(mixed_samples),
        "configs": normalized,
        "counts_per_config": counts,
        "source_sizes": source_sizes,
        "indices_by_config": indices_by_config,
        "mixed_samples": mixed_samples,
    }


def _write_json(payload: dict, path: Path) -> None:
    """Persist a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    """Generate and save deterministic primary and secondary sample index files."""
    args = parse_args()

    primary_payload = _build_group_payload(
        data_dir=args.data_dir,
        split=args.split,
        seed=args.seed,
        group_name="primary",
        configs=args.primary_configs,
        total=args.primary_total,
    )
    secondary_payload = _build_group_payload(
        data_dir=args.data_dir,
        split=args.split,
        seed=args.seed,
        group_name="secondary",
        configs=args.secondary_configs,
        total=args.secondary_total,
    )

    _write_json(primary_payload, args.primary_output)
    _write_json(secondary_payload, args.secondary_output)

    combined_payload = {
        "dataset": "rungalileo/ragbench",
        "data_dir": str(args.data_dir),
        "split": args.split,
        "seed": args.seed,
        "primary": primary_payload,
        "secondary": secondary_payload,
    }
    _write_json(combined_payload, args.combined_output)

    print(
        "Saved deterministic sample indices to "
        f"'{args.primary_output}', '{args.secondary_output}', and '{args.combined_output}'."
    )


if __name__ == "__main__":
    main()

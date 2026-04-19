"""Download a deterministic RAGBench sample for MVP runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RANDOM_SEED


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sample download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=50, help="Number of rows to sample.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rungalileo/ragbench",
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hotpotqa",
        help="Dataset config/subset (default: hotpotqa).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ragbench_50"),
        help="Path to save sampled dataset.",
    )
    parser.add_argument(
        "--indices-path",
        type=Path,
        default=Path("data/mvp_eval_indices_50.json"),
        help="Path to save sampled indices.",
    )
    return parser.parse_args()


def download_sample(dataset_name: str, dataset_config: str, split: str, sample_size: int):
    """Load RAGBench and return a deterministic sampled dataset and indices."""
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if sample_size > len(dataset):
        msg = f"Requested sample_size={sample_size}, but split has only {len(dataset)} rows."
        raise ValueError(msg)

    shuffled = dataset.shuffle(seed=RANDOM_SEED)
    sampled = shuffled.select(range(sample_size))

    sampled_indices: list = []
    if "id" in sampled.column_names:
        sampled_indices = list(sampled["id"])
    else:
        sampled_indices = list(range(sample_size))

    return sampled, sampled_indices


def save_outputs(sampled_dataset, output_dir: Path):
    """Persist sampled dataset and selected IDs to disk."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_dataset.save_to_disk(str(output_dir))


def save_indices(indices: list, indices_path: Path):
    """Write selected sample identifiers to a JSON file."""
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    with indices_path.open("w", encoding="utf-8") as file:
        json.dump(indices, file, indent=2)


def main() -> None:
    """Execute the deterministic RAGBench sample download flow."""
    args = parse_args()
    sampled_dataset, sampled_indices = download_sample(
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.split,
        sample_size=args.sample_size,
    )
    save_outputs(sampled_dataset, args.output_dir)
    save_indices(sampled_indices, args.indices_path)
    print(
        f"Saved {len(sampled_dataset)} rows to '{args.output_dir}' and indices to '{args.indices_path}'."
    )


if __name__ == "__main__":
    main()

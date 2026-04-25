"""Download and persist team-standard RAGBench configs for local reuse."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import Dataset, load_dataset

DEFAULT_DATASET = "rungalileo/ragbench"
DEFAULT_SPLIT = "test"
PRIMARY_CONFIGS = ["hotpotqa", "msmarco", "hagrid"]
OPTIONAL_CONFIGS = ["delucionqa", "cuad", "emanual"]
DEFAULT_CONFIGS = PRIMARY_CONFIGS + OPTIONAL_CONFIGS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Dataset split to save locally (default: test).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Dataset configs to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ragbench"),
        help="Root directory for local dataset copies.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing config folders.",
    )
    return parser.parse_args()


def _is_complete_saved_dataset(dataset_path: Path) -> bool:
    """Return True if a folder appears to be a complete HF saved dataset."""
    if not dataset_path.is_dir():
        return False

    required_files = ["dataset_info.json", "state.json"]
    has_required_files = all(
        (dataset_path / filename).is_file() for filename in required_files
    )
    has_arrow_file = any(dataset_path.glob("*.arrow"))
    return has_required_files and has_arrow_file


def _normalize_configs(configs: list[str]) -> list[str]:
    """Deduplicate config names while preserving order."""
    return list(dict.fromkeys(configs))


def download_config(
    dataset_name: str,
    config_name: str,
    split: str,
    destination: Path,
) -> int:
    """Download one config split from Hugging Face and save it to disk."""
    dataset: Dataset = load_dataset(dataset_name, config_name, split=split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))
    return len(dataset)


def verify_saved_configs(output_dir: Path, configs: list[str]) -> None:
    """Verify that all requested config folders were saved successfully."""
    missing_or_invalid = [
        config_name
        for config_name in configs
        if not _is_complete_saved_dataset(output_dir / config_name)
    ]
    if missing_or_invalid:
        missing_text = ", ".join(missing_or_invalid)
        msg = (
            "The following config folders are missing or incomplete: "
            f"{missing_text}. Rerun the download script with --overwrite."
        )
        raise RuntimeError(msg)


def main() -> None:
    """Run the team dataset download flow and verify outputs."""
    args = parse_args()
    configs = _normalize_configs(args.configs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0

    for config_name in configs:
        destination = args.output_dir / config_name

        if destination.exists():
            if args.overwrite:
                shutil.rmtree(destination)
            elif _is_complete_saved_dataset(destination):
                print(
                    f"Skipping {config_name}: found existing dataset at '{destination}'."
                )
                skipped += 1
                continue
            else:
                msg = (
                    f"Found incomplete dataset at '{destination}'. "
                    "Rerun with --overwrite to replace it."
                )
                raise RuntimeError(msg)

        print(
            f"Downloading {config_name} ({args.split}) from {args.dataset} "
            f"to '{destination}'..."
        )
        row_count = download_config(
            dataset_name=args.dataset,
            config_name=config_name,
            split=args.split,
            destination=destination,
        )
        print(f"Saved {row_count} rows for {config_name}.")
        downloaded += 1

    verify_saved_configs(args.output_dir, configs)
    print(
        "All requested config folders are present and valid under "
        f"'{args.output_dir}'. Downloaded {downloaded}, skipped {skipped}."
    )


if __name__ == "__main__":
    main()

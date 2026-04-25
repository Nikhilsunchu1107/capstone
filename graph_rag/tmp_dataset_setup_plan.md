# Temporary Plan: RAGBench Dataset Download and Setup

Created: 2026-04-25
Status: Draft reference plan for future execution

## Goal

Create a repeatable team setup for downloading and using the six decided RAGBench configs so every member runs evaluations on identical data.

- Primary configs: `hotpotqa`, `msmarco`, `hagrid`
- Optional configs: `delucionqa`, `cuad`, `emanual`
- Split to use: `test` only

## Phase 1: Centralized Dataset Download (One teammate)

1. Create `download_datasets.py` to download all six configs from `rungalileo/ragbench`.
2. Save each config under `./data/ragbench/<config_name>`.
3. Run and verify all six folders exist.
4. Keep large raw data out of normal Git history (prepare ignore/LFS strategy before committing data blobs).

## Phase 2: Team Data Sharing

1. Use Git LFS for `data/ragbench/**` if dataset files are shared via Git.
2. Commit `.gitattributes` with LFS tracking rules.
3. Push dataset artifacts once; teammates pull the exact same local copies.

## Phase 3: Shared Evaluation Sample Generation (One teammate)

1. Create `generate_eval_samples.py` that loads local datasets with `load_from_disk()`.
2. Generate team-standard index files using fixed seed (`RANDOM_SEED=42`):
   - primary mixed sample indices (500 total)
   - secondary mixed sample indices (200 total)
3. Commit and share the index JSON files (small files) for deterministic evaluation across the team.

## Dependencies

- Python `3.12` (repo runtime)
- `datasets`
- `git`
- `git-lfs` (if sharing large data via Git)

## Risks and Mitigation

- HIGH: Interrupted/partial initial download -> rerun script and verify all config folders.
- MEDIUM: Teammates missing Git LFS -> provide one-time setup instruction before clone/pull.
- LOW: Upstream dataset changes -> freeze by sharing local saved copies and index files.

## Estimated Complexity

- Overall: MEDIUM
- Initial setup by one teammate: ~1-2 hours (network dependent)
- Per-teammate onboarding: ~10-15 minutes

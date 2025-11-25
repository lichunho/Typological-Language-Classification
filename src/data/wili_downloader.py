"""
WiLI-2018 download and preprocessing utilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import requests
from rich import print as rprint
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from src.config.settings import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_LANGUAGES,
)

WILI_URL = "https://zenodo.org/record/841984/files/wili-2018.zip?download=1"
ARCHIVE_NAME = "wili-2018.zip"
EXTRACTED_DIRNAME = "wili-2018"


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {ARCHIVE_NAME}",
        )
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        progress.close()
    return destination


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_archive(archive_path: Path, output_dir: Path) -> Path:
    extraction_target = output_dir / EXTRACTED_DIRNAME
    extraction_target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extraction_target)
    return extraction_target


def _load_split(data_dir: Path, split: str) -> pd.DataFrame:
    x_file = data_dir / f"x_{split}.txt"
    y_file = data_dir / f"y_{split}.txt"
    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError(
            f"Expected files {x_file.name} / {y_file.name} in {data_dir}. "
            "Ensure the WiLI archive has been extracted."
        )
    texts = x_file.read_text(encoding="utf-8").splitlines()
    labels = y_file.read_text(encoding="utf-8").splitlines()
    if len(texts) != len(labels):
        raise ValueError(f"Split {split} has mismatched text/label counts.")
    return pd.DataFrame({"text": texts, "language": labels, "split": split})


def _balanced_subset(
    df: pd.DataFrame,
    languages: Sequence[str],
    limit_per_language: Optional[int],
) -> pd.DataFrame:
    counters = defaultdict(int)
    rows = []
    for _, row in df.iterrows():
        lang = row["language"]
        if lang not in languages:
            continue
        if limit_per_language and counters[lang] >= limit_per_language:
            continue
        counters[lang] += 1
        rows.append(row)
    subset = pd.DataFrame(rows)
    if subset.empty:
        raise ValueError("No rows selected; check language list or limits.")
    return subset


def _stratified_triplet(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.")

    X = df["text"]
    y = df["language"]

    sss_primary = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio + test_ratio, random_state=seed
        )
    train_idx, temp_idx = next(sss_primary.split(X, y))

    temp_df = df.iloc[temp_idx]
    temp_y = temp_df["language"]

    secondary_test_size = test_ratio / (val_ratio + test_ratio)
    sss_secondary = StratifiedShuffleSplit(
        n_splits=1, test_size=secondary_test_size, random_state=seed
    )
    val_idx_rel, test_idx_rel = next(sss_secondary.split(temp_df["text"], temp_y))

    train_df = df.iloc[train_idx].copy()
    val_df = temp_df.iloc[val_idx_rel].copy()
    test_df = temp_df.iloc[test_idx_rel].copy()
    return train_df, val_df, test_df


def _write_metadata(
    output_base: Path,
    rows: pd.DataFrame,
    archive_path: Path,
    languages: Sequence[str],
    splits_summary: dict,
) -> None:
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": WILI_URL,
        "archive_path": str(archive_path),
        "archive_sha256": _file_sha256(archive_path) if archive_path.exists() else None,
        "languages": {
            lang: int((rows["language"] == lang).sum()) for lang in languages
        },
        "splits": splits_summary,
    }
    metadata_path = output_base.with_suffix(output_base.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rprint(f"[cyan]Wrote metadata → {metadata_path}")


def download_cmd(args: argparse.Namespace) -> None:
    destination = Path(args.destination)
    if destination.exists():
        rprint(f"[yellow]{destination} already exists; skipping download.")
        return
    _download_file(args.url, destination)
    rprint(f"[green]Downloaded WiLI archive to {destination}")


def prepare_subset_cmd(args: argparse.Namespace) -> None:
    """
    Extract the archive (if needed) and emit a balanced CSV for training.
    """
    raw_dir = RAW_DATA_DIR / EXTRACTED_DIRNAME
    archive_path = Path(args.archive_path)
    if not raw_dir.exists():
        if not archive_path.exists():
            rprint("[yellow]Archive missing; downloading now...")
            _download_file(WILI_URL, archive_path)
        rprint("[cyan]Extracting archive...")
        _extract_archive(archive_path, RAW_DATA_DIR)

    frames = [_load_split(raw_dir, split) for split in args.splits]
    combined = pd.concat(frames, ignore_index=True)
    target_langs = (
        [code.strip() for code in args.languages.split(",") if code.strip()]
        if args.languages
        else TARGET_LANGUAGES
    )
    limit = None if args.limit_per_language is None or args.limit_per_language < 0 else args.limit_per_language
    subset = _balanced_subset(combined, target_langs, limit)

    train_df, val_df, test_df = _stratified_triplet(
        subset, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    splits_map = {"train": train_df, "val": val_df, "test": test_df}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_output = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    combined_output.to_csv(output_path, index=False)
    rprint(
        f"[green]Saved subset with {len(combined_output)} rows, "
        f"{len(target_langs)} languages → {output_path}"
    )

    for name, frame in splits_map.items():
        split_file = output_path.with_name(f"{output_path.stem}_{name}.csv")
        frame.to_csv(split_file, index=False)
        rprint(f"[blue]{name} split → {split_file} ({len(frame)} rows)")

    _write_metadata(
        output_path,
        combined_output,
        archive_path,
        target_langs,
        {name: len(frame) for name, frame in splits_map.items()},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and preprocess WiLI-2018 language data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl_parser = subparsers.add_parser("download", help="Fetch the WiLI archive.")
    dl_parser.add_argument("--url", default=WILI_URL, help="WiLI download URL.")
    dl_parser.add_argument(
        "--destination",
        default=str(RAW_DATA_DIR / ARCHIVE_NAME),
        help="Where to store the downloaded zip.",
    )
    dl_parser.set_defaults(func=download_cmd)

    prep_parser = subparsers.add_parser(
        "prepare-subset", help="Create balanced CSV splits."
    )
    prep_parser.add_argument(
        "--languages",
        type=str,
        default="",
        help="Comma-separated ISO-639-3 codes (default: config list).",
    )
    prep_parser.add_argument(
        "--limit-per-language",
        type=int,
        default=-1,
        help="Max samples per language per split (-1 means use everything).",
    )
    prep_parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Which WiLI files to combine (default: train test).",
    )
    prep_parser.add_argument(
        "--archive-path",
        default=str(RAW_DATA_DIR / ARCHIVE_NAME),
        help="Path to the downloaded WiLI archive.",
    )
    prep_parser.add_argument(
        "--output-path",
        default=str(PROCESSED_DATA_DIR / "wili_subset.csv"),
        help="Location to store the combined CSV.",
    )
    prep_parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Train proportion."
    )
    prep_parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation proportion."
    )
    prep_parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test proportion."
    )
    prep_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    prep_parser.set_defaults(func=prepare_subset_cmd)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
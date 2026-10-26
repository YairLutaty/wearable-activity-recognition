#!/usr/bin/env python3
"""Export the frozen windowed train/test sample datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = (
    PROJECT_DIR / "baseline" / "artifacts" / "samples.parquet"
)
SOURCE_META_PATH = (
    PROJECT_DIR / "baseline" / "artifacts" / "split_meta.json"
)

OUTPUT_DIR = PROJECT_DIR / "modified"
TRAIN_PATH = OUTPUT_DIR / "train_samples.parquet"
TEST_PATH = OUTPUT_DIR / "test_samples.parquet"
META_PATH = OUTPUT_DIR / "windowed_split_meta.json"

EXPECTED_TOTAL_ROWS = 14_985_272
EXPECTED_TRAIN_ROWS = 11_921_624
EXPECTED_TEST_ROWS = 3_063_648
EXPECTED_COLUMNS = 11


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    for output_path in [TRAIN_PATH, TEST_PATH, META_PATH]:
        if output_path.exists():
            raise FileExistsError(
                f"{output_path} already exists. "
                "Remove it explicitly before rebuilding."
            )

    source_meta = json.loads(
        SOURCE_META_PATH.read_text(encoding="utf-8")
    )

    samples = pd.read_parquet(SOURCE_PATH)

    assert samples.shape == (
        EXPECTED_TOTAL_ROWS,
        EXPECTED_COLUMNS,
    )

    train_samples = samples.loc[
        samples["split"] == "train"
    ].copy()

    test_samples = samples.loc[
        samples["split"] == "test"
    ].copy()

    assert train_samples.shape == (
        EXPECTED_TRAIN_ROWS,
        EXPECTED_COLUMNS,
    )
    assert test_samples.shape == (
        EXPECTED_TEST_ROWS,
        EXPECTED_COLUMNS,
    )
    assert len(train_samples) + len(test_samples) == len(samples)

    train_subjects = sorted(
        int(subject_id)
        for subject_id in train_samples["user_id"].unique()
    )
    test_subjects = sorted(
        int(subject_id)
        for subject_id in test_samples["user_id"].unique()
    )

    assert set(train_subjects).isdisjoint(test_subjects)
    assert train_subjects == source_meta["train_user_ids"]
    assert test_subjects == source_meta["test_user_ids"]

    assert train_samples["split"].eq("train").all()
    assert test_samples["split"].eq("test").all()

    output_meta = {
        "stage": "windowed samples before model-specific preprocessing",
        "source": str(
            SOURCE_PATH.relative_to(PROJECT_DIR)
        ),
        "train_file": TRAIN_PATH.name,
        "test_file": TEST_PATH.name,
        "columns": list(samples.columns),
        "n_total_rows": len(samples),
        "n_train_rows": len(train_samples),
        "n_test_rows": len(test_samples),
        "n_train_windows": train_samples["window_id"].nunique(),
        "n_test_windows": test_samples["window_id"].nunique(),
        "train_subject_ids": train_subjects,
        "test_subject_ids": test_subjects,
        "window_s": source_meta["window_s"],
        "min_samples": source_meta["min_samples"],
        "notes": (
            "Sensor samples already assigned to five-second, "
            "sensor-specific windows. No resampling, scaling, "
            "feature extraction, or multimodal alignment applied."
        ),
    }

    # Confirm JSON serialization before writing large files.
    metadata_json = json.dumps(output_meta, indent=2)

    train_samples.to_parquet(
        TRAIN_PATH,
        index=False,
        compression="zstd",
    )
    test_samples.to_parquet(
        TEST_PATH,
        index=False,
        compression="zstd",
    )

    META_PATH.write_text(
        metadata_json,
        encoding="utf-8",
    )

    print("Train shape:", train_samples.shape)
    print("Test shape:", test_samples.shape)
    print("Train subjects:", len(train_subjects))
    print("Test subjects:", len(test_subjects))
    print("Saved:", TRAIN_PATH)
    print("Saved:", TEST_PATH)
    print("Saved:", META_PATH)


if __name__ == "__main__":
    main()
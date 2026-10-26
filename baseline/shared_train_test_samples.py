#!/usr/bin/env python3
"""Shared train/test sample table (5 s windows).

Builds a group-shared long-format table: one row = one raw sensor sample
inside a 5 s window.

Columns:
  split, user_id, activity, device, sensor, window_id, window_idx,
  timestamp, x, y, z

Protocol (do not change without group agreement):
  - 5 s non-overlapping windows; drop incomplete last window
  - drop windows with fewer than 50 samples
  - drop exact duplicate rows before windowing
  - fixed test user_ids (see TEST_USER_IDS)

Artifacts:
  artifacts/samples.parquet
  artifacts/split_meta.json

Usage:
  python shared_train_test_samples.py
  python shared_train_test_samples.py --force-rebuild
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "raw"
ART = ROOT / "artifacts"

STREAMS = [
    ("phone", "accel"),
    ("phone", "gyro"),
    ("watch", "accel"),
    ("watch", "gyro"),
]

# ---- shared protocol (keep identical across the group) ----
WINDOW_S = 5.0
MIN_SAMPLES = 50
TEST_USER_IDS = [1606, 1607, 1618, 1626, 1629, 1639, 1640, 1641, 1643, 1646]

COLUMNS = [
    "split",
    "user_id",
    "activity",
    "device",
    "sensor",
    "window_id",
    "window_idx",
    "timestamp",
    "x",
    "y",
    "z",
]


def list_stream_files(device: str, sensor: str) -> list[Path]:
    return sorted((RAW / device / sensor).glob(f"data_*_{sensor}_{device}.txt"))


def parse_user_id(path: Path) -> int:
    return int(path.name.split("_")[1])


def read_sensor_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["user_id", "activity", "timestamp", "x", "y", "z"],
        dtype={
            "user_id": "int32",
            "activity": "string",
            "timestamp": "int64",
            "x": "float64",
            "y": "float64",
        },
        converters={"z": lambda v: float(str(v).rstrip(";"))},
    )
    df["z"] = df["z"].astype("float64")
    return df.drop_duplicates().reset_index(drop=True)


def iter_activity_bouts(df: pd.DataFrame):
    acts = df["activity"].to_numpy()
    change = np.r_[True, acts[1:] != acts[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        yield df.iloc[s:e]


def make_window_id(
    user_id: int, device: str, sensor: str, activity: str, bout_window: int
) -> str:
    """Stable unique id — same on every machine given the same protocol."""
    return f"{user_id}_{device}_{sensor}_{activity}_{bout_window:04d}"


def windows_from_bout(
    bout: pd.DataFrame,
    *,
    device: str,
    sensor: str,
    split: str,
) -> list[pd.DataFrame]:
    """Return one small DataFrame per full 5s window (long format)."""
    if len(bout) < MIN_SAMPLES:
        return []

    ts = bout["timestamp"].to_numpy(dtype=np.int64)
    t0 = float(ts[0])
    t_rel = (ts.astype(np.float64) - t0) / 1e9
    n_full = int(np.floor(float(t_rel[-1]) / WINDOW_S))
    if n_full <= 0:
        return []

    user_id = int(bout["user_id"].iloc[0])
    activity = str(bout["activity"].iloc[0])
    out: list[pd.DataFrame] = []

    for w in range(n_full):
        start, end = w * WINDOW_S, (w + 1) * WINDOW_S
        mask = (t_rel >= start) & (t_rel < end)
        n = int(mask.sum())
        if n < MIN_SAMPLES:
            continue

        block = bout.loc[mask, ["timestamp", "x", "y", "z"]].reset_index(drop=True)
        window_id = make_window_id(user_id, device, sensor, activity, w)
        part = pd.DataFrame(
            {
                "split": split,
                "user_id": np.int32(user_id),
                "activity": activity,
                "device": device,
                "sensor": sensor,
                "window_id": window_id,
                "window_idx": np.arange(len(block), dtype=np.int32),
                "timestamp": block["timestamp"].to_numpy(dtype=np.int64),
                "x": block["x"].to_numpy(dtype=np.float64),
                "y": block["y"].to_numpy(dtype=np.float64),
                "z": block["z"].to_numpy(dtype=np.float64),
            }
        )
        out.append(part)
    return out


def build_samples(*, force_rebuild: bool = False) -> pd.DataFrame:
    ART.mkdir(exist_ok=True)
    samples_path = ART / "samples.parquet"
    test_set = set(TEST_USER_IDS)

    if samples_path.exists() and not force_rebuild:
        print(f"Found cache: {samples_path} — loading (use --force-rebuild to rebuild)")
        return pd.read_parquet(samples_path)

    chunks: list[pd.DataFrame] = []
    files = [(d, s, p) for d, s in STREAMS for p in list_stream_files(d, s)]
    print(f"Building sample table from {len(files)} files...")

    for device, sensor, path in tqdm(files, desc="Windowing samples"):
        user_id = parse_user_id(path)
        split = "test" if user_id in test_set else "train"
        df = read_sensor_file(path)
        df["user_id"] = np.int32(user_id)
        for bout in iter_activity_bouts(df):
            chunks.extend(
                windows_from_bout(bout, device=device, sensor=sensor, split=split)
            )

    samples = pd.concat(chunks, ignore_index=True)
    samples = samples[COLUMNS]

    samples["split"] = samples["split"].astype("category")
    samples["activity"] = samples["activity"].astype("category")
    samples["device"] = samples["device"].astype("category")
    samples["sensor"] = samples["sensor"].astype("category")
    samples["window_id"] = samples["window_id"].astype("category")

    samples.to_parquet(samples_path, index=False)
    print(f"Saved {samples_path} ({len(samples):,} rows)")
    return samples


def write_meta(samples: pd.DataFrame) -> dict:
    samples_path = ART / "samples.parquet"
    meta_path = ART / "split_meta.json"

    train_users = sorted(
        samples.loc[samples["split"].astype(str) == "train", "user_id"].unique().tolist()
    )
    test_users = sorted(
        samples.loc[samples["split"].astype(str) == "test", "user_id"].unique().tolist()
    )
    n_windows = int(samples["window_id"].nunique())

    meta = {
        "columns": COLUMNS,
        "window_s": WINDOW_S,
        "min_samples": MIN_SAMPLES,
        "test_user_ids": TEST_USER_IDS,
        "train_user_ids": train_users,
        "n_rows": int(len(samples)),
        "n_windows": n_windows,
        "n_train_rows": int((samples["split"].astype(str) == "train").sum()),
        "n_test_rows": int((samples["split"].astype(str) == "test").sum()),
        "window_id_format": "{user_id}_{device}_{sensor}_{activity}_{bout_window:04d}",
        "window_idx": "0-based sample index within window",
        "artifact": str(samples_path.name),
        "notes": (
            "Share artifacts/samples.parquet + artifacts/split_meta.json. "
            "Do not reshuffle users. Rebuild only with identical constants."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Meta -> {meta_path}")
    return meta


def run_integrity_checks(samples: pd.DataFrame) -> None:
    g = samples.groupby("window_id", observed=True)
    n_per_window = g.size()
    print(
        "Samples/window: min",
        int(n_per_window.min()),
        "median",
        float(n_per_window.median()),
        "max",
        int(n_per_window.max()),
    )
    assert (n_per_window >= MIN_SAMPLES).all()

    def _bad_window_idx(s: pd.Series) -> bool:
        return not (
            int(s.min()) == 0 and int(s.max()) == len(s) - 1 and bool(s.is_unique)
        )

    bad_idx = g["window_idx"].agg(_bad_window_idx)
    print("Windows with bad window_idx:", int(bad_idx.sum()))

    n_splits_per_user = (
        samples.assign(split_str=samples["split"].astype(str))
        .groupby("user_id")["split_str"]
        .nunique()
    )
    mixed_users = n_splits_per_user[n_splits_per_user > 1].index.tolist()
    print("Users spanning both splits:", mixed_users)
    assert len(mixed_users) == 0

    test_users = sorted(
        samples.loc[samples["split"].astype(str) == "test", "user_id"].unique().tolist()
    )
    assert set(test_users) == set(TEST_USER_IDS)
    print("Integrity checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild artifacts/samples.parquet even if it already exists",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip integrity checks",
    )
    args = parser.parse_args()

    print(f"ROOT={ROOT}")
    print(f"TEST_USER_IDS={TEST_USER_IDS}")

    samples = build_samples(force_rebuild=args.force_rebuild)
    meta = write_meta(samples)

    print("\nSplit counts (rows):")
    print(samples["split"].astype(str).value_counts().rename("n_rows").to_string())
    print(f"Unique windows: {meta['n_windows']:,}")
    print(f"Train users ({len(meta['train_user_ids'])}): {meta['train_user_ids'][:5]} ...")
    print(f"Test  users ({len(meta['test_user_ids'])}): {meta['test_user_ids']}")
    print(samples.head(10).to_string(index=False))

    if not args.skip_checks:
        run_integrity_checks(samples)

    train = samples[samples["split"].astype(str) == "train"]
    test = samples[samples["split"].astype(str) == "test"]
    print(f"\ntrain shape={train.shape}, test shape={test.shape}")


if __name__ == "__main__":
    main()

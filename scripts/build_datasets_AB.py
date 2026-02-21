"""
Build Dataset A and Dataset B from the EMG-EPN612 dataset.

Dataset A  — Overall users 1-459:
    • Users 1-306   → trainingJSON/userX/userX.json  → trainingSamples
    • Users 307-459 → testingJSON/userY/userY.json   → trainingSamples
      (userY = overall_user - 306, i.e. user1 … user153 in testingJSON)

Dataset B  — Overall users 460-612:
    • Users 460-612 → testingJSON/userY/userY.json   → trainingSamples
      (userY = overall_user - 306, i.e. user154 … user306 in testingJSON)

Validations performed on every user:
    1. Exactly 150 registrations (idx_1 … idx_150).
    2. Every non-noGesture sample has 'groundTruth' and
       'startPointforGestureExecution' keys.
    3. Every noGesture sample has 'startPointforGestureExecution'.

Output:
    datasets/dataset_A.pkl   — dict  {overall_user_id: trainingSamples_dict}
    datasets/dataset_B.pkl   — dict  {overall_user_id: trainingSamples_dict}

Usage:
    cd <project root>
    python scripts/build_datasets_AB.py
"""

import json
import os
import sys
import pickle
import time
from pathlib import Path

# ── Paths (relative to project root) ────────────────────────────────────────
BASE_PATH      = Path("EMG-EPN612 Dataset")
TRAINING_PATH  = BASE_PATH / "trainingJSON"
TESTING_PATH   = BASE_PATH / "testingJSON"
OUTPUT_DIR     = Path("datasets")
OUTPUT_DIR.mkdir(exist_ok=True)

EXPECTED_SAMPLES = 150  # each user must have exactly 150 registrations


# ── helpers ─────────────────────────────────────────────────────────────────
def load_training_samples(json_path: Path) -> dict:
    """Load and return the trainingSamples dict from a user JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["trainingSamples"]


def validate_user(overall_user_id: int, samples: dict) -> list[str]:
    """Return a list of validation error messages (empty = OK)."""
    errors = []

    # ── check 1: exactly 150 registrations ──────────────────────────────────
    n = len(samples)
    if n != EXPECTED_SAMPLES:
        errors.append(
            f"  User {overall_user_id}: expected {EXPECTED_SAMPLES} samples, "
            f"found {n}"
        )

    # ── check 2 & 3: required keys per sample ──────────────────────────────
    for idx_key, sample in samples.items():
        gesture = sample.get("gestureName", "<MISSING>")
        has_gt  = "groundTruth" in sample
        has_sp  = "startPointforGestureExecution" in sample

        if gesture != "noGesture":
            # non-noGesture → must have groundTruth AND startPoint
            if not has_gt:
                errors.append(
                    f"  User {overall_user_id}, {idx_key} ({gesture}): "
                    f"missing 'groundTruth'"
                )
            if not has_sp:
                errors.append(
                    f"  User {overall_user_id}, {idx_key} ({gesture}): "
                    f"missing 'startPointforGestureExecution'"
                )
        else:
            # noGesture → startPoint should still exist
            if not has_sp:
                errors.append(
                    f"  User {overall_user_id}, {idx_key} (noGesture): "
                    f"missing 'startPointforGestureExecution'"
                )

    return errors


def resolve_json_path(folder: Path, user_folder_name: str) -> Path:
    """Return the path to the JSON file inside a user folder."""
    return folder / user_folder_name / f"{user_folder_name}.json"


# ── main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    dataset_a: dict[int, dict] = {}   # overall_user_id → trainingSamples
    dataset_b: dict[int, dict] = {}
    all_errors: list[str] = []

    total_users = 612
    print("=" * 70)
    print("Building Dataset A (overall users 1-459) and Dataset B (460-612)")
    print("=" * 70)

    # ── Part 1: trainingJSON users 1-306  →  Dataset A ──────────────────────
    print("\n[1/3] Loading trainingJSON users 1-306 …")
    for uid in range(1, 307):
        folder_name = f"user{uid}"
        json_path   = resolve_json_path(TRAINING_PATH, folder_name)
        if not json_path.exists():
            all_errors.append(f"  User {uid}: file not found -> {json_path}")
            continue

        samples = load_training_samples(json_path)
        errs    = validate_user(uid, samples)
        all_errors.extend(errs)
        dataset_a[uid] = samples

        if uid % 50 == 0 or uid == 306:
            sys.stdout.write(f"\r  loaded {uid}/306")
            sys.stdout.flush()

    print(f"\n  [OK] trainingJSON done - {len(dataset_a)} users loaded")

    # ── Part 2: testingJSON users 1-153 (overall 307-459)  →  Dataset A ─────
    print("\n[2/3] Loading testingJSON users 1-153 (overall 307-459) …")
    for local_uid in range(1, 154):
        overall_uid = local_uid + 306          # map to overall user id
        folder_name = f"user{local_uid}"
        json_path   = resolve_json_path(TESTING_PATH, folder_name)
        if not json_path.exists():
            all_errors.append(
                f"  User {overall_uid} (testingJSON/{folder_name}): "
                f"file not found -> {json_path}"
            )
            continue

        samples = load_training_samples(json_path)
        errs    = validate_user(overall_uid, samples)
        all_errors.extend(errs)
        dataset_a[overall_uid] = samples

        if local_uid % 50 == 0 or local_uid == 153:
            sys.stdout.write(f"\r  loaded {local_uid}/153")
            sys.stdout.flush()

    print(f"\n  [OK] testingJSON (part A) done - Dataset A total: {len(dataset_a)} users")

    # ── Part 3: testingJSON users 154-306 (overall 460-612)  →  Dataset B ───
    print("\n[3/3] Loading testingJSON users 154-306 (overall 460-612) …")
    for local_uid in range(154, 307):
        overall_uid = local_uid + 306          # map to overall user id
        folder_name = f"user{local_uid}"
        json_path   = resolve_json_path(TESTING_PATH, folder_name)
        if not json_path.exists():
            all_errors.append(
                f"  User {overall_uid} (testingJSON/{folder_name}): "
                f"file not found -> {json_path}"
            )
            continue

        samples = load_training_samples(json_path)
        errs    = validate_user(overall_uid, samples)
        all_errors.extend(errs)
        dataset_b[overall_uid] = samples

        if local_uid % 50 == 0 or local_uid == 306:
            sys.stdout.write(f"\r  loaded {local_uid - 153}/153")
            sys.stdout.flush()

    print(f"\n  [OK] testingJSON (part B) done - Dataset B total: {len(dataset_b)} users")

    # ── Validation report ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if all_errors:
        print(f"WARNING: {len(all_errors)} validation issues found:\n")
        for e in all_errors[:50]:          # cap console output
            print(e)
        if len(all_errors) > 50:
            print(f"  … and {len(all_errors) - 50} more")
    else:
        print("[OK] All validations passed - no issues found.")

    # ── Summary statistics ──────────────────────────────────────────────────
    total_a_samples = sum(len(v) for v in dataset_a.values())
    total_b_samples = sum(len(v) for v in dataset_b.values())
    print(f"\n  Dataset A: {len(dataset_a)} users, "
          f"{total_a_samples} total registrations")
    print(f"  Dataset B: {len(dataset_b)} users, "
          f"{total_b_samples} total registrations")

    # ── Save ────────────────────────────────────────────────────────────────
    path_a = OUTPUT_DIR / "dataset_A.pkl"
    path_b = OUTPUT_DIR / "dataset_B.pkl"

    with open(path_a, "wb") as f:
        pickle.dump(dataset_a, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Saved Dataset A -> {path_a}  "
          f"({path_a.stat().st_size / 1e6:.1f} MB)")

    with open(path_b, "wb") as f:
        pickle.dump(dataset_b, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved Dataset B -> {path_b}  "
          f"({path_b.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

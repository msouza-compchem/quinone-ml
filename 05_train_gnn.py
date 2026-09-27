#!/usr/bin/env python3
"""
Phase B, step 2 — graph neural network via Chemprop, compared against the
Random Forest baseline.

WHAT THIS SCRIPT DOES
----------------------
1. Rebuilds the SAME random and scaffold splits used for the baseline (same
   seed, same GroupShuffleSplit logic), so the comparison is apples-to-apples.
2. Writes those splits to disk as the CSV files Chemprop expects.
3. Calls Chemprop to train a directed message-passing neural network for IP
   and EA jointly (a single multi-task model, not two separate ones).
4. Evaluates it exactly like the baseline: R^2 and MAE, on both splits, plus
   mu/eta/omega derived from the predicted IP/EA.
5. Prints the two models side by side. If the GNN does not beat the Random
   Forest on the scaffold split, THAT is the result to report -- not a reason
   to hide the comparison.

CHEMPROP VERSION
-----------------
Chemprop's command-line interface changed between v1 (`chemprop_train`,
`chemprop_predict`, underscore-separated flags) and v2 (`chemprop train`,
`chemprop predict`, hyphenated flags). This script detects which is installed
and builds the command accordingly, but CLI flags are exactly the kind of thing
that changes between minor versions. If a command fails, the fix is almost
always to run it with --help and compare against what this script sent --
print the exact command before running it, so that comparison is easy.

Usage:
    python 05_train_gnn.py --smoke-test          # 2 epochs, sanity check only
    python 05_train_gnn.py                       # full training
    python 05_train_gnn.py --epochs 50
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

TARGETS = ["ip", "ea"]


# --------------------------------------------------------------------------
# splits -- identical logic to 04_train_baseline.py, so the comparison is fair
# --------------------------------------------------------------------------
def random_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(np.arange(n), test_size=0.2, random_state=seed)


def scaffold_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def write_split_files(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                      out_dir: Path, split_name: str) -> tuple[Path, Path]:
    """
    Write the columns Chemprop needs: smiles plus every target column.
    Two files per split -- train and test -- so Chemprop trains on one and we
    evaluate on the other exactly once.
    """
    cols = ["smiles"] + TARGETS
    train_path = out_dir / f"{split_name}_train.csv"
    test_path = out_dir / f"{split_name}_test.csv"
    df.iloc[train_idx][cols].to_csv(train_path, index=False)
    df.iloc[test_idx][cols].to_csv(test_path, index=False)
    return train_path, test_path


# --------------------------------------------------------------------------
# chemprop invocation
# --------------------------------------------------------------------------
def detect_chemprop() -> str:
    """
    Return 'v2' or 'v1' depending on what is installed, or exit with
    installation instructions if neither is found.
    """
    if shutil.which("chemprop"):
        return "v2"
    if shutil.which("chemprop_train"):
        return "v1"
    sys.exit(
        "Chemprop not found on PATH.\n"
        "Install with:\n"
        "  pip install chemprop\n"
        "(this pulls in PyTorch and PyTorch Lightning -- expect a large download)"
    )


def train_chemprop(version: str, train_path: Path, save_dir: Path,
                   epochs: int) -> list[str]:
    """
    Build and run the training command for whichever Chemprop version is
    installed. The command is printed before running, so that if it fails,
    the exact thing that was attempted is visible and can be compared against
    'chemprop train --help' / 'chemprop_train --help'.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    if version == "v2":
        # chemprop v2 requires epochs > warmup_epochs (default warmup is 2).
        # Pin warmup explicitly so a low --epochs value (e.g. the smoke test)
        # never collides with that default.
        warmup = max(1, min(2, epochs - 1))
        cmd = [
            "chemprop", "train",
            "--data-path", str(train_path),
            "--task-type", "regression",
            "--target-columns", *TARGETS,
            "--smiles-columns", "smiles",
            "--save-dir", str(save_dir),
            "--epochs", str(epochs),
            "--warmup-epochs", str(warmup),
            "--split", "random",          # we already hold out our own test set;
            "--split-sizes", "0.9", "0.1", "0.0",   # this only carves a val set
        ]
    else:  # v1
        cmd = [
            "chemprop_train",
            "--data_path", str(train_path),
            "--dataset_type", "regression",
            "--target_columns", *TARGETS,
            "--smiles_columns", "smiles",
            "--save_dir", str(save_dir),
            "--epochs", str(epochs),
            "--split_type", "random",
            "--split_sizes", "0.9", "0.1", "0.0",
        ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("---- chemprop stdout (tail) ----")
        print("\n".join(result.stdout.splitlines()[-30:]))
        print("---- chemprop stderr (tail) ----")
        print("\n".join(result.stderr.splitlines()[-30:]))
        sys.exit(
            f"\nTraining failed (exit code {result.returncode}). Compare the "
            f"command above against the output of "
            f"'{'chemprop train' if version == 'v2' else 'chemprop_train'} "
            f"--help' -- flag names are the most common source of mismatch "
            f"between Chemprop versions."
        )
    return cmd


def predict_chemprop(version: str, test_path: Path, save_dir: Path,
                     out_path: Path) -> None:
    """Run inference on the held-out test set with the model just trained."""
    if version == "v2":
        cmd = [
            "chemprop", "predict",
            "--test-path", str(test_path),
            "--model-paths", str(save_dir / "model_0" / "best.pt"),
            "--preds-path", str(out_path),
        ]
    else:
        cmd = [
            "chemprop_predict",
            "--test_path", str(test_path),
            "--checkpoint_dir", str(save_dir),
            "--preds_path", str(out_path),
        ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(f"Prediction failed (exit code {result.returncode}).")


# --------------------------------------------------------------------------
def conceptual_dft_from_predictions(ip: np.ndarray, ea: np.ndarray) -> dict:
    """Same formulas as everywhere else in this project -- see 04_train_baseline.py."""
    mu = -(ip + ea) / 2.0
    eta = (ip - ea) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(eta > 0.05, mu**2 / (2.0 * eta), np.nan)
    return {"mu": mu, "eta": eta, "omega": omega}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"r2": round(r2_score(y_true, y_pred), 3),
            "mae": round(mean_absolute_error(y_true, y_pred), 3)}


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/descriptors_final.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--split-dir", type=Path, default=Path("data/gnn_splits"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true",
                    help="2 epochs on a 500-molecule sample -- run this first")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)

    version = detect_chemprop()
    print(f"Chemprop {version} detected.\n")

    df = pd.read_csv(args.input)
    missing = df["scaffold"].isna().sum()
    if missing:
        df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    epochs = 5 if args.smoke_test else args.epochs
    if args.smoke_test:
        df = df.sample(n=min(500, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"SMOKE TEST: {len(df)} molecules, {epochs} epochs. "
              f"This checks the pipeline runs -- the metrics below are not "
              f"meaningful with this little data.\n")

    print(f"{len(df)} molecules, {df['scaffold'].nunique()} scaffolds\n")

    rows = []
    all_preds = {}   # keyed by split name, holds test-set predictions for mu/eta/omega

    for split_name, idx_fn in [("random", lambda: random_split(len(df), args.seed)),
                               ("scaffold", lambda: scaffold_split(df["scaffold"].values, args.seed))]:
        print(f"\n{'=' * 60}\n{split_name.upper()} SPLIT\n{'=' * 60}")

        train_idx, test_idx = idx_fn()
        train_path, test_path = write_split_files(df, train_idx, test_idx,
                                                   args.split_dir, split_name)
        print(f"  train: {len(train_idx)}   test: {len(test_idx)}")

        save_dir = args.out / f"gnn_{split_name}"
        train_chemprop(version, train_path, save_dir, epochs)

        preds_path = args.out / f"gnn_{split_name}_preds.csv"
        predict_chemprop(version, test_path, save_dir, preds_path)

        preds = pd.read_csv(preds_path)
        truth = pd.read_csv(test_path)

        for target in TARGETS:
            pred_col = [c for c in preds.columns if target in c.lower()]
            if not pred_col:
                print(f"  ! could not find a prediction column for '{target}' "
                      f"in {preds_path} -- columns present: {list(preds.columns)}")
                continue
            m = evaluate(truth[target].values, preds[pred_col[0]].values)
            m.update(target=target, split=split_name)
            rows.append(m)
            print(f"  {target:3s}  R2 {m['r2']:.3f}   MAE {m['mae']:.3f} eV")

        all_preds[split_name] = {
            "ip": preds[[c for c in preds.columns if "ip" in c.lower()][0]].values,
            "ea": preds[[c for c in preds.columns if "ea" in c.lower()][0]].values,
            "true_ip": truth["ip"].values,
            "true_ea": truth["ea"].values,
        }

    report = pd.DataFrame(rows)
    report.to_csv(args.out / "gnn_metrics.csv", index=False)

    if args.smoke_test:
        print("\nSmoke test finished without crashing. Run without --smoke-test "
              "for the full training and comparison.")
        return

    # ---- derived descriptors, scaffold split only, same as the baseline ----
    p = all_preds["scaffold"]
    pred_desc = conceptual_dft_from_predictions(p["ip"], p["ea"])
    true_desc = conceptual_dft_from_predictions(p["true_ip"], p["true_ea"])

    print("\nDerived-descriptor accuracy (GNN, scaffold-split test set):")
    for name in ["mu", "eta", "omega"]:
        mask = ~np.isnan(pred_desc[name]) & ~np.isnan(true_desc[name])
        m = evaluate(true_desc[name][mask], pred_desc[name][mask])
        print(f"  {name:6s}  R2 {m['r2']:.3f}   MAE {m['mae']:.3f} eV")

    # ---- the comparison this whole script exists to make ------------------
    baseline_path = args.out / "baseline_metrics.csv"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        print("\n" + "=" * 62)
        print("Random Forest vs. GNN -- scaffold-split R2 (the honest number)")
        print("=" * 62)
        for target in TARGETS:
            b = base[(base.target == target) & (base.split == "scaffold")]
            g = report[(report.target == target) & (report.split == "scaffold")]
            if len(b) and len(g):
                b_r2, g_r2 = b.iloc[0].r2, g.iloc[0].r2
                verdict = "GNN wins" if g_r2 > b_r2 else "Random Forest wins"
                print(f"  {target}:  RF {b_r2:.3f}   GNN {g_r2:.3f}   "
                      f"({verdict} by {abs(g_r2 - b_r2):.3f})")
        print("\nIf the Random Forest wins or the two are within noise, that is")
        print("the result to report: a fixed 2048-bit fingerprint already")
        print("captures most of what a learned representation offers here.")
    else:
        print(f"\n(No {baseline_path} found -- run 04_train_baseline.py first "
              f"for a side-by-side comparison.)")


if __name__ == "__main__":
    main()

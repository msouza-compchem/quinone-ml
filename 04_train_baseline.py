#!/usr/bin/env python3
"""
Phase B, step 1 — baseline model: Random Forest on Morgan fingerprints.

WHAT THIS SCRIPT DOES
----------------------
Trains two regressors -- one for IP, one for EA -- using Morgan fingerprints as
input. Each is validated TWICE: once with a random train/test split (the
optimistic number everyone reports) and once with a scaffold split (the honest
number, which is usually lower). Reporting both, side by side, is the point of
this whole phase.

mu, eta and omega are then computed from the PREDICTED IP and EA, exactly as
they were computed from the true IP and EA when the dataset was built. This
keeps the model from having to learn eta and omega directly, which carry more
noise than IP and EA do (as measured against the lapachol DFT reference: mu
error 1%, eta error 24%, omega error 34%).

WHY A RANDOM FOREST FIRST
--------------------------
A random forest on fixed fingerprints is fast to train, has almost no
hyperparameters to tune, and is often hard to beat with 10,000-scale datasets.
It is the benchmark the GNN in the next script has to justify itself against --
if the GNN does not clearly outperform this, the fingerprint model is the one to
report as the main result, and that is a legitimate finding, not a failure.

Usage:
    python 04_train_baseline.py
    python 04_train_baseline.py --n-estimators 500      # slower, usually better
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

RDLogger.DisableLog("rdApp.*")

TARGETS = ["ip", "ea"]
FP_RADIUS = 2          # "ECFP4" in the older naming -- radius 2 means each bit
                       # encodes an environment up to 2 bonds from its atom
FP_BITS = 2048


# --------------------------------------------------------------------------
def morgan_fingerprint(smiles: str) -> np.ndarray | None:
    """
    Convert one SMILES string into a fixed-length bit vector.

    Each of the 2048 positions answers "is a particular local atomic
    environment present in this molecule?". Two molecules that share many such
    environments will have similar vectors, which is what lets a model learn
    from them at all.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    arr = np.zeros(FP_BITS, dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute fingerprints for every molecule, dropping any that fail to parse.

    Returns the feature matrix and the matching (possibly shrunk) dataframe, so
    the two always stay aligned by row position -- a common source of subtle
    bugs is a feature matrix and a label column that silently drift out of sync
    after a filtering step.
    """
    feats, keep_idx = [], []
    for i, smi in enumerate(df["smiles"]):
        fp = morgan_fingerprint(smi)
        if fp is not None:
            feats.append(fp)
            keep_idx.append(i)
    X = np.array(feats)
    kept_df = df.iloc[keep_idx].reset_index(drop=True)
    if len(kept_df) < len(df):
        print(f"  ! {len(df) - len(kept_df)} molecules dropped "
              f"(fingerprint could not be computed)")
    return X, kept_df


def conceptual_dft_from_predictions(ip: np.ndarray, ea: np.ndarray) -> dict:
    """
    Same finite-difference formulas used when the dataset was built:
        mu = -(IP+EA)/2, eta = (IP-EA)/2, omega = mu^2 / (2 eta)

    Applying them to PREDICTED ip/ea, rather than training directly on
    mu/eta/omega, is the whole point of this design -- see the module
    docstring.
    """
    mu = -(ip + ea) / 2.0
    eta = (ip - ea) / 2.0
    # a few predictions can give eta close to zero; guard the division
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(eta > 0.05, mu**2 / (2.0 * eta), np.nan)
    return {"mu": mu, "eta": eta, "omega": omega}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R^2 and mean absolute error, the two numbers this whole phase turns on."""
    return {
        "r2": round(r2_score(y_true, y_pred), 3),
        "mae": round(mean_absolute_error(y_true, y_pred), 3),
    }


# --------------------------------------------------------------------------
def random_split(X, y, groups, seed):
    """
    The split everyone reports first, and the one that overstates performance.

    train_test_split shuffles rows independently of which scaffold they belong
    to, so near-identical molecules routinely end up on both sides.
    """
    return train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)


def scaffold_split(X, y, groups, seed):
    """
    The split that actually measures generalisation to new chemistry.

    GroupShuffleSplit guarantees that every row sharing the same 'groups' value
    (here, the Murcko scaffold) lands entirely in train OR entirely in test,
    never split across the two. A model can no longer succeed by having
    memorised a close relative of the test molecule -- it has to have learned
    something that transfers to a scaffold it never saw.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/descriptors_final.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n-estimators", type=int, default=200,
                    help="trees in the forest; more is slower but usually better")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} molecules, {df['scaffold'].nunique()} scaffolds")

    missing = df["scaffold"].isna().sum()
    if missing:
        print(f"  ! {missing} molecules have no scaffold recorded and are excluded")
        df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    print("\nComputing Morgan fingerprints...")
    t0 = time.time()
    X, df = build_features(df)
    print(f"  {X.shape[0]} molecules x {X.shape[1]} bits "
          f"({time.time() - t0:.1f}s)")

    groups = df["scaffold"].values
    rows = []          # one row per (target, split-type) combination, for the report

    for target in TARGETS:
        y = df[target].values

        for split_name, split_fn in [("random", random_split),
                                      ("scaffold", scaffold_split)]:
            train_idx, test_idx = split_fn(X, y, groups, args.seed)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                n_jobs=-1,                    # use every CPU core available
                random_state=args.seed,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = evaluate(y_test, y_pred)
            metrics.update(target=target, split=split_name,
                           n_train=len(train_idx), n_test=len(test_idx))
            rows.append(metrics)

            model_path = args.out / f"rf_{target}_{split_name}.joblib"
            joblib.dump(model, model_path)

            print(f"  {target:3s} / {split_name:8s} split  ->  "
                  f"R2 {metrics['r2']:.3f}   MAE {metrics['mae']:.3f} eV")

    report = pd.DataFrame(rows)
    report.to_csv(args.out / "baseline_metrics.csv", index=False)

    # ---- the number that matters most: how much the scaffold split costs ---
    print("\n" + "=" * 62)
    print("Random vs. scaffold split -- the gap IS the result")
    print("=" * 62)
    for target in TARGETS:
        r = report[(report.target == target) & (report.split == "random")].iloc[0]
        s = report[(report.target == target) & (report.split == "scaffold")].iloc[0]
        drop = r.r2 - s.r2
        print(f"  {target}:  random R2 {r.r2:.3f}  ->  scaffold R2 {s.r2:.3f}  "
              f"(drop of {drop:.3f})")

    # ---- conceptual-DFT descriptors from the scaffold-split predictions ----
    # Only the scaffold split is used here: it is the honest estimate, and the
    # one that should be quoted anywhere this model's real-world performance
    # is discussed.
    print("\nDeriving mu, eta, omega from scaffold-split IP/EA predictions...")
    model_ip = joblib.load(args.out / "rf_ip_scaffold.joblib")
    model_ea = joblib.load(args.out / "rf_ea_scaffold.joblib")

    _, test_idx = scaffold_split(X, df["ip"].values, groups, args.seed)
    X_test = X[test_idx]
    true_sub = df.iloc[test_idx]

    pred_ip = model_ip.predict(X_test)
    pred_ea = model_ea.predict(X_test)
    pred_desc = conceptual_dft_from_predictions(pred_ip, pred_ea)
    true_desc = conceptual_dft_from_predictions(
        true_sub["ip"].values, true_sub["ea"].values)

    print("\nDerived-descriptor accuracy (scaffold-split test set):")
    for name in ["mu", "eta", "omega"]:
        mask = ~np.isnan(pred_desc[name]) & ~np.isnan(true_desc[name])
        m = evaluate(true_desc[name][mask], pred_desc[name][mask])
        print(f"  {name:6s}  R2 {m['r2']:.3f}   MAE {m['mae']:.3f} eV   "
              f"({mask.sum()} of {len(mask)} usable)")

    print(f"\nWrote {args.out / 'baseline_metrics.csv'} and "
          f"{args.out}/rf_*.joblib")
    print("\nReference for comparison -- error against B3LYP-D4/def2-TZVP for")
    print("lapachol itself: mu 1%, eta 24%, omega 34% (single-point, not a")
    print("test-set average, but the right order of magnitude to expect).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase A, step 2 — GFN2-xTB electronic descriptors from vertical IP and EA.

Why not the orbital energies
----------------------------
GFN2-xTB orbital eigenvalues are a poor proxy for DFT here. Benchmarked against
the B3LYP-D4/def2-TZVP reference for lapachol:

    HOMO-LUMO gap   DFT 3.127 eV   ·   xTB eigenvalues 0.966 eV   ·   error 2.16
    electron affin. DFT 3.770 eV   ·   xTB --vipea     3.677 eV   ·   error 0.09

The minimal basis compresses gaps badly, but vertical IP and EA are computed as
total-energy differences between the neutral and the charged species, and those
the method reproduces well. So the descriptors here are built from IP and EA.

Conceptual-DFT quantities then follow from the finite-difference definitions,
which is their original form — no Koopmans approximation involved:

    mu    = -(IP + EA) / 2          chemical potential
    eta   =  (IP - EA) / 2          chemical hardness
    omega =  mu^2 / (2 * eta)       electrophilicity index

Cost: --vipea runs three SCC calculations instead of one, roughly 10-20 s per
molecule. The run is resumable — results are appended as they finish.

Usage:
    python 02_xtb_descriptors.py --subset 5000
    python 02_xtb_descriptors.py --subset 5000 --mode opt      # relaxed geometry
    python 02_xtb_descriptors.py --limit 20                    # smoke test
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

FIELDS = ["smiles", "scaffold", "core", "mode",
          "ip", "ea", "mu", "eta", "omega",
          "homo", "lumo", "gap", "e_total", "status"]


# --------------------------------------------------------------------------
def to_xyz(smiles: str, path: Path) -> bool:
    """Generate a 3D structure with ETKDG plus a short MMFF relaxation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return False

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), smiles]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return True


def parse_xtb(text: str) -> dict:
    """Extract IP, EA and, for reference, the orbital energies."""
    out: dict[str, float] = {}

    m = re.search(r"delta SCC IP \(eV\):\s+(-?[\d.]+)", text)
    if m:
        out["ip"] = float(m.group(1))

    m = re.search(r"delta SCC EA \(eV\):\s+(-?[\d.]+)", text)
    if m:
        out["ea"] = float(m.group(1))

    # Orbital energies are recorded too, so the two routes can be compared
    m = re.search(r"^\s*\d+\s+[\d.]+\s+-?[\d.]+\s+(-?[\d.]+)\s+\(HOMO\)",
                  text, re.M)
    if m:
        out["homo"] = float(m.group(1))
    m = re.search(r"^\s*\d+\s+(?:[\d.]+\s+)?-?[\d.]+\s+(-?[\d.]+)\s+\(LUMO\)",
                  text, re.M)
    if m:
        out["lumo"] = float(m.group(1))
    m = re.search(r"HL-Gap\s+-?[\d.]+\s+Eh\s+(-?[\d.]+)\s+eV", text)
    if m:
        out["gap"] = float(m.group(1))
    m = re.search(r"TOTAL ENERGY\s+(-?[\d.]+)\s+Eh", text)
    if m:
        out["e_total"] = float(m.group(1))

    return out


def conceptual_dft(ip: float, ea: float) -> dict:
    """
    Finite-difference conceptual-DFT descriptors.

    eta can approach zero for a species whose IP and EA are close, which would
    send omega to infinity. Such cases are flagged rather than reported.
    """
    mu = -(ip + ea) / 2.0
    eta = (ip - ea) / 2.0
    if eta <= 0.05:
        return {"mu": round(mu, 4), "eta": round(eta, 4), "omega": None}
    return {"mu": round(mu, 4), "eta": round(eta, 4),
            "omega": round(mu * mu / (2.0 * eta), 4)}


def run_one(args: tuple) -> dict:
    """Process a single molecule. Runs in a worker process."""
    smiles, scaffold, core, mode, xtb_bin, timeout = args
    record = {"smiles": smiles, "scaffold": scaffold, "core": core,
              "mode": mode, "status": "ok"}

    workdir = Path(tempfile.mkdtemp(prefix="xtb_"))
    try:
        xyz = workdir / "mol.xyz"
        if not to_xyz(smiles, xyz):
            record["status"] = "embed_failed"
            return record

        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")

        # Optional geometry relaxation before the property calculation
        geom = "mol.xyz"
        if mode == "opt":
            p = subprocess.run(
                [xtb_bin, "mol.xyz", "--gfn", "2", "--alpb", "water", "--opt"],
                cwd=workdir, capture_output=True, text=True,
                timeout=timeout, env=env)
            if (workdir / "xtbopt.xyz").exists():
                geom = "xtbopt.xyz"
            elif p.returncode:
                record["status"] = "opt_failed"
                return record

        proc = subprocess.run(
            [xtb_bin, geom, "--gfn", "2", "--alpb", "water", "--vipea"],
            cwd=workdir, capture_output=True, text=True,
            timeout=timeout, env=env)

        parsed = parse_xtb(proc.stdout)
        if "ip" not in parsed or "ea" not in parsed:
            record["status"] = "scf_failed" if proc.returncode else "parse_failed"
            return record

        record.update(parsed)
        record.update(conceptual_dft(parsed["ip"], parsed["ea"]))
        return record

    except subprocess.TimeoutExpired:
        record["status"] = "timeout"
        return record
    except Exception as e:
        record["status"] = f"error:{type(e).__name__}"
        return record
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def stratified_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Draw a subset that keeps each quinone core in proportion and spreads across
    scaffolds, so the training set is not dominated by one structural family.
    """
    per_core = (df["core"].value_counts(normalize=True) * n).round().astype(int)
    chunks = []
    for core, k in per_core.items():
        pool = df[df["core"] == core]
        first = pool.groupby("scaffold", group_keys=False).head(1)   # one per scaffold
        if len(first) >= k:
            chunks.append(first.sample(n=k, random_state=seed))
        else:
            rest = pool.drop(first.index)
            extra = rest.sample(n=min(k - len(first), len(rest)), random_state=seed)
            chunks.append(pd.concat([first, extra]))
    return pd.concat(chunks).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["sp", "opt"], default="sp")
    ap.add_argument("--input", type=Path, default=Path("data/quinones.csv"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    xtb_bin = shutil.which("xtb")
    if not xtb_bin:
        sys.exit("xtb not found on PATH. conda install -c conda-forge xtb")

    timeout = args.timeout or (300 if args.mode == "sp" else 900)
    out_path = args.out or Path(f"data/descriptors_{args.mode}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.subset:
        df = stratified_subset(df, args.subset, args.seed)
        print(f"Stratified subset: {len(df)} molecules, "
              f"{df['scaffold'].nunique()} scaffolds")
        subset_path = out_path.with_name(f"subset_{args.subset}.csv")
        df.to_csv(subset_path, index=False)      # record exactly what was chosen
        print(f"  subset written to {subset_path}")
    if args.limit:
        df = df.head(args.limit)

    done: set[str] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(prev["smiles"])
        print(f"Resuming: {len(done)} molecules already computed")

    todo = df[~df["smiles"].isin(done)]
    if todo.empty:
        print("Nothing left to do.")
        return

    print(f"\nMode: {args.mode}  ·  {len(todo)} molecules  ·  "
          f"{args.workers} workers  ·  timeout {timeout}s")
    print("Descriptors from vertical IP and EA. Safe to interrupt and restart.\n")

    jobs = [(r.smiles, r.scaffold, r.core, args.mode, xtb_bin, timeout)
            for r in todo.itertuples()]

    new_file = not out_path.exists()
    t0 = time.time()
    counts = {"ok": 0, "failed": 0}

    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                writer.writerow(rec)
                counts["ok" if rec["status"] == "ok" else "failed"] += 1
                if i % 25 == 0 or i == len(jobs):
                    fh.flush()
                    rate = i / (time.time() - t0)
                    eta = (len(jobs) - i) / rate / 3600
                    print(f"  {i}/{len(jobs)}  ok {counts['ok']}  "
                          f"failed {counts['failed']}  "
                          f"{rate * 60:.1f} mol/min  eta {eta:.1f} h", flush=True)

    # ---- summary ---------------------------------------------------------
    res = pd.read_csv(out_path)
    good = res[res["status"] == "ok"]
    print(f"\nWrote {out_path}")
    print(f"  successful : {len(good)} / {len(res)} "
          f"({100 * len(good) / len(res):.1f}%)")
    if len(res) > len(good):
        print("\n  failures by cause:")
        print(res[res["status"] != "ok"]["status"].value_counts().to_string())

    if len(good):
        print("\nDescriptors (eV):")
        cols = [c for c in ["ip", "ea", "mu", "eta", "omega"] if c in good]
        print(good[cols].describe().loc[["mean", "std", "min", "max"]]
              .round(3).to_string())
        print("\nReference — lapachol at B3LYP-D4/def2-TZVP, CPCM(water):")
        print("  EA 3.770   mu -4.819   eta 1.564   omega 7.425  (eV)")


if __name__ == "__main__":
    main()

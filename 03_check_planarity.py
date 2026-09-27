#!/usr/bin/env python3
"""
Phase A, quality control — check ring planarity of the generated 3D geometries.

WHY THIS SCRIPT EXISTS
-----------------------
ETKDG plus a short MMFF relaxation gives a geometry that is chemically
reasonable, but nothing in that pipeline *enforces* planarity. A quinone ring is
an aromatic, conjugated system and should come out flat. If the embedding
algorithm had trouble with a particular molecule -- a crowded substituent
pattern, an unusual ring fusion -- it can produce a ring that is visibly puckered
even though RDKit reports no error.

A model trained on a handful of badly-puckered geometries mixed in with 11,500
good ones will not fail loudly. It will just be a little bit wrong, in a way
that is very hard to trace back to its source. This script finds those cases
before they reach the model.

WHAT "PLANARITY" MEANS HERE, GEOMETRICALLY
--------------------------------------------
Take every atom in the ring. If they lay on a perfect plane, each atom's
distance to that plane would be exactly zero. In practice it never is, because
bond lengths and angles are not perfectly idealised -- but for an aromatic ring
that distance should be a small fraction of an angstrom. A ring where some
atoms sit far above or below the mean plane is not aromatic-flat, and its
xTB descriptors should not be trusted at face value.

The plane itself is found by Singular Value Decomposition (SVD), which is the
standard, numerically stable way to fit a plane through a cloud of points --
the same technique used to measure the FAD/lapachol ring angle in the
molecular-dynamics project.

Usage:
    python 03_check_planarity.py
    python 03_check_planarity.py --threshold 0.15   # stricter cutoff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")     # RDKit prints a lot of routine parsing noise


def embed_3d(smiles: str, seed: int = 42) -> Chem.Mol | None:
    """
    Rebuild the same 3D geometry the descriptor script used.

    This MUST use the same random seed and the same embedding parameters as
    02_xtb_descriptors.py. If it did not, we would be checking the planarity of
    a *different* geometry than the one that was actually sent to xTB, and the
    quality check would be meaningless.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)                       # xTB needs explicit hydrogens

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None                          # embedding failed outright

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass                                     # MMFF can lack parameters for some atoms

    return mol


def find_quinone_ring_atoms(mol: Chem.Mol) -> list[int] | None:
    """
    Return the atom indices of the ring that carries the two quinone carbonyls.

    A molecule can have several rings (the naphthoquinone core plus whatever
    substituents were attached). We do not want to test "the ring" in the
    abstract -- we want the specific ring that defines the quinone's
    electronic character, because that is the one whose flatness the
    descriptors depend on.

    The SMARTS pattern below reads as: a ring carbon double-bonded to oxygen
    (a carbonyl), somewhere in an aromatic or conjugated six-membered ring.
    RDKit's ring-finding then hands back the atom indices of the smallest ring
    containing that carbonyl carbon.
    """
    # Matches a ring atom that is a carbonyl carbon: C(=O) where the C is in a ring
    patt = Chem.MolFromSmarts("[#6;R]=[OX1]")
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return None

    ring_info = mol.GetRingInfo()
    carbonyl_carbon = matches[0][0]              # first match, the carbon atom

    # Find the smallest ring that contains this carbonyl carbon
    for ring in ring_info.AtomRings():
        if carbonyl_carbon in ring and len(ring) == 6:
            return list(ring)
    return None


def ring_planarity_deviation(mol: Chem.Mol, ring_atoms: list[int]) -> float:
    """
    Return the RMS distance of the ring atoms to their best-fit plane, in
    angstrom. Zero means perfectly flat.

    The method:
      1. Take the 3D coordinates of just the ring atoms.
      2. Center them on their own centroid (subtract the mean) -- this is
         necessary because SVD finds a plane THROUGH THE ORIGIN, and we want
         the plane through the ring's own center, not through (0,0,0).
      3. Run SVD. The three singular vectors it returns are, in order, the
         directions of decreasing spread in the data. For a flat ring, the
         first two vectors span the ring plane, and the third -- the one with
         the LEAST spread -- is the normal vector perpendicular to that plane.
      4. Project every atom's centered coordinate onto that normal vector.
         That projection IS the atom's height above or below the plane.
      5. Take the root-mean-square of those heights: one number summarizing
         how far, on average, the ring atoms sit off the flat plane.
    """
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in ring_atoms])

    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # Vt's rows are the principal directions, ordered by how much of the
    # spread in the data each one explains. Vt[2] -- the last one -- is the
    # direction along which the ring atoms vary LEAST: the normal to the
    # best-fit plane.
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[2]

    # Height of each atom above/below the plane = its coordinate along the
    # normal direction. A perfectly flat ring gives all zeros here.
    heights = centered @ normal                  # matrix-vector product

    return float(np.sqrt(np.mean(heights ** 2)))  # root-mean-square


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/descriptors_clean.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/planarity_check.csv"))
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="RMS deviation (A) above which a ring is flagged")
    ap.add_argument("--limit", type=int, default=None, help="smoke test")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"Checking {len(df)} molecules for ring planarity...")

    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        result = {"smiles": row.smiles, "deviation": None, "status": "ok"}

        mol = embed_3d(row.smiles)
        if mol is None:
            result["status"] = "embed_failed"
            rows.append(result)
            continue

        ring = find_quinone_ring_atoms(mol)
        if ring is None:
            result["status"] = "no_quinone_ring_found"
            rows.append(result)
            continue

        result["deviation"] = round(ring_planarity_deviation(mol, ring), 4)
        rows.append(result)

        if i % 500 == 0:
            print(f"  {i}/{len(df)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    checked = out[out.deviation.notna()]
    flagged = checked[checked.deviation > args.threshold]

    print(f"\nMolecules with a usable ring measurement: {len(checked)} / {len(out)}")
    print(f"Deviation (A): mean {checked.deviation.mean():.3f}  "
          f"median {checked.deviation.median():.3f}  "
          f"max {checked.deviation.max():.3f}")
    print(f"\nFlagged as non-planar (> {args.threshold} A): {len(flagged)} "
          f"({100 * len(flagged) / len(checked):.1f}%)")

    if len(flagged):
        print("\nWorst five:")
        print(flagged.nlargest(5, "deviation")[["smiles", "deviation"]]
              .to_string(index=False))
        print(f"\nMerge this file with descriptors_clean.csv on 'smiles' and "
              f"drop rows with deviation > {args.threshold} before training, "
              f"or keep them and record the criterion in the README.")


if __name__ == "__main__":
    main()

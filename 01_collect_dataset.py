#!/usr/bin/env python3
"""
Phase A, step 1 — assemble and curate a quinone dataset.

Retrieves quinone-containing structures from PubChem by substructure search,
filters them to a size range that quantum calculations can handle, removes
duplicates and salts, and writes a clean SMILES table ready for xTB.

Run this BEFORE writing any model code. The size and quality of what comes out
determines whether the project is viable as planned.

Usage:
    python 01_collect_dataset.py                    # full run
    python 01_collect_dataset.py --limit 200        # quick test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")          # silence RDKit parsing chatter

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Substructure queries. Each targets a quinone core; together they span the
# classes relevant to NQO1 chemistry.
CORES = {
    "1,4-benzoquinone":   "O=C1C=CC(=O)C=C1",
    "1,2-benzoquinone":   "O=C1C(=O)C=CC=C1",
    "1,4-naphthoquinone": "O=C1C=CC(=O)c2ccccc12",
    "1,2-naphthoquinone": "O=C1C(=O)c2ccccc2C=C1",
    "anthraquinone":      "O=C1c2ccccc2C(=O)c2ccccc12",
}

# Size window: small enough for DFT later, large enough to be interesting.
MIN_HEAVY, MAX_HEAVY = 10, 40
MAX_MW = 600.0


def substructure_search(smarts: str, name: str, limit: int) -> list[str]:
    """
    Run an asynchronous substructure search on PubChem and return CIDs.

    PubChem answers these queries in two steps: the first call returns a listkey,
    and the results are polled until the job finishes. The polling loop below is
    deliberately patient — the service is free and rate-limited.
    """
    # PubChem rejects a substructure search that carries an operation in the
    # path ("Search input may not be combined with any operation"), and it
    # rejects a SMILES embedded in the URL. So: POST the SMILES, ask only for a
    # format, receive a listkey, then poll that listkey for the CIDs.
    try:
        r = requests.post(f"{PUBCHEM}/compound/substructure/smiles/JSON",
                          data={"smiles": smarts, "MaxRecords": limit},
                          timeout=180)
        if r.status_code not in (200, 202):
            print(f"  ! {name}: search rejected ({r.status_code}) {r.text[:120]}")
            return []
        data = r.json()
    except Exception as e:
        print(f"  ! {name}: search failed ({e})")
        return []

    if "Waiting" not in data:
        print(f"  ! {name}: unexpected response {str(data)[:120]}")
        return []

    key = data["Waiting"]["ListKey"]
    data = {}
    for _ in range(40):
        time.sleep(4)
        try:
            p = requests.get(f"{PUBCHEM}/compound/listkey/{key}/cids/JSON",
                             timeout=120)
            if p.status_code == 200 and "IdentifierList" in p.text:
                data = p.json()
                break
        except Exception:
            continue
    else:
        print(f"  ! {name}: listkey {key} timed out")
        return []

    cids = data.get("IdentifierList", {}).get("CID", [])
    print(f"  {name}: {len(cids)} hits")
    return [str(c) for c in cids]


def fetch_smiles(cids: list[str], chunk: int = 100) -> dict[str, str]:
    """
    Retrieve SMILES for a list of CIDs, in polite chunks.

    PubChem renamed its SMILES properties: a request for CanonicalSMILES is
    still accepted, but the response comes back keyed as ConnectivitySMILES or
    simply SMILES depending on what was asked. We request the current names and
    accept whichever key the server returns.
    """
    out: dict[str, str] = {}
    for i in range(0, len(cids), chunk):
        batch = ",".join(cids[i:i + chunk])
        url = f"{PUBCHEM}/compound/cid/{batch}/property/SMILES,ConnectivitySMILES/JSON"
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            for rec in r.json()["PropertyTable"]["Properties"]:
                smi = (rec.get("ConnectivitySMILES")
                       or rec.get("SMILES")
                       or rec.get("CanonicalSMILES"))
                if smi:
                    out[str(rec["CID"])] = smi
        except Exception as e:
            print(f"  ! chunk {i // chunk} failed ({e})")
        time.sleep(0.3)                          # stay under the rate limit
        if (i // chunk) % 10 == 0 and i:
            print(f"    fetched {len(out)} SMILES...")
    return out


def curate(smiles: str) -> dict | None:
    """
    Clean one structure and compute the bookkeeping fields.

    Returns None when the molecule should be dropped, so the caller can count
    losses by category rather than silently losing compounds.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Strip counter-ions: a salt is not a different electronic structure
    mol = SaltRemover.SaltRemover().StripMol(mol, dontRemoveEverything=True)
    if mol.GetNumAtoms() == 0:
        return None

    # Keep only the largest fragment, for mixtures that survived the stripper
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    heavy = mol.GetNumHeavyAtoms()
    mw = Descriptors.MolWt(mol)
    if not (MIN_HEAVY <= heavy <= MAX_HEAVY) or mw > MAX_MW:
        return None

    # Elements: quantum methods here are parameterised for organic chemistry
    allowed = {"C", "H", "N", "O", "S", "F", "Cl", "Br"}
    if any(a.GetSymbol() not in allowed for a in mol.GetAtoms()):
        return None

    # Charged species need different treatment downstream; exclude for now
    if Chem.GetFormalCharge(mol) != 0:
        return None

    canonical = Chem.MolToSmiles(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    return {
        "smiles": canonical,
        "scaffold": scaffold,               # needed for the scaffold split later
        "heavy_atoms": heavy,
        "mw": round(mw, 2),
        "n_rotatable": Descriptors.NumRotatableBonds(mol),
        "n_rings": Descriptors.RingCount(mol),
        "logp": round(Descriptors.MolLogP(mol), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=2000,
                    help="max hits per substructure query")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 1. search --------------------------------------------------------
    print("Searching PubChem by substructure:")
    cid_to_core: dict[str, str] = {}
    for name, smarts in CORES.items():
        for cid in substructure_search(smarts, name, args.limit):
            cid_to_core.setdefault(cid, name)    # first core wins
    print(f"\nUnique CIDs across all cores: {len(cid_to_core)}")

    if not cid_to_core:
        print("\nNothing retrieved. Check the network, or run again later —")
        print("PubChem substructure search is asynchronous and sometimes busy.")
        return

    # ---- 2. fetch ---------------------------------------------------------
    print("\nFetching SMILES:")
    smiles_map = fetch_smiles(list(cid_to_core))
    print(f"  retrieved {len(smiles_map)}")

    # ---- 3. curate --------------------------------------------------------
    print("\nCurating:")
    rows, dropped = [], 0
    for cid, smi in smiles_map.items():
        rec = curate(smi)
        if rec is None:
            dropped += 1
            continue
        rec["cid"] = cid
        rec["core"] = cid_to_core[cid]
        rows.append(rec)

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)

    print(f"  dropped by filters      : {dropped}")
    print(f"  duplicate structures    : {before - len(df)}")
    print(f"  final dataset           : {len(df)}")

    # ---- 4. report --------------------------------------------------------
    path = args.out / "quinones.csv"
    df.to_csv(path, index=False)
    print(f"\nWrote {path}")

    if df.empty:
        print("\nNo molecules survived. Inspect the messages above before rerunning.")
        return

    print("\nBy core:")
    print(df["core"].value_counts().to_string())
    print(f"\nUnique Murcko scaffolds: {df['scaffold'].nunique()}")
    print(f"Heavy atoms: median {df['heavy_atoms'].median():.0f}, "
          f"range {df['heavy_atoms'].min()}–{df['heavy_atoms'].max()}")

    # The number that decides the project
    print("\n" + "=" * 62)
    n, s = len(df), df["scaffold"].nunique()
    if n >= 1500 and s >= 150:
        print("Dataset is large and diverse enough for the planned GNN.")
    elif n >= 500:
        print("Workable, but tight for a GNN. Compare against a random-forest")
        print("baseline on Morgan fingerprints before committing to the network.")
    else:
        print("Too small for deep learning. Either widen the substructure")
        print("queries, or reduce the scope to a fingerprint-based model.")
    print(f"  {n} molecules · {s} distinct scaffolds")
    print("=" * 62)


if __name__ == "__main__":
    main()

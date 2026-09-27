# quinone-ml: a surrogate model for quinone electronic descriptors

A machine-learned surrogate for the electronic descriptors that predict quinone
reactivity toward NQO1 — built to scale the single-molecule study in
[lapachol-nqo1](https://github.com/msouza-compchem/lapachol-nqo1) to a chemical
family, at a cost DFT cannot sustain across thousands of compounds.

---

## 1. The problem

NQO1 bioactivates some quinones and detoxifies others. The lapachol study
showed that electronic descriptors — the electrophilicity index ω above all —
correlate with which branch a molecule falls into. DFT gives those descriptors
reliably, but costs hours per molecule. Screening a natural-product library of
thousands of quinones at that cost is not feasible on the hardware available
here.

This project builds a three-tier pipeline instead: a semi-empirical method
screens cheaply across thousands of molecules, and a machine-learned model
is trained to reproduce those descriptors even faster, with the error this
approximation costs measured rather than assumed.

---

## 2. Phase A — dataset

**11,614** quinone structures (1,4- and 1,2-benzoquinones, 1,4- and
1,2-naphthoquinones, anthraquinones) were retrieved from PubChem by
substructure search and curated: salts stripped, size-filtered to
10–40 heavy atoms and ≤600 Da, restricted to common organic elements, and
deduplicated.

**GFN2-xTB** (ALPB water) was run on each structure to obtain vertical
ionisation potential (IP) and electron affinity (EA) via the `--vipea`
delta-SCF protocol — **not** from orbital eigenvalues. This choice was
validated against the lapachol DFT reference (B3LYP-D4/def2-TZVP, CPCM water):

| Descriptor | DFT (adiabatic) | xTB eigenvalues | xTB `--vipea` (vertical) |
|---|---|---|---|
| HOMO–LUMO gap / IP–EA gap | 3.127 eV | 0.966 eV (69% error) | — |
| Electron affinity | 3.770 eV | −3.255 eV (LUMO) | 3.677 eV (**2% error**) |

Orbital eigenvalues from a minimal-basis semi-empirical method are a poor proxy
for DFT frontier orbitals; total-energy differences (IP, EA) are not, because
systematic errors in the neutral and charged-species energies largely cancel.
Conceptual-DFT descriptors were then derived from IP and EA by their original
finite-difference definitions — no Koopmans approximation:

```
mu    = -(IP + EA) / 2
eta   =  (IP - EA) / 2
omega =  mu^2 / (2 * eta)
```

**Quality control**, in order:

| Step | Excluded | Reason |
|---|---|---|
| xTB convergence | 22 (0.2%) | 3D embedding failure |
| Electronic outliers | 89 (0.8%) | EA < 0 or η < 0.6 eV — highly conjugated polycyclic systems where the finite-difference electrophilicity diverges |
| Ring non-planarity | 73 (0.7%) | RMS deviation from the mean quinone-ring plane > 0.20 Å (SVD-fitted plane), typically sterically crowded fused-ring systems |
| No ring match | 360 (3.1%) | SMARTS pattern found no six-membered quinone-carbonyl ring — fused lactones or fully aromatised systems; retained without a planarity verdict |

**Final dataset: 11,430 molecules, 3,189 Murcko scaffolds**, in
`data/descriptors_final.csv`. Excluded molecules are preserved in
`data/descriptors_excluded*.csv` rather than deleted.

---

## 3. Phase B — surrogate model

### 3.1 Design: predict IP/EA, derive the rest

The model predicts **IP and EA directly**, and μ, η, ω are computed from those
predictions by the same formulas used to build the dataset — rather than
training separately on μ, η and ω.

This follows directly from how error propagates through the finite-difference
formulas. μ is an average of IP and EA, so errors of opposite sign partially
cancel; η and ω involve a *difference*, so errors add. Training a model
directly on η or ω would teach it to reproduce that amplified noise instead of
letting it emerge, correctly, from two well-behaved predictions.

### 3.2 Baseline: Random Forest on Morgan fingerprints

A Random Forest (200 trees) on radius-2 Morgan fingerprints (2048 bits) was
trained for IP and for EA, each validated two ways:

- **Random split** — the number most QSAR papers report, and the one that
  overstates performance, because near-identical molecules from the same
  scaffold routinely land on both sides of the split.
- **Scaffold split** (`GroupShuffleSplit` on the Murcko scaffold) — every
  molecule sharing a scaffold is kept entirely in train or entirely in test.
  This is the honest estimate of generalisation to unseen chemistry.

**Results:**

| Target | Random split R² | Scaffold split R² | Scaffold MAE (eV) |
|---|---|---|---|
| IP | 0.823 | **0.764** | 0.170 |
| EA | 0.812 | **0.693** | 0.131 |

The scaffold split costs 0.06 R² for IP and 0.12 for EA — a real but moderate
penalty, and the number that should be quoted whenever this model's real-world
performance is discussed. EA generalises worse than IP, consistent with EA
being more sensitive to substituent electronics (donor/acceptor character) that
vary more between structural families than the π-system energetics dominating
IP.

**Derived descriptors, from the scaffold-split test set:**

| Descriptor | R² | MAE (eV) |
|---|---|---|
| μ | 0.770 | 0.125 |
| η | 0.631 | 0.091 |
| ω | 0.554 | 0.893 |

The same error cascade observed when comparing xTB to DFT for lapachol
reappears here, one level removed: μ (an average) survives the model's own
prediction error best, η (a difference) worse, and ω (which divides by η)
worst. This is not a peculiarity of one method — it is structural to the
finite-difference formulas themselves, and it shows up identically whether the
approximation being layered is xTB-on-DFT or a model-on-xTB.

### 3.3 What this baseline means for the next step

An R² of 0.76 (IP) and 0.69 (EA) under honest, scaffold-held-out validation is
a legitimate, reportable result on its own — with a 2048-bit fixed fingerprint
and no learned representation at all. A graph neural network is the natural
next comparison, but it is not guaranteed to improve on this: if it does not,
that is itself informative, and this fingerprint baseline is what should be
reported as the primary model.

---

## 4. Reproducing this work

```bash
git clone https://github.com/msouza-compchem/quinone-ml.git
cd quinone-ml
conda env create -f environment.yml
conda activate lapachol
```

```bash
python 01_collect_dataset.py                    # Phase A: ~15 min, needs network
python 02_xtb_descriptors.py                    # Phase A: ~30 min, 3 workers
python 03_check_planarity.py                    # Phase A QC: ~15 min
python 04_train_baseline.py                     # Phase B: ~5 min
```

Every long-running script is resumable: results are appended as they finish,
and a rerun skips molecules already present in the output file.

| Component | Version / setting |
|---|---|
| xTB | GFN2, `--alpb water --vipea` |
| RDKit | ETKDGv3 embedding, seed 42, MMFF94 relaxation |
| scikit-learn | RandomForestRegressor, 200 trees |
| Fingerprint | Morgan, radius 2, 2048 bits |

---

## 5. Limitations

- **A single DFT reference point.** The xTB-vs-DFT comparison rests entirely on
  lapachol. A stratified subset in DFT (`--subset` in `02_xtb_descriptors.py`
  supports this) would let the vertical-EA agreement be checked across the
  chemical space rather than at one point — this is the most useful next
  addition to the reference itself.
- **Vertical, not adiabatic, IP/EA.** No geometry relaxation follows electron
  removal/addition. The reorganisation energy this omits happened to
  approximately cancel the method-level error for lapachol's EA (3.770 eV
  adiabatic DFT vs. 3.677 eV vertical xTB); this is not guaranteed to hold
  molecule to molecule.
- **A fixed, unlearned representation.** Morgan fingerprints encode local
  atomic environments chosen by a hashing scheme fixed in advance, blind to
  this specific task. A graph neural network could, in principle, learn a
  representation better suited to predicting IP/EA — Phase B's next step.
- **No applicability domain yet.** The model will produce a confident-looking
  prediction for a query molecule far outside the training distribution,
  with no warning. This is planned via extended similarity indices over the
  scaffold space.
- **Geometry from a single conformer**, not a Boltzmann-weighted ensemble —
  unlike the lapachol study, which used CREST sampling. At this dataset scale,
  full conformational search was not computationally tractable; this is the
  main geometric approximation beyond the electronic-structure method itself.

---

## 6. References

Method and software choices follow lapachol-nqo1; see that repository's
`docs/REFERENCES.md` for full citations of xTB, RDKit, and scikit-learn.
Additional:

- Bannwarth, C.; Ehlert, S.; Grimme, S. GFN2-xTB — An Accurate and Broadly
  Parametrized Self-Consistent Tight-Binding Quantum Chemical Method. *J. Chem.
  Theory Comput.* **2019**, *15*, 1652–1671.
- Bemis, G. W.; Murcko, M. A. The Properties of Known Drugs. 1. Molecular
  Frameworks. *J. Med. Chem.* **1996**, *39*, 2887–2893. — Murcko scaffold
  definition used for the train/test split.
- Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf.
  Model.* **2010**, *50*, 742–754. — Morgan/ECFP fingerprint algorithm.

## 7. Licence

MIT — see [`LICENSE`](LICENSE).

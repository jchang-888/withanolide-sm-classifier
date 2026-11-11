"""
Step 2: two label sets (Option C) + sequence features.

Label sets
  A = Moore S1 labels as published (410 SM, 2,212 GM)
  B = A + 11 core sterol-pathway genes that Moore left "unknown", added as GM
      (SMT1, SMT2, SMO1-1, SMO2-1, CYP51G1, HYD1, DWF7, DWF5, DWF1, CYP710A1, CAS1)
  Why both: in A the model never sees a core sterol enzyme labeled GM.
  Caveat for B: the ashwagandha negatives are orthologs of these exact genes,
  so B makes the negatives easier on purpose. Compare A vs B, don't treat B alone as the result.

Features (computed for every training and test protein)
  aac      20   amino acid composition (fraction of each amino acid)
  dpc      400  dipeptide composition (fraction of each pair, e.g. "AL")
  physchem 6    length (log), mean hydrophobicity, charge, aromatic / tiny / proline fractions
  esm2     320  protein language model embedding, made by scripts/03b_esm_embeddings.py (run on your Mac)

Inputs:  data/processed/train_labels.csv, data/processed/test_labels.csv,
         data/processed/test_proteins.fa, TAIR10_pep_20110103_representative_gene_model
Outputs: data/processed/labels_A.csv, labels_B.csv
         data/processed/all_proteins.fa          every train + test protein in one file
         data/processed/features_aac.npz, features_dpc.npz, features_physchem.npz
         data/processed/step2_report.txt

Run from the repo root:  python scripts/03_build_features.py
"""
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
TAIR = ROOT / "TAIR10_pep_20110103_representative_gene_model"
AA = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
    seqs, name, chunks = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(chunks)
            name, chunks = line[1:].split()[0], []
        elif line:
            chunks.append(line)
    if name:
        seqs[name] = "".join(chunks)
    return {k: v.rstrip("*").upper() for k, v in seqs.items()}


# ------------------------------------------------------------------ label sets
train = pd.read_csv(PROC / "train_labels.csv")
tair = {k.split(".")[0]: v for k, v in read_fasta(TAIR).items()}

sterol_gm = {
    "AT5G13710": "SMT1", "AT1G20330": "SMT2", "AT4G12110": "SMO1-1", "AT1G07420": "SMO2-1",
    "AT1G11680": "CYP51G1", "AT1G20050": "HYD1", "AT3G02580": "DWF7", "AT1G50430": "DWF5",
    "AT3G19820": "DWF1", "AT2G34500": "CYP710A1", "AT2G07050": "CAS1",
}
already = set(train.gene) & set(sterol_gm)
assert not already, f"these were already labeled in Moore: {already}"

A = train[["gene", "label", "family"]].copy()
A["source"] = "Moore S1"
add = pd.DataFrame({
    "gene": list(sterol_gm),
    "label": "GM",
    "family": ["p450" if n.startswith("CYP") else "none" for n in sterol_gm.values()],
    "source": [f"added: {n} (core sterol pathway)" for n in sterol_gm.values()],
})
B = pd.concat([A, add], ignore_index=True)
A.to_csv(PROC / "labels_A.csv", index=False)
B.to_csv(PROC / "labels_B.csv", index=False)

# ------------------------------------------------------------------ all proteins
test = pd.read_csv(PROC / "test_labels.csv")
test_seqs = read_fasta(PROC / "test_proteins.fa")
seqs = {g: tair[g] for g in B.gene}
seqs.update({p: test_seqs[p] for p in test.protein})
ids = list(seqs)
with open(PROC / "all_proteins.fa", "w") as fh:
    for i in ids:
        fh.write(f">{i}\n{seqs[i]}\n")

# ------------------------------------------------------------------ features
def clean(s):
    return "".join(c for c in s if c in AA)


def aac(s):
    s = clean(s)
    return np.array([s.count(a) / len(s) for a in AA])


PAIRS = ["".join(p) for p in product(AA, AA)]
PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}


def dpc(s):
    s = clean(s)
    v = np.zeros(400)
    for i in range(len(s) - 1):
        v[PAIR_IDX[s[i:i + 2]]] += 1
    return v / max(len(s) - 1, 1)


# Kyte-Doolittle hydrophobicity scale (Kyte & Doolittle 1982, J Mol Biol 157:105)
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
PHYS_NAMES = ["log_length", "mean_hydrophobicity", "net_charge_per_res",
              "aromatic_frac", "tiny_frac", "proline_frac"]


def physchem(s):
    s = clean(s)
    n = len(s)
    return np.array([
        np.log10(n),
        np.mean([KD[c] for c in s]),
        (s.count("K") + s.count("R") - s.count("D") - s.count("E")) / n,
        sum(s.count(c) for c in "FWY") / n,
        sum(s.count(c) for c in "AGS") / n,
        s.count("P") / n,
    ])


for name, fn, cols in [("aac", aac, list(AA)), ("dpc", dpc, PAIRS), ("physchem", physchem, PHYS_NAMES)]:
    X = np.vstack([fn(seqs[i]) for i in ids])
    np.savez_compressed(PROC / f"features_{name}.npz", ids=np.array(ids), X=X, columns=np.array(cols))

nonstd = {i: sum(c not in AA for c in seqs[i]) for i in ids}
lines = [
    f"Label set A: {A.label.value_counts().to_dict()}",
    f"Label set B: {B.label.value_counts().to_dict()}  (added {len(add)} sterol genes as GM)",
    f"Proteins with features: {len(ids)}  (train B {len(B)} + test {len(test)})",
    f"Proteins with non-standard letters (X, U, etc., dropped before counting): "
    f"{sum(v > 0 for v in nonstd.values())}",
    "Feature sets: aac 20, dpc 400, physchem 6",
]
(PROC / "step2_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

"""
Step 1a: build the Arabidopsis training set (SM vs GM enzymes).

Inputs (all in this repo):
  data/moore2019/pnas.1817074116.sd01.xlsx         Moore et al. 2019 PNAS, Dataset S1
  TAIR10_pep_20110103_representative_gene_model    TAIR10 proteins, one per gene

Outputs:
  data/processed/train_labels.csv     gene, label (SM/GM), strict label, description, family, Moore SM score
  data/processed/train_proteins.fa    protein sequences for the labeled genes
  data/processed/step1a_report.txt    counts and any problems found

Run from the repo root:  python scripts/01_build_training_set.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S1 = ROOT / "data/moore2019/pnas.1817074116.sd01.xlsx"
TAIR = ROOT / "TAIR10_pep_20110103_representative_gene_model"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)


def read_fasta(path):
    """Return {id: sequence}. id = first word of the header line."""
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
    return seqs


# 1. labels from Moore S1, Sheet 1 (row 2 holds the real column names)
s1 = pd.read_excel(S1, sheet_name="Sheet 1", header=1)
s1 = s1.rename(columns={
    "TAIR gene name": "gene",
    "TAIR short desc.": "description",
    "Instances defined as": "label",                               # union of GO + AraCyc
    "Instances (with evidence code) defined as": "label_strict",   # experimental evidence only
    "Targeted families": "family",
    "Mean SM score": "moore_sm_score",
})
s1 = s1.dropna(subset=["gene"])
s1["gene"] = s1["gene"].str.upper().str.strip()
labeled = s1[s1["label"].isin(["SM", "GM"])].copy()

# 2. sequences: TAIR ids look like AT1G01010.1, strip the ".1" to get the gene name
tair = read_fasta(TAIR)
tair_by_gene = {k.split(".")[0].upper(): v.rstrip("*") for k, v in tair.items()}

labeled["has_seq"] = labeled["gene"].isin(tair_by_gene)
missing = labeled.loc[~labeled["has_seq"], "gene"].tolist()
labeled = labeled[labeled["has_seq"]].copy()
labeled["length"] = labeled["gene"].map(lambda g: len(tair_by_gene[g]))

cols = ["gene", "label", "label_strict", "description", "family", "moore_sm_score", "length"]
labeled[cols].to_csv(OUT / "train_labels.csv", index=False)
with open(OUT / "train_proteins.fa", "w") as fh:
    for g in labeled["gene"]:
        fh.write(f">{g}\n{tair_by_gene[g]}\n")

# 3. report
lines = [
    f"Moore S1 rows: {len(s1)}",
    f"TAIR10 proteins: {len(tair)}",
    "Union labels in S1: " + s1["label"].value_counts(dropna=False).to_dict().__str__(),
    f"Labeled genes with no TAIR10 sequence: {len(missing)} {missing}",
    "Final training set: " + labeled["label"].value_counts().to_dict().__str__(),
    "Strict labels within final set: " + labeled["label_strict"].value_counts(dropna=False).to_dict().__str__(),
    "P450s (family == p450) by label: " + labeled[labeled["family"] == "p450"]["label"].value_counts().to_dict().__str__(),
    f"Protein length median SM / GM: {labeled[labeled.label=='SM'].length.median()} / {labeled[labeled.label=='GM'].length.median()}",
]
(OUT / "step1a_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

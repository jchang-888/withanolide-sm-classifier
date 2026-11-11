"""
Step 3: homology-aware cross-validation folds.

Problem: if a protein's close relative is in training, the model can "recognize" it
instead of learning anything about function. Random splits hide this.
Fix: group related proteins into clusters, and keep each cluster entirely in one fold.

How:
  1. all-vs-all search of training proteins with phmmer (pyhmmer)
  2. identity = identical positions / length of the shorter protein
  3. two clusterings (single linkage = linked if ANY pair passes):
       id40    : linked if identity >= 40%   (close relatives)
       family  : linked if E-value < 1e-10   (any clear homology, much stricter split)
  4. 5 folds with StratifiedGroupKFold (keeps SM:GM ratio similar per fold)
Also saves test -> training hits for the nearest-neighbor baseline and explanations.

Inputs:  data/processed/labels_B.csv, data/processed/all_proteins.fa, data/processed/test_labels.csv
Outputs: data/processed/train_vs_train_hits.csv.gz   all hits with E < 1e-3
         data/processed/test_vs_train_hits.csv       all hits with E < 1e-3
         data/processed/folds.csv                    gene, clusters, fold for each split
         data/processed/step3_report.txt

Run from the repo root:  python scripts/04_homology_split.py   (a few minutes)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyhmmer
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
AA = set("ACDEFGHIKLMNPQRSTVWY")


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
    return seqs


labels = pd.read_csv(PROC / "labels_B.csv")          # B is a superset of A
test = pd.read_csv(PROC / "test_labels.csv")
seqs = read_fasta(PROC / "all_proteins.fa")
alphabet = pyhmmer.easel.Alphabet.amino()


def digitize(ids):
    out = []
    for i in ids:
        s = "".join(c if c in AA else "X" for c in seqs[i])
        out.append(pyhmmer.easel.TextSequence(name=i.encode(), sequence=s).digitize(alphabet))
    return out


def search(query_ids, target_ids, max_e=1e-3):
    db = pyhmmer.easel.DigitalSequenceBlock(alphabet, digitize(target_ids))
    rows = []
    for hits in pyhmmer.hmmer.phmmer(digitize(query_ids), db, cpus=0, E=max_e):
        q = hits.query.name
        q = q.decode() if isinstance(q, bytes) else q
        for h in hits:
            t = h.name.decode() if isinstance(h.name, bytes) else h.name
            if t == q:
                continue
            ident_line = h.best_domain.alignment.identity_sequence
            n_ident = sum(c.isalpha() for c in ident_line)
            shorter = min(len(seqs[q]), len(seqs[t]))
            rows.append((q, t, h.evalue, round(h.score, 1), round(n_ident / shorter, 3)))
    return pd.DataFrame(rows, columns=["query", "target", "evalue", "bitscore", "identity"])


train_ids = labels.gene.tolist()
tt = search(train_ids, train_ids)
tt.to_csv(PROC / "train_vs_train_hits.csv.gz", index=False)
xt = search(test.protein.tolist(), train_ids)
xt.to_csv(PROC / "test_vs_train_hits.csv", index=False)


def components(ids, edges):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = {r: k for k, r in enumerate(sorted({find(i) for i in ids}))}
    return {i: roots[find(i)] for i in ids}


folds = labels[["gene", "label"]].copy()
report = [f"Training proteins (set B): {len(train_ids)}",
          f"train-vs-train hits (E<1e-3): {len(tt)}   test-vs-train hits: {len(xt)}"]
for name, mask in [("id40", tt.identity >= 0.40), ("family", tt.evalue < 1e-10)]:
    comp = components(train_ids, tt.loc[mask, ["query", "target"]].itertuples(index=False))
    folds[f"cluster_{name}"] = folds.gene.map(comp)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    fold = np.zeros(len(folds), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(folds, folds.label, folds[f"cluster_{name}"])):
        fold[te] = k
    folds[f"fold_{name}"] = fold
    sizes = folds[f"cluster_{name}"].value_counts()
    per_fold = folds.groupby(f"fold_{name}").label.value_counts().unstack().fillna(0).astype(int)
    report += [
        f"\n[{name}] clusters: {len(sizes)}   singletons: {(sizes == 1).sum()}   "
        f"largest: {sizes.iloc[0]}   top 5 sizes: {sizes.head().tolist()}",
        f"[{name}] per fold (GM / SM):\n{per_fold.to_string()}",
    ]

folds.to_csv(PROC / "folds.csv", index=False)
# how close is each test protein to anything in training?
best = xt.sort_values("bitscore", ascending=False).drop_duplicates("query").set_index("query")
t = test.set_index("protein")
t["best_train_hit"] = best.target
t["best_identity"] = best.identity
report.append("\nTest proteins: best identity to any training protein (median by label): "
              + str(t.groupby("label").best_identity.median().round(3).to_dict()))
report.append(f"Test proteins with no hit at E<1e-3: {t.best_train_hit.isna().sum()}")
(PROC / "step3_report.txt").write_text("\n".join(report) + "\n")
print("\n".join(report))

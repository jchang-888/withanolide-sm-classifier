"""
Steps 4 + 5: baselines, logistic regression and a neural network, cross-validated.

For every label set (A, B) x split (random, id40, family) x 5 folds:
  Baselines
    majority     always "GM" (score is constant, so AuROC = 0.5)
    family_prior fraction SM among training genes in the same Moore "Targeted families" group
    nearest_nbr  bitscore-weighted SM fraction of the 5 best phmmer hits in the training folds
  Models (on aac + dpc + physchem, and on esm2 if features_esm2.npz exists)
    logreg       logistic regression, balanced class weights, C = 0.1 (not tuned)
    mlp          PyTorch neural network: inputs -> 128 -> 32 -> 1, ReLU, dropout 0.3,
                 weighted loss for the 5.4:1 imbalance, Adam, 60 epochs (not tuned)

Metrics (all computed on held-out folds only)
  auroc     0.5 = coin flip, 1.0 = perfect ranking
  auprc     average precision. Coin flip = fraction SM (~0.156)
  auroc_p450  AuROC among P450s only = can it tell SM P450s from GM P450s?

Inputs:  data/processed/labels_A.csv, labels_B.csv, folds.csv, features_*.npz,
         train_vs_train_hits.csv.gz
Outputs: data/processed/cv_metrics.csv          one row per label set x split x method
         data/processed/cv_predictions.csv.gz   every held-out score
         data/processed/step45_report.txt

Run from the repo root:  python scripts/05_train_evaluate.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
SEED = 0


def load_features(names):
    blocks = []
    for n in names:
        z = np.load(PROC / f"features_{n}.npz", allow_pickle=False)
        blocks.append(pd.DataFrame(z["X"], index=z["ids"]))
    return pd.concat(blocks, axis=1)


FEATURE_SETS = {"seqcomp": ["aac", "dpc", "physchem"]}
if (PROC / "features_esm2.npz").exists():
    FEATURE_SETS["esm2"] = ["esm2"]
FEATS = {k: load_features(v) for k, v in FEATURE_SETS.items()}

folds_all = pd.read_csv(PROC / "folds.csv").set_index("gene")
hits = pd.read_csv(PROC / "train_vs_train_hits.csv.gz")


# ------------------------------------------------------------------ methods
def family_prior(tr, te):
    p = tr.groupby("family").y.mean()
    return te.family.map(p).fillna(tr.y.mean()).values


def nearest_nbr(tr, te, k=5):
    trset = set(tr.index)
    h = hits[hits["query"].isin(te.index) & hits.target.isin(trset)].copy()
    h["y"] = h.target.map(tr.y)
    h = h.sort_values("bitscore", ascending=False).groupby("query").head(k)
    score = h.groupby("query").apply(lambda d: np.average(d.y, weights=d.bitscore), include_groups=False)
    return te.index.map(score).fillna(tr.y.mean()).values


def logreg(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp(Xtr, ytr, Xte, epochs=60):
    torch.manual_seed(SEED)
    sc = StandardScaler().fit(Xtr)
    xtr = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    xte = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    model = MLP(xtr.shape[1])
    pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(xtr), generator=g).split(64):
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr_t[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xte)).numpy()


# ------------------------------------------------------------------ run
preds, metrics = [], []
for lset in ["A", "B"]:
    lab = pd.read_csv(PROC / f"labels_{lset}.csv").set_index("gene")
    lab["y"] = (lab.label == "SM").astype(int)
    lab["family"] = lab.family.fillna("none")
    rnd = StratifiedKFold(5, shuffle=True, random_state=SEED)
    split_cols = {"random": pd.Series(0, index=lab.index)}
    for k, (_, te) in enumerate(rnd.split(lab, lab.y)):
        split_cols["random"].iloc[te] = k
    for s in ["id40", "family"]:
        split_cols[s] = folds_all.loc[lab.index, f"fold_{s}"]

    for split, fold in split_cols.items():
        for k in range(5):
            tr, te = lab[fold != k], lab[fold == k]
            scores = {"majority": np.zeros(len(te)),
                      "family_prior": family_prior(tr, te),
                      "nearest_nbr": nearest_nbr(tr, te)}
            for fs, X in FEATS.items():
                scores[f"logreg_{fs}"] = logreg(X.loc[tr.index].values, tr.y.values, X.loc[te.index].values)
                scores[f"mlp_{fs}"] = mlp(X.loc[tr.index].values, tr.y.values, X.loc[te.index].values)
            for method, sc in scores.items():
                preds.append(pd.DataFrame({"label_set": lset, "split": split, "fold": k, "method": method,
                                           "gene": te.index, "y": te.y.values,
                                           "family": te.family.values, "score": sc}))
        print(f"done: label set {lset}, split {split}")

P = pd.concat(preds, ignore_index=True)
P.to_csv(PROC / "cv_predictions.csv.gz", index=False)


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


for (lset, split, method), d in P.groupby(["label_set", "split", "method"]):
    p450 = d[d.family == "p450"]
    metrics.append({"label_set": lset, "split": split, "method": method,
                    "auroc": round(safe_auc(d.y, d.score), 3),
                    "auprc": round(average_precision_score(d.y, d.score), 3),
                    "auroc_p450": round(safe_auc(p450.y, p450.score), 3),
                    "n": len(d), "n_SM": int(d.y.sum())})
M = pd.DataFrame(metrics)
M.to_csv(PROC / "cv_metrics.csv", index=False)

order = ["majority", "family_prior", "nearest_nbr"] + [f"{m}_{fs}" for fs in FEATS for m in ("logreg", "mlp")]
lines = ["Pooled out-of-fold metrics (AuROC / AuPRC / AuROC within P450s)", ""]
for lset in ["A", "B"]:
    t = M[M.label_set == lset].copy()
    t["cell"] = t.auroc.astype(str) + " / " + t.auprc.astype(str) + " / " + t.auroc_p450.astype(str)
    tab = t.pivot(index="method", columns="split", values="cell").reindex(order)[["random", "id40", "family"]]
    lines += [f"Label set {lset}", tab.to_string(), ""]
(PROC / "step45_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

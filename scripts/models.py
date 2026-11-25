"""
Shared model code used by 05_train_evaluate.py and 06_score_test_set.py.

  load_feature_sets()  -> {"seqcomp": DataFrame, "esm2": DataFrame (if the file exists)}
  logreg(Xtr, ytr, Xte)            logistic regression, balanced class weights, C = 0.1
  mlp(Xtr, ytr, Xte, seed=0)       PyTorch neural network: inputs -> 128 -> 32 -> 1
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROC = Path(__file__).resolve().parent.parent / "data/processed"


def load_features(names):
    blocks = []
    for n in names:
        z = np.load(PROC / f"features_{n}.npz", allow_pickle=False)
        blocks.append(pd.DataFrame(z["X"], index=z["ids"]))
    return pd.concat(blocks, axis=1)


def load_feature_sets():
    sets = {"seqcomp": ["aac", "dpc", "physchem"]}
    if (PROC / "features_esm2.npz").exists():
        sets["esm2"] = ["esm2"]
    return {k: load_features(v) for k, v in sets.items()}


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


def mlp(Xtr, ytr, Xte, seed=0, epochs=60):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    xtr = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    xte = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    model = MLP(xtr.shape[1])
    # weighted loss: an SM mistake costs (#GM / #SM) times more, to offset the 5.4:1 imbalance
    pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(xtr), generator=g).split(64):
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr_t[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xte)).numpy()

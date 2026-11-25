"""
Step 6: the actual question. Train on ALL Arabidopsis data, score the 45 ashwagandha proteins.

Groups
  tested SM   cloned withanolide enzymes, enzyme reported active (11)
  cluster SM  other cluster enzymes, never cloned (18, includes SDH1)
  sterol GM   ashwagandha orthologs of core sterol / brassinosteroid genes (16)

Methods: nearest_nbr, logreg, mlp (same settings as Step 5, MLP averaged over 5 seeds)
         + "unfamiliarity" = 1 - identity to the closest training protein.
           If unfamiliarity alone separates SM from GM, the models may be scoring
           "looks new" instead of "looks specialized" (the confound from Step 3).

Each label set (A, B) is trained separately.
AuROC confidence intervals: 2,000 bootstrap resamples of the 45 proteins
(small n, so the intervals are wide and approximate).

Inputs:  data/processed/labels_A.csv, labels_B.csv, test_labels.csv, features_*.npz,
         test_vs_train_hits.csv, train_labels.csv (for descriptions)
Outputs: data/processed/test_scores.csv     every protein x label set x method, + closest Arabidopsis gene
         data/processed/test_metrics.csv    AuROC (95% CI) per label set x method x comparison
         figures/fig1_test_scores.png       scores by group
         figures/fig2_score_vs_identity.png confound check
         data/processed/step6_report.txt

Run from the repo root:  python scripts/06_score_test_set.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

from models import load_feature_sets, logreg, mlp  # same model code as Step 5
FEATS = load_feature_sets()

test = pd.read_csv(PROC / "test_labels.csv").set_index("protein")
test["group"] = np.where(test.label == "GM", "sterol GM",
                         np.where(test.tier.str.startswith("cloned, enzyme"), "tested SM", "cluster SM"))
test["y"] = (test.label == "SM").astype(int)
desc = pd.read_csv(PROC / "train_labels.csv").set_index("gene").description
xt = pd.read_csv(PROC / "test_vs_train_hits.csv")

rows = []
for lset in ["A", "B"]:
    lab = pd.read_csv(PROC / f"labels_{lset}.csv").set_index("gene")
    lab["y"] = (lab.label == "SM").astype(int)
    h = xt[xt.target.isin(lab.index)].copy()
    h["y"] = h.target.map(lab.y)
    h = h.sort_values("bitscore", ascending=False)
    top = h.drop_duplicates("query").set_index("query")
    top5 = h.groupby("query").head(5)
    nn_score = top5.groupby("query").apply(lambda d: np.average(d.y, weights=d.bitscore), include_groups=False)

    scores = {"nearest_nbr": test.index.map(nn_score).fillna(lab.y.mean()).values,
              "unfamiliarity": 1 - test.index.map(top.identity).fillna(0).values}
    for fs, X in FEATS.items():
        Xtr, Xte = X.loc[lab.index].values, X.loc[test.index].values
        scores[f"logreg_{fs}"] = logreg(Xtr, lab.y.values, Xte)
        runs = [mlp(Xtr, lab.y.values, Xte, seed=seed) for seed in range(5)]
        scores[f"mlp_{fs}"] = np.mean(runs, axis=0)
        scores[f"mlp_{fs}_seed_sd"] = np.std(runs, axis=0)

    for m, s in scores.items():
        d = test[["gene", "name", "group", "y"]].copy()
        d["label_set"], d["method"], d["score"] = lset, m, s
        d["closest_arabidopsis"] = test.index.map(top.target)
        d["closest_identity"] = test.index.map(top.identity)
        d["closest_label"] = d.closest_arabidopsis.map(lab.label)
        d["closest_description"] = d.closest_arabidopsis.map(desc).fillna(
            d.closest_arabidopsis.map(lab.source))
        rows.append(d.reset_index())

S = pd.concat(rows, ignore_index=True)
S.to_csv(PROC / "test_scores.csv", index=False)

# ------------------------------------------------------------------ metrics with bootstrap CI
rng = np.random.default_rng(0)


def auc_ci(y, s, n=2000):
    y, s = np.asarray(y), np.asarray(s)
    est = roc_auc_score(y, s)
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(est, 3), round(lo, 3), round(hi, 3)


metrics = []
for (lset, m), d in S[~S.method.str.endswith("_seed_sd")].groupby(["label_set", "method"]):
    for comp, sub in [("all SM vs GM", d), ("tested SM vs GM", d[d.group != "cluster SM"])]:
        est, lo, hi = auc_ci(sub.y, sub.score)
        metrics.append({"label_set": lset, "method": m, "comparison": comp, "auroc": est,
                        "ci_low": lo, "ci_high": hi, "n_SM": int(sub.y.sum()), "n_GM": int((1 - sub.y).sum()),
                        "frac_SM_over_0.5": round((sub[sub.y == 1].score > 0.5).mean(), 2),
                        "frac_GM_over_0.5": round((sub[sub.y == 0].score > 0.5).mean(), 2)})
M = pd.DataFrame(metrics)
M.to_csv(PROC / "test_metrics.csv", index=False)

# ------------------------------------------------------------------ figures
COL = {"tested SM": "#2a78d6", "sterol GM": "#eb6834", "cluster SM": "#1baf7a"}  # validated palette
ORDER = ["tested SM", "cluster SM", "sterol GM"]
methods = ["nearest_nbr", "logreg_seqcomp", "mlp_seqcomp"] + (["mlp_esm2"] if "esm2" in FEATS else [])
fig, axes = plt.subplots(2, len(methods), figsize=(3.2 * len(methods), 6.4), sharey=True)
for r, lset in enumerate(["A", "B"]):
    for c, m in enumerate(methods):
        ax = axes[r, c]
        d = S[(S.label_set == lset) & (S.method == m)]
        for k, g in enumerate(ORDER):
            v = d[d.group == g].score.values
            x = k + np.random.default_rng(k).uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=36, color=COL[g], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
            ax.plot([k - 0.28, k + 0.28], [np.median(v)] * 2, color="#0b0b0b", lw=2, zorder=4)
        ax.axhline(0.5, color="#c3c2b7", lw=1, ls="--", zorder=1)
        ax.set_xticks(range(3), [g.replace(" ", "\n") for g in ORDER], fontsize=9, color="#52514e")
        ax.set_ylim(-0.03, 1.03)
        auc = M[(M.label_set == lset) & (M.method == m) & (M.comparison == "all SM vs GM")].iloc[0]
        ax.set_title(f"{m}  (labels {lset})\nAuROC {auc.auroc} [{auc.ci_low}-{auc.ci_high}]", fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if c == 0:
            ax.set_ylabel("predicted SM score", color="#52514e")
fig.suptitle("Ashwagandha proteins scored by Arabidopsis-trained models (line = median)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig1_test_scores.png", dpi=200, facecolor="#fcfcfb")

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, lset in zip(axes, ["A", "B"]):
    d = S[(S.label_set == lset) & (S.method == "mlp_seqcomp")]
    for g in ORDER:
        dd = d[d.group == g]
        ax.scatter(dd.closest_identity * 100, dd.score, s=36, color=COL[g], label=g,
                   edgecolor="#fcfcfb", linewidth=1.5)
    ax.set_xlabel("identity to closest Arabidopsis training protein (%)", color="#52514e")
    ax.set_title(f"mlp_seqcomp, labels {lset}", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("predicted SM score", color="#52514e")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Confound check: do scores just track how unfamiliar a protein is?", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig2_score_vs_identity.png", dpi=200, facecolor="#fcfcfb")

# ------------------------------------------------------------------ report
lines = ["AuROC on ashwagandha test set [95% bootstrap CI]", ""]
for comp in ["all SM vs GM", "tested SM vs GM"]:
    t = M[M.comparison == comp].copy()
    t["cell"] = t.auroc.astype(str) + " [" + t.ci_low.astype(str) + "-" + t.ci_high.astype(str) + "]"
    lines += [f"{comp}", t.pivot(index="method", columns="label_set", values="cell").to_string(), ""]
mlp_ab = S[S.method == "mlp_seqcomp"].pivot_table(index=["protein", "name", "group"], columns="label_set", values="score")
mlp_ab["B_minus_A"] = mlp_ab.B - mlp_ab.A
lines += ["mlp_seqcomp score per protein (A vs B):", mlp_ab.round(2).sort_values("B_minus_A").to_string(), ""]
d = S[(S.label_set == "A") & (S.method == "mlp_seqcomp")]
lines.append("Spearman corr (mlp_seqcomp A score vs closest identity): "
             f"{d.score.corr(d.closest_identity, method='spearman'):.2f}")
(PROC / "step6_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

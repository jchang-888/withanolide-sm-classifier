# Step 6 - Results on Ashwagandha
## Setup

* train on ALL arabidopsis data (no folds), score the 45 ashwagandha proteins
* 3 groups
	1. tested SM = cloned + reported active (11)
	2. cluster SM = in the cluster, never cloned (18)
	3. sterol GM = orthologs of normal sterol / brassinosteroid genes (16)
* mlp averaged over 5 seeds (random starting weights) bc it's unstable
* AuROC error bars = bootstrap (resample the 45 proteins 2,000 times)
* "unfamiliarity" = 1 - identity to closest training protein
	* if this alone separates SM vs GM, the models might just be scoring "looks new"

## Results (AuROC, all SM vs GM)

| method | labels A | labels B |
|---|---|---|
| nearest_nbr | 0.85 [0.73-0.96] | 0.88 [0.77-0.96] |
| mlp | 0.80 [0.65-0.92] | 0.81 [0.68-0.93] |
| logreg | 0.71 [0.55-0.85] | 0.72 [0.56-0.86] |
| unfamiliarity | 0.49 [0.26-0.72] | 0.94 [0.86-1.0] |

## What it means

1. labels A = the real result
	* unfamiliarity = 0.49 = coin flip → NOT confounded
	* so the step 3 worry only applies to B
2. labels B = confounded on purpose
	* unfamiliarity alone gets 0.94 bc B contains the negatives' exact orthologs
	* (ofc, that's why we ran A too)
3. nearest neighbor ≥ mlp > logreg, but the error bars all overlap
	* 45 proteins = can't say which method is better

## Per enzyme (labels A)

| enzyme | mlp | nearest nbr | closest arabidopsis gene | its label |
|---|---|---|---|---|
| CYP88C7 (x3) | ~1.0 | 0.4 | CYP708A1 / ROT3 | SM / GM |
| CYP88C10 | 1.0 | 0.2 | CPD (CYP90A1) | GM |
| CYP87G1 (x3) | 0.16 - 0.61 | 1.0 | CYP87A2 | SM |
| CYP749B2 | 0.45 | 0.0 | BAS1 (CYP734A1) | GM |
| 24ISO | 0.22 | 0.5 | weak hits only | |
| SDH2 | 0.0 | 0.19 | IBR1 (short-chain dehydrogenase) | GM |
| SULF1 | 0.0 | 1.0 | SOT16 (glucosinolate sulfotransferase) | SM |

* missed = called GM-ish by both methods: 24ISO, CYP749B2, SDH2
	* 24ISO = first committed step
	* with labels B (DWF1 = GM) 24ISO drops from 0.22 → 0.00
	* = 24ISO looks like general metabolism, fits the idea it came from a DWF1-like gene ?
* CYP749B2's closest arabidopsis gene = BAS1, a brassinosteroid-inactivating P450 (GM)
* the two methods disagree on SULF1 and CYP87G1 (nearest nbr says SM, mlp says GM-ish)
* false alarms: DWF4, ROT3, BR6OX1, CAS1 orthologs score 0.40 - 0.68 in mlp
	* model thinks some GM P450s look specialized

## Big takeaway

* "Arabidopsis-trained SM/GM models catch some withanolide enzymes (the CYP88s) but miss the pathway's sterol-like steps (24ISO, CYP749B2, SDH2)"
* = a genome-wide SM predictor would probably have missed the first committed step ?
* careful!! n = 11 tested enzymes, not a statistical claim

## Gaps

1. tiny test set, CIs are wide
2. mlp is unstable: same protein can swing up to ~0.5 between seeds
3. HYD1 + DWF5 orthologs had no hit in labels A (their arabidopsis versions aren't in training)
4. composition features only, esm2 might change the picture
5. "tested" = enzyme reported active, not checked per gene copy

# Links

* Reynolds et al. 2026 Nature Plants: https://www.nature.com/articles/s41477-026-02220-z
* Moore et al. 2019 PNAS: https://doi.org/10.1073/pnas.1817074116
* bootstrap CIs (Efron & Tibshirani 1993, "An Introduction to the Bootstrap"): https://doi.org/10.1201/9780429246593

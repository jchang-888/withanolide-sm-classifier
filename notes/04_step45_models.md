# Steps 4 + 5 - Baselines + Models
## Basics

* baseline = dumb method the real model has to beat
	* if the fancy model can't beat it, the fancy model adds nothing
* AuROC = if you pick 1 random SM + 1 random GM, how often does SM score higher?
	* 0.5 = coin flip, 1.0 = perfect
* AuPRC = average precision, better for imbalanced data
	* coin flip = fraction SM = 0.156
* auroc_p450 = only P450s → can it tell SM P450s from GM P450s? (the hard part)

## Methods

1. majority = always guess GM (84% "accurate", AuROC 0.5, useless)
2. family_prior = % SM in the same Moore family group (p450, terpene synthase, ...)
3. nearest_nbr = look up the 5 most similar training proteins, average their labels
4. logreg = logistic regression on aac + dpc + physchem (426 numbers)
5. mlp = my neural network (PyTorch)
	* 426 inputs → 128 → 32 → 1 output
	* ReLU, dropout 0.3 (randomly turns off neurons so it can't memorize)
	* weighted loss = SM mistakes count ~5.4x more (bc imbalance)
	* 60 epochs, not tuned

## Results (label set A, AuROC)

| method | random split | id40 | family |
|---|---|---|---|
| family_prior | 0.61 | 0.55 | 0.55 |
| nearest_nbr | 0.94 | 0.82 | 0.58 |
| logreg | 0.82 | 0.66 | 0.60 |
| mlp | 0.87 | 0.70 | 0.65 |

* label set B is basically the same (only 11 genes added)

## What it means

1. leakage is HUGE
	* nearest neighbor: 0.94 (random) → 0.58 (family)
	* random split = mostly just recognizing relatives
	* Moore reported AuROC 0.847 (but with way more features, and not a family split)
2. nearest neighbor beats everything at id40
	* if a close relative exists, just copy its label
3. at the family split everything is weak (0.55 - 0.65)
	* mlp is the best AuROC (0.65) but its AuPRC (0.24) < family prior (0.31) ?
	* = sequence composition alone barely knows "SM vs GM" for a brand-new family
4. P450s: family split, mlp 0.59 (A) vs 0.45 (B) → small changes flip it, so prob noise

## Gaps

1. no error bars yet, so differences under ~0.05 might just be luck
2. nothing tuned (C, layers, epochs)
3. family_prior only uses Moore's 4 family groups, most genes = "none"
4. esm2 not in yet (needs my mac run)

# Links

* Moore et al. 2019 PNAS: https://doi.org/10.1073/pnas.1817074116
* scikit-learn LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
* PyTorch: https://pytorch.org
* Saito & Rehmsmeier 2015 (why PR curves for imbalanced data): https://doi.org/10.1371/journal.pone.0118432

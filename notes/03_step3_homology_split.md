# Step 3 - Homology Split
## Basics

* problem = data leakage
	* if a protein's close cousin is in training, the model just "recognizes" it
	* random split → scores look way better than they really are
* fix = group related proteins, keep each group in ONE fold
* fold = 1 of 5 chunks, train on 4, test on the 5th, rotate (5-fold cross validation)

## How

* all-vs-all search = every training protein vs every other (phmmer)
	* 2,633 x 2,633, took ~6.5 min
	* 54,558 hits (E < 0.001)
* identity = identical positions / length of shorter protein
* 2 ways to group (single linkage = linked if ANY pair passes)
	1. id40 = linked if ≥ 40% identical (close relatives only)
	2. family = linked if E < 1e-10 (any clear homology, much stricter)
* StratifiedGroupKFold = keeps the SM:GM ratio about the same in every fold

## Results

| split | clusters | singletons | largest cluster |
|---|---|---|---|
| id40 | 1,320 | 855 | 44 |
| family | 764 | 386 | 144 |

* family split: the largest cluster = basically ALL the P450s (140 P450 + 4 others, 90 SM / 54 GM)
	* so all P450s end up in the same fold
	* model can't learn "P450 → SM" from other P450s when testing P450s
	* = the hardest, most honest test

## Test set distance (important!!)

* best identity to any training protein (median)
	* GM negatives: 76%
	* SM positives: 35%
* so the negatives have close arabidopsis relatives, the withanolide enzymes don't
* (ofc, the negatives were picked as orthologs of arabidopsis genes, so this is by design)
* means "far from everything" could get confused with "SM" ?
	* need to check this in step 6

## Gaps

1. single linkage can chain unrelated proteins together (A~B, B~C → A, C grouped)
2. 40% cutoff is a convention, not a law
3. distance was measured against set B, which includes the negatives' exact orthologs

# Links

* pyhmmer: https://pyhmmer.readthedocs.io
* HMMER (phmmer): http://hmmer.org
* scikit-learn StratifiedGroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

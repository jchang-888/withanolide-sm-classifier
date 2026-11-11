# Step 1 - Getting the Data
## Basics

* goal = train a model on arabidopsis enzymes, test it on ashwagandha withanolide enzymes
* question: does it call withanolide enzymes "SM" or does it get fooled bc they look like normal sterol enzymes?
* SM = specialized metabolism (lineage specific, defense, medicines)
* GM = general metabolism (every plant needs it, aka primary metabolism)
* Moore paper: "genes encoding GM or SM enzymes can belong to the same family"
	* that's the whole reason this is hard

## Files

1. TAIR10 proteins = arabidopsis sequences, one per gene
	* "representative gene model" = one splice form per gene (the other file has all of them = duplicates)
	* 27,416 proteins
2. Moore S1 = the SM/GM labels
3. ashwagandha proteins = from weng lab genome on zenodo
	* 44,100 proteins, 36,057 genes (some genes have >1 transcript, t1 / t2)
4. weng supplementary = primers + plasmids
	* turned out super useful (see below)

# Training Set (Step 1a)
## How

* labels from "Instances defined as" column = union of GO + AraCyc annotations
* dropped "unknown" genes (2,304 of them)
* matched gene names to TAIR (AT1G01010.1 → AT1G01010)
	* all labeled genes matched, 0 missing

## Results

| | count |
|---|---|
| SM | 410 |
| GM | 2,212 |
| total | 2,622 |

* imbalanced ~5.4 : 1
	* so plain accuracy is useless (calling everything GM = 84% "accurate")
	* use balanced accuracy / precision-recall instead
* P450s: 87 SM, 51 GM
* SM and GM proteins are basically the same length (median 448 vs 439) → length alone won't cheat

## Gaps (important!!)

1. most core sterol genes are NOT in the training set
	* SMT1, CYP51, DWF1, DWF5, DWF7, CYP710A, CAS1 = all "unknown" in Moore
	* only the brassinosteroid P450s (CYP90s, CYP85s) are labeled GM
	* so the model never learns "sterol enzyme = GM" ?
2. DWF1 is missing and 24ISO probably evolved from a DWF1-like gene ? (need to check paper)
	* means the model has no labeled relative for 24ISO
3. strict labels (experimental evidence only) = only 196 SM
	* use as a robustness check later

# Test Set (Step 1b)
## Positives (SM)

* gene IDs from the authors' own code on github (20231016_desq2.R)
* 36 cluster genes total, 2 clusters
	* cluster 1 = g3303 to g3320
	* cluster 2 = g18181 to g18197
* problem: which copy did they actually test?
	* ex. 3 copies of CYP87G1, 3 of CYP88C7
* fix: translate the cloning primers → match to start / end of each protein
	* fwd primer = start of protein (ATG...)
	* rev primer = end of protein (reverse complement → stop codon)

| enzyme | tested copy | notes |
|---|---|---|
| 24ISO | g3313 | script calls it 24ISO_2 |
| CYP87G1 | g3303, g3312b, g3312 | all 3 copies cloned |
| CYP749B2 | g18183 | in cluster 2! not g3306 like the script name suggests |
| CYP88C7 | g3310, g3315, g18191 | g3310 + g3315 are 99.8% identical |
| CYP88C10 | g3311 | not FUN_112423 |
| SDH2 | g3314 | |
| SULF1 | g3320 | |
| SDH1 | g3307 | cloned, not sure it worked |

* tiers
	1. cloned + enzyme reported active = 11 (best positives)
	2. cloned, unclear = 1
	3. in cluster, never cloned = 17 (weaker, ex. OGDs + acyltransferases might not be in the pathway)
	4. excluded: 5 pseudogenes, 2 non-enzymes

## Negatives (GM)

* ashwagandha's own copies of normal sterol / brassinosteroid genes
* why ashwagandha and not tomato?
	* same species as positives → species can't explain the difference
* how = reciprocal best hit (RBH)
	* search arabidopsis gene → best ashwagandha hit (skip cluster genes)
	* search that hit back → if best hit = the original gene, it's an ortholog
* tool = phmmer (pyhmmer)
* 17 queries → 16 RBH
	* BR6OX2 dropped, ashwagandha only has 1 CYP85 and it maps back to BR6OX1 (makes sense)

## Final test set

* 29 SM + 16 GM = 45 proteins
* small! → describe results, no p-values

## Gaps

1. "reported active" is at the enzyme level, not per copy (ex. not sure all 3 CYP88C7 copies worked)
2. negatives are only "GM" by assumption (they're orthologs of known GM genes)
3. tiers 2-3 positives are guesses based on location in the cluster

# Links

* Moore et al. 2019 PNAS: https://doi.org/10.1073/pnas.1817074116
* Reynolds et al. 2026 Nature Plants: https://www.nature.com/articles/s41477-026-02220-z
* ashwagandha genome (Zenodo): https://zenodo.org/records/17932505
* authors' code: https://github.com/erinreynolds/withania_somnifera
* TAIR10 mirror: https://solgenomics.net/ftp/genomes/Arabidopsis_thaliana/annotation/TAIR10

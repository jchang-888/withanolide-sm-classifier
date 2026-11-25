# Can an Arabidopsis-trained model spot withanolide enzymes?

Train a classifier on Arabidopsis enzymes labeled specialized (SM) or general (GM) metabolism (Moore et al. 2019), then test it on ashwagandha withanolide pathway enzymes (Reynolds et al. 2026) vs ashwagandha's normal sterol enzymes.

## Data (all files live in this repo)

| File | What | Source |
|---|---|---|
| `TAIR10_pep_20110103_representative_gene_model` | Arabidopsis proteins, one per gene | TAIR10 via [Sol Genomics mirror](https://solgenomics.net/ftp/genomes/Arabidopsis_thaliana/annotation/TAIR10) |
| `withania_somnifera_proteins.fa` | Ashwagandha proteins | [Zenodo 10.5281/zenodo.17932505](https://zenodo.org/records/17932505) |
| `weng_paper_supplementary_data.xlsx` | Primers and plasmids | [Reynolds et al. 2026, Nature Plants](https://www.nature.com/articles/s41477-026-02220-z) supplementary |
| `data/moore2019/pnas.1817074116.sd01.xlsx` | SM/GM labels (Dataset S1) | [Moore et al. 2019, PNAS](https://doi.org/10.1073/pnas.1817074116) |
| `data/moore2019/pnas.1817074116.sd03.xlsx` | Moore model performance (Dataset S3) | same |

Cluster gene IDs were copied from the authors' code: [erinreynolds/withania_somnifera](https://github.com/erinreynolds/withania_somnifera), `20231016_desq2.R`, commit `45b8999`.

## Run

```
pip install -r requirements.txt
python scripts/01_build_training_set.py   # Step 1a training set
python scripts/02_build_test_set.py       # Step 1b ashwagandha test set
python scripts/03_build_features.py       # Step 2 label sets A/B + features
python scripts/03b_esm_embeddings.py      # Step 2b ESM-2 embeddings (optional, downloads model)
python scripts/04_homology_split.py       # Step 3 homology-aware folds (~6 min)
python scripts/05_train_evaluate.py       # Steps 4-5 baselines + models, cross-validated
python scripts/06_score_test_set.py       # Step 6 score ashwagandha proteins + figures
```

## Results so far (composition features, label set A)

| | Cross-validation, family split (AuROC) | Ashwagandha test (AuROC, 95% CI) |
|---|---|---|
| Nearest neighbor | 0.58 | 0.85 [0.73-0.96] |
| Logistic regression | 0.60 | 0.71 [0.55-0.85] |
| Neural network (PyTorch MLP) | 0.65 | 0.80 [0.65-0.92] |

- Random splits inflate scores a lot (nearest neighbor 0.94 random vs 0.58 family split)
- CYP88 enzymes score as specialized. 24ISO, CYP749B2 and SDH2 score as general metabolism
- Test set is 45 proteins (11 tested enzymes), so results are descriptive
- See `notes/` for each step and `figures/` for plots

## Outputs (`data/processed/`)

- `train_labels.csv`, `train_proteins.fa`: 2,622 Arabidopsis enzymes (410 SM, 2,212 GM)
- `withanolide_cluster_genes.csv`: all 36 cluster genes with tier (cloned / not cloned / pseudogene)
- `primer_matches.csv`: cloning primer to protein matches (how tested copies were identified)
- `sterol_negatives.csv`: reciprocal best hits for 17 Arabidopsis sterol/brassinosteroid genes
- `test_labels.csv`, `test_proteins.fa`: 29 SM + 16 GM ashwagandha proteins
- `step1a_report.txt`, `step1b_report.txt`: counts and checks

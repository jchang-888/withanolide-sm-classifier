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
python scripts/01_build_training_set.py
python scripts/02_build_test_set.py
```

## Outputs (`data/processed/`)

- `train_labels.csv`, `train_proteins.fa`: 2,622 Arabidopsis enzymes (410 SM, 2,212 GM)
- `withanolide_cluster_genes.csv`: all 36 cluster genes with tier (cloned / not cloned / pseudogene)
- `primer_matches.csv`: cloning primer to protein matches (how tested copies were identified)
- `sterol_negatives.csv`: reciprocal best hits for 17 Arabidopsis sterol/brassinosteroid genes
- `test_labels.csv`, `test_proteins.fa`: 29 SM + 16 GM ashwagandha proteins
- `step1a_report.txt`, `step1b_report.txt`: counts and checks

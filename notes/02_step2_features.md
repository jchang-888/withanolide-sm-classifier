# Step 2 - Labels + Features
## Two label sets (option C)

* A = Moore labels as published (410 SM, 2,212 GM)
* B = A + 11 core sterol genes added as GM (2,223 GM)
	* SMT1, SMT2, SMO1-1, SMO2-1, CYP51G1, HYD1, DWF7, DWF5, DWF1, CYP710A1, CAS1
	* all were "unknown" in Moore even tho they're textbook primary metabolism
* why both?
	* A = fair test of the published labels
	* B = what a model "should" know
	* difference between A and B = finding in itself
* catch with B (important!!)
	1. the ashwagandha negatives are orthologs of these exact 11 genes
	2. so B makes the negatives easier on purpose (ofc nearest neighbor will nail them)
	3. DWF1 is GM in B, and 24ISO might be related to DWF1 ? → could pull 24ISO toward GM

## Features = turning a protein into numbers

* a model can't read letters, needs a fixed-length list of numbers
1. aac = amino acid composition (20 numbers)
	* fraction of each amino acid
	* ex. 8% leucine
2. dpc = dipeptide composition (400 numbers)
	* fraction of each pair of neighbors, 20 x 20 = 400
	* captures a tiny bit of order, unlike aac
3. physchem (6 numbers)
	* length, hydrophobicity (Kyte-Doolittle scale), charge, aromatic / tiny / proline fractions
4. esm2 (320 numbers) - run on my mac
	* protein language model = trained on millions of proteins to guess hidden amino acids
	* the "embedding" = its internal summary of a protein
	* average over every residue → 1 vector per protein
	* smallest model (8M parameters)

## Gaps

1. aac / dpc throw away almost all order info → probably weak ?
2. 1 protein had odd letters (X etc.), those got dropped before counting
3. esm cuts proteins at 1,022 residues (model limit)

# Links

* ESM-2 model: https://huggingface.co/facebook/esm2_t6_8M_UR50D
* Lin et al. 2023 Science (ESM-2): https://doi.org/10.1126/science.ade2574
* Kyte & Doolittle 1982 (hydrophobicity): https://doi.org/10.1016/0022-2836(82)90515-0

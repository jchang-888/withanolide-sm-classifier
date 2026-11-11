"""
Step 2b: ESM-2 protein language model embeddings (run this on your own computer).

Model: facebook/esm2_t6_8M_UR50D (smallest ESM-2, 8M parameters, 320 numbers per protein)
  Lin et al. 2023, Science 379:1123, https://huggingface.co/facebook/esm2_t6_8M_UR50D
Each protein -> one 320-number vector = average over all residues of the last layer.
Proteins longer than 1,022 residues are cut to the first 1,022 (model limit).

Needs: pip install torch transformers
Input:  data/processed/all_proteins.fa   (made by scripts/03_build_features.py)
Output: data/processed/features_esm2.npz

Run from the repo root:  python scripts/03b_esm_embeddings.py
First run downloads the model (~30 MB) from Hugging Face.
"""
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MODEL = "facebook/esm2_t6_8M_UR50D"
MAX_RES = 1022


def read_fasta(path):
    seqs, name, chunks = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(chunks)
            name, chunks = line[1:].split()[0], []
        elif line:
            chunks.append(line)
    if name:
        seqs[name] = "".join(chunks)
    return seqs


def main(model_name=MODEL):
    seqs = read_fasta(PROC / "all_proteins.fa")
    ids = list(seqs)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    torch.set_grad_enabled(False)

    out, t0 = [], time.time()
    for n, i in enumerate(ids, 1):
        s = seqs[i][:MAX_RES]
        enc = tok(s, return_tensors="pt")
        h = model(**enc).last_hidden_state[0]      # (tokens, 320), includes start + end tokens
        out.append(h[1:-1].mean(0).numpy())        # average over residues only
        if n % 200 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} proteins, {time.time() - t0:.0f} s")

    X = np.vstack(out)
    np.savez_compressed(PROC / "features_esm2.npz", ids=np.array(ids), X=X,
                        columns=np.array([f"esm{k}" for k in range(X.shape[1])]))
    print("saved", PROC / "features_esm2.npz", X.shape)


if __name__ == "__main__":
    main(*sys.argv[1:])

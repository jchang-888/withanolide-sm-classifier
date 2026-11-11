"""
Step 1b: build the ashwagandha (Withania somnifera) test set.

Positives (SM): withanolide cluster enzymes from the Weng Lab genome.
  Gene IDs come from the authors' code (github.com/erinreynolds/withania_somnifera,
  file 20231016_desq2.R, commit 45b8999). Which copy was actually tested comes from
  matching the cloning primers in weng_paper_supplementary_data.xlsx to protein ends.

Negatives (GM): ashwagandha versions of normal sterol and brassinosteroid enzymes,
  found as reciprocal best hits to well-known Arabidopsis genes (phmmer, pyhmmer).

Inputs:
  withania_somnifera_proteins.fa                   Zenodo 10.5281/zenodo.17932505
  weng_paper_supplementary_data.xlsx               Reynolds et al. 2026 Nature Plants, supplementary
  TAIR10_pep_20110103_representative_gene_model    TAIR10 proteins

Outputs:
  data/processed/withanolide_cluster_genes.csv     every cluster gene, name, tier, evidence
  data/processed/primer_matches.csv                each cloning primer and the protein(s) it matches
  data/processed/sterol_negatives.csv              reciprocal best hits used as GM negatives
  data/processed/test_labels.csv                   final test set (SM / GM)
  data/processed/test_proteins.fa
  data/processed/step1b_report.txt

Run from the repo root:  python scripts/02_build_test_set.py
"""
from pathlib import Path
import re
import pandas as pd
import pyhmmer
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT / "withania_somnifera_proteins.fa"
SUPP = ROOT / "weng_paper_supplementary_data.xlsx"
TAIR = ROOT / "TAIR10_pep_20110103_representative_gene_model"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)


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
    return {k: v.rstrip("*") for k, v in seqs.items()}


ws = read_fasta(WS)          # ids look like g3303.t1_1  (gene . transcript _1)
tair = read_fasta(TAIR)      # ids look like AT1G01010.1
gene_of = lambda pid: pid.split(".")[0]

# ---------------------------------------------------------------- positives
# names from the authors' script (gene id -> label they used). Copied by hand,
# source: 20231016_desq2.R lines 20-103, commit 45b89999de9fe5d3483a5be8f1d1070816c3abcf
script_names = {
    # cluster 1
    "g3303": "CYP87G1_1", "g3304": "24ISO_1", "g3305": "SULF_2", "g3306": "CYP749B2_1",
    "FUN_112423": "CYP88C10_1", "g3307": "SDH_1", "g3308": "AT_1", "g3309": "ALP1-like",
    "g3310": "CYP88C7_1", "g3311": "CYP88C10_2", "g3312b": "CYP87G1_2", "g3312": "CYP87G1_3",
    "g3313": "24ISO_2", "g3314": "SDH_2", "g3315": "CYP88C7_2", "g3316_g3317": "psuedo_CYP749B2",
    "g3318": "AT_2", "g3319": "zinc-finger-like", "g3320": "SULF_1",
    # cluster 2
    "g18181": "CYP749B2_5", "g18182": "OGD7", "g18183": "CYP749B2_4", "g18184": "CYP749B2_3",
    "g18185": "CYP88C8_2", "g18186": "pseudo_CYP749B2", "g18187": "OGD6", "g18188": "OGD5",
    "g18189": "OGD4", "g18190": "pseudo_OGD", "g18191": "CYP88C7_3", "g18192": "CYP88C8_1",
    "g18193": "OGD3", "g18194": "OGD2", "g18195": "OGD1", "g18196": "pseudo_24ISO",
    "g18197": "psuedo_CYP87G1",
}
cluster_ids = set(script_names)

# enzymes the paper reports as working pathway steps (Reynolds et al. 2026 abstract / thesis)
characterized = {"CYP87G1", "CYP749B2", "CYP88C7", "CYP88C10", "SDH2", "SULF1", "24ISO"}

# match cloning primers to protein N-terminus (fwd) or C-terminus (rev)
primers = pd.read_excel(SUPP, sheet_name="primers")
rows = []
for _, r in primers.iterrows():
    name, seq = str(r["Name"]), str(r["Sequence"]).strip()
    if not re.search(r"CYP|SULF|SDH|24_iso", name, re.I) or "qpcr" in name.lower():
        continue
    m = re.search(r"[ACGT]{15,}$", seq)            # gene-specific part is the uppercase 3' end
    if not m:
        continue
    g = m.group(0)
    if "fwd" in name.lower():
        if "ATG" not in g:
            continue
        s = g[g.find("ATG"):]
        pep = str(Seq(s[: len(s) // 3 * 3]).translate())
        if "*" in pep or len(pep) < 6:
            continue
        hits = sorted({gene_of(k) for k, v in ws.items() if v.startswith(pep)})
        end = "N-term"
    elif "rev" in name.lower():
        rc = str(Seq(g).reverse_complement())
        # reverse primer ends on the stop codon, so read it in the frame that ends at the 3' end
        pep = str(Seq(rc[len(rc) % 3:]).translate())
        if pep.endswith("*"):
            pep = pep[:-1]
        if "*" in pep or len(pep) < 5:      # an internal stop means the frame or the primer is wrong
            continue
        hits = sorted({gene_of(k) for k, v in ws.items() if v.endswith(pep)})
        end = "C-term"
    else:
        continue
    rows.append({"primer": r["Code"], "name": name, "end": end, "peptide": pep,
                 "matching_genes": ";".join(hits), "n_matches": len(hits),
                 "description": r["Description"]})
pm = pd.DataFrame(rows)
pm.to_csv(OUT / "primer_matches.csv", index=False)


def tested_genes(prefix):
    """genes matched by both a fwd and a rev cloning primer for this enzyme name"""
    sub = pm[pm["name"].str.upper().str.replace("-", "_").str.startswith(prefix.upper())]
    fwd = [set(x.split(";")) for x in sub[sub.end == "N-term"].matching_genes if x]
    rev = [set(x.split(";")) for x in sub[sub.end == "C-term"].matching_genes if x]
    out = set()
    for f in fwd:
        for rv in rev:
            out |= (f & rv)
    return out


tested = {
    "CYP87G1": tested_genes("CYP87G1"),
    "CYP749B2": tested_genes("CYP749B2"),
    "CYP88C7": tested_genes("CYP88C7"),
    "CYP88C10": tested_genes("CYP88C10"),
    "SDH2": tested_genes("SDH2"),
    "SDH1": tested_genes("SDH1"),
    "SULF1": tested_genes("SULF1"),
    "24ISO": {"g3313"} if "g3313" in pm[pm.name.str.startswith("24_iso")].matching_genes.str.cat(sep=";") else set(),
}

cl = []
for gid, label in script_names.items():
    pids = [k for k in ws if gene_of(k) == gid]
    enzyme = re.sub(r"[_-]\d+$", "", label)
    tested_as = [e for e, gs in tested.items() if gid in gs]
    if "pseudo" in label.lower() or "psuedo" in label.lower():
        tier = "pseudogene (excluded)"
    elif label in ("ALP1-like", "zinc-finger-like"):
        tier = "not an enzyme (excluded)"
    elif tested_as and tested_as[0] in characterized:
        tier = "cloned, enzyme reported active"
    elif tested_as:
        tier = "cloned, activity not confirmed here"
    else:
        tier = "cluster, not cloned"
    cl.append({"gene": gid, "script_label": label, "enzyme": enzyme,
               "cloned_as": ";".join(tested_as), "tier": tier,
               "protein_ids": ";".join(pids), "in_fasta": bool(pids)})
cl = pd.DataFrame(cl)
cl.to_csv(OUT / "withanolide_cluster_genes.csv", index=False)

# ---------------------------------------------------------------- negatives
# well-known Arabidopsis sterol / brassinosteroid (general metabolism) genes
gm_queries = {
    "AT5G13710": "SMT1", "AT1G20330": "SMT2", "AT4G12110": "SMO1-1", "AT1G07420": "SMO2-1",
    "AT1G11680": "CYP51G1", "AT1G20050": "HYD1", "AT3G02580": "DWF7", "AT1G50430": "DWF5",
    "AT3G19820": "DWF1", "AT2G34500": "CYP710A1", "AT3G50660": "DWF4/CYP90B1",
    "AT5G05690": "CPD/CYP90A1", "AT3G13730": "CYP90D1", "AT5G38970": "BR6OX1/CYP85A1",
    "AT3G30180": "BR6OX2/CYP85A2", "AT4G36380": "ROT3/CYP90C1", "AT2G07050": "CAS1",
}
alphabet = pyhmmer.easel.Alphabet.amino()


def digital(seqdict):
    out = []
    for k, v in seqdict.items():
        v = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", v.upper())
        out.append(pyhmmer.easel.TextSequence(name=k.encode(), sequence=v).digitize(alphabet))
    return out


ws_db = pyhmmer.easel.DigitalSequenceBlock(alphabet, digital(ws))
tair_db = pyhmmer.easel.DigitalSequenceBlock(alphabet, digital(tair))
queries = digital({k + ".1": tair[k + ".1"] for k in gm_queries})


def best_hit(query_seqs, db, skip=lambda n: False):
    res = {}
    for hits in pyhmmer.hmmer.phmmer(query_seqs, db, cpus=0):
        qname = hits.query.name
        qname = qname.decode() if isinstance(qname, bytes) else qname
        for h in hits:
            n = h.name.decode() if isinstance(h.name, bytes) else h.name
            if not skip(n):
                res[qname] = (n, h.evalue, h.score)
                break
    return res


fwd = best_hit(queries, ws_db, skip=lambda n: gene_of(n) in cluster_ids)
back_q = digital({v[0]: ws[v[0]] for v in fwd.values()})
back = best_hit(back_q, tair_db)

neg = []
for q, (wid, ev, sc) in fwd.items():
    agi = q.split(".")[0]
    back_hit = back.get(wid, ("", None, None))[0].split(".")[0]
    neg.append({"arabidopsis_gene": agi, "name": gm_queries[agi], "withania_protein": wid,
                "withania_gene": gene_of(wid), "evalue": ev, "bitscore": round(sc, 1),
                "back_hit": back_hit, "reciprocal": back_hit == agi})
neg = pd.DataFrame(neg)
neg.to_csv(OUT / "sterol_negatives.csv", index=False)

# ---------------------------------------------------------------- final test set
def pick_protein(gid):
    """use transcript t1 when a gene has several"""
    pids = sorted(k for k in ws if gene_of(k) == gid)
    return pids[0] if pids else None


test = []
for _, r in cl[cl.tier.isin(["cloned, enzyme reported active", "cloned, activity not confirmed here", "cluster, not cloned"])].iterrows():
    test.append({"protein": pick_protein(r.gene), "gene": r.gene, "label": "SM",
                 "name": r.cloned_as or r.enzyme, "tier": r.tier})
for _, r in neg[neg.reciprocal].drop_duplicates("withania_gene").iterrows():
    test.append({"protein": r.withania_protein, "gene": r.withania_gene, "label": "GM",
                 "name": r["name"] + " ortholog", "tier": "reciprocal best hit"})
test = pd.DataFrame(test)
test["length"] = test.protein.map(lambda p: len(ws[p]))
test.to_csv(OUT / "test_labels.csv", index=False)
with open(OUT / "test_proteins.fa", "w") as fh:
    for p in test.protein:
        fh.write(f">{p}\n{ws[p]}\n")

lines = [
    f"Withania proteins: {len(ws)}  ({len({gene_of(k) for k in ws})} genes)",
    f"Cluster genes listed in authors' script: {len(cl)}  found in FASTA: {cl.in_fasta.sum()}",
    "Cluster tiers: " + str(cl.tier.value_counts().to_dict()),
    "Primer-confirmed tested copies: " + str({k: sorted(v) for k, v in tested.items()}),
    f"Negative queries: {len(neg)}  reciprocal best hits: {neg.reciprocal.sum()}",
    "Non-reciprocal (dropped): " + str(neg[~neg.reciprocal][['name', 'withania_gene', 'back_hit']].values.tolist()),
    "Final test set: " + str(test.label.value_counts().to_dict()),
]
(OUT / "step1b_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

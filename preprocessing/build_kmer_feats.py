

import os, sys, csv, itertools
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
import config as C
csv.field_size_limit(10 ** 8)

K = 3
BASES = "AUGC"
KMERS = {"".join(p): i for i, p in enumerate(itertools.product(BASES, repeat=K))}
D = len(KMERS)
NREG = C.K_RNA_REGION


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def seq2regions(seq):
    s = "".join(c for c in str(seq).upper().replace("T", "U") if c in BASES)
    if len(s) < K:
        return torch.zeros(NREG, D)
    L = len(s) - K + 1
    onehot = np.zeros((L, D), dtype=np.float32)
    for i in range(L):
        j = KMERS.get(s[i:i + K])
        if j is not None:
            onehot[i, j] = 1
    x = torch.tensor(onehot).t().unsqueeze(0)
    xp = F.adaptive_avg_pool1d(x, min(NREG, L)).squeeze(0).t()
    if xp.shape[0] < NREG:
        xp = torch.cat([xp, torch.zeros(NREG - xp.shape[0], D)], 0)
    return xp


def main():
    name2idx = torch.load(os.path.join(C.FEAT_DIR, "rna_feats.pt"))["name2idx"]
    smap = {r["ncRNA_Name"]: r.get("Sequence", "") for r in read_csv(C.NCRNA_CSV)}
    feat = torch.zeros(len(name2idx), NREG, D)
    for name, i in name2idx.items():
        feat[i] = seq2regions(smap.get(name, ""))
    torch.save({"feat": feat, "name2idx": name2idx, "k": K},
               os.path.join(C.FEAT_DIR, "rna_kmer.pt"))
    print(f"k-mer区域特征 [{len(name2idx)},{NREG},{D}] -> {C.FEAT_DIR}/rna_kmer.pt")


if __name__ == "__main__":
    main()

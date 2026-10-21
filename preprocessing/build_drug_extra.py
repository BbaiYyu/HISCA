

import os, sys, csv
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
import config as C

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
RADIUS = 2


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():

    drug_feats = torch.load(os.path.join(C.FEAT_DIR, "drug_feats.pt"))
    name2idx = drug_feats["name2idx"]
    smi = {r["Drug_Name"]: r.get("SMILES", "") for r in read_csv(C.DRUG_CSV)}
    N = len(name2idx)
    ecfp = np.zeros((N, N_BITS), dtype=np.float32)
    miss = 0
    for name, i in name2idx.items():
        mol = Chem.MolFromSmiles(str(smi.get(name, "")))
        if mol is None:
            miss += 1
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
        for bit in fp.GetOnBits():
            ecfp[i, bit] = 1.0
    torch.save({"ecfp": torch.from_numpy(ecfp), "name2idx": name2idx},
               os.path.join(C.FEAT_DIR, "drug_extra.pt"))
    print(f"ECFP{RADIUS*2} {N_BITS}位 x {N} 个药 (无效SMILES={miss}) -> {C.FEAT_DIR}/drug_extra.pt")
    print(f"  平均激活位数={ecfp.sum(1).mean():.1f}")


if __name__ == "__main__":
    main()

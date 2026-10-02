# -*- coding: utf-8 -*-
"""为药物补【分子结构指纹】(ECFP4/Morgan, 2048位)——ChemBERTa是序列级语言模型,
易漏子结构/官能团信息; 指纹显式编码半径2的子结构, 是药物响应预测的强特征。
按建模用到的 drug_names(与 drug_feats.pt 的 name2idx 对齐)输出。
产出 features/drug_extra.pt {'ecfp':[N_drug,2048], 'name2idx':{...}}
用法: python build_drug_extra.py
"""
import os, sys, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors   # 用它取Morgan指纹, 避开AllChem在NumPy2.x下崩的3D/shape模块
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
RADIUS = 2


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    # 与 drug_feats.pt 的顺序严格对齐(load_all 用它的 name2idx)
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
        for bit in fp.GetOnBits():        # 手动置位, 不用 ConvertToNumpyArray(可能崩)
            ecfp[i, bit] = 1.0
    torch.save({"ecfp": torch.from_numpy(ecfp), "name2idx": name2idx},
               os.path.join(C.FEAT_DIR, "drug_extra.pt"))
    print(f"ECFP{RADIUS*2} {N_BITS}位 x {N} 个药 (无效SMILES={miss}) -> {C.FEAT_DIR}/drug_extra.pt")
    print(f"  平均激活位数={ecfp.sum(1).mean():.1f}")


if __name__ == "__main__":
    main()

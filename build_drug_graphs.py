# -*- coding: utf-8 -*-
"""为药物构建【原子级分子图】(可训练GCN用)——比静态指纹更强的结构编码。
原子特征 69 维(元素/杂化/度/电荷/氢数/芳香/自由基/成环), 与主流药物GNN一致。
按 drug_feats.pt 的 name2idx 顺序输出 278 个图。
产出 features/drug_graphs.pt {'graphs':[(x[n,69], edge_index[2,e]), ...], 'name2idx':{...}}
用法: python build_drug_graphs.py
"""
import os, sys, csv
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ATOM_TYPES = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B',
              'V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu',
              'Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
HYB = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
       HybridizationType.SP3D, HybridizationType.SP3D2, 'other']
DEG = [0,1,2,3,4,5]; CHG = [-2,-1,0,1,2]; NHS = [0,1,2,3,4]
MAXN = 100


def onek(v, choices):
    if v not in choices: v = choices[-1]
    return [int(v == c) for c in choices]


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(1, 69), torch.zeros(2, 0, dtype=torch.long)
    feats = []
    for a in mol.GetAtoms():
        hyb = a.GetHybridization(); hyb = hyb if hyb in HYB[:-1] else 'other'
        feats.append(onek(a.GetSymbol(), ATOM_TYPES) + onek(hyb, HYB) + onek(a.GetDegree(), DEG)
                     + onek(a.GetFormalCharge(), CHG) + onek(a.GetTotalNumHs(), NHS)
                     + [int(a.GetIsAromatic()), a.GetNumRadicalElectrons(), int(a.IsInRing())])
    feats = feats[:MAXN]
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i < len(feats) and j < len(feats):
            ei += [[i, j], [j, i]]
    x = torch.tensor(feats, dtype=torch.float)
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.zeros(2, 0, dtype=torch.long)
    return x, edge_index


def main():
    name2idx = torch.load(os.path.join(C.FEAT_DIR, "drug_feats.pt"))["name2idx"]
    smi = {r["Drug_Name"]: r.get("SMILES", "") for r in read_csv(C.DRUG_CSV)}
    graphs = [None] * len(name2idx)
    for name, i in name2idx.items():
        graphs[i] = mol_to_graph(str(smi.get(name, "")))
    torch.save({"graphs": graphs, "name2idx": name2idx, "atom_dim": 69},
               os.path.join(C.FEAT_DIR, "drug_graphs.pt"))
    print(f"药物原子图 x {len(graphs)}, 原子特征69维 -> {C.FEAT_DIR}/drug_graphs.pt")


if __name__ == "__main__":
    main()

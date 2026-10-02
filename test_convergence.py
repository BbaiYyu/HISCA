# -*- coding: utf-8 -*-
"""机理信号验证：看"ncRNA靶集 与 药物机理基因集 的汇聚程度"能否区分 关联 vs 非关联。
若正样本的汇聚分明显高于负样本(AUC>0.6) → 机理信号成立, 值得建模型分支;
若≈0.5 → 直接重叠太稀疏, 需要 PPI/通路网络做间接汇聚。
用法: python test_convergence.py
"""
import os, sys, json, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from sklearn.metrics import roc_auc_score
import pandas as pd

mir_tgt = json.load(open(os.path.join(C.DATA_DIR, "ncrna_targets.json"), encoding="utf-8"))
drug_mech = json.load(open(os.path.join(C.DATA_DIR, "drug_mech.json"), encoding="utf-8"))
mir_tgt = {k: set(v) for k, v in mir_tgt.items() if v}
drug_g = {k: set(v["genes"]) for k, v in drug_mech.items() if v["genes"]}

m = pd.read_csv(C.EDGE_CSV)
pos = m[m["rel"].isin(["resistant", "sensitive"])][["ncRNA_Name", "Drug_Name"]].drop_duplicates()
# 只看两边都有注释的 miRNA-药 对
pos = pos[pos["ncRNA_Name"].isin(mir_tgt) & pos["Drug_Name"].isin(drug_g)]
print(f"两边都有机理注释的正样本对: {len(pos)}")

rna_list = sorted(mir_tgt); drug_list = sorted(drug_g)
pos_set = set(zip(pos["ncRNA_Name"], pos["Drug_Name"]))
rng = random.Random(2024)

def scores(pairs):
    ov, jac = [], []
    for r, d in pairs:
        a, b = mir_tgt[r], drug_g[d]
        inter = len(a & b)
        ov.append(inter); jac.append(inter / (len(a | b) + 1e-9))
    return np.array(ov), np.array(jac)

# 度数匹配负样本(与正样本同分布节点)
pr = pos["ncRNA_Name"].tolist(); pd_ = pos["Drug_Name"].tolist()
neg = []
while len(neg) < len(pos):
    r = pr[rng.randrange(len(pr))]; d = pd_[rng.randrange(len(pd_))]
    if (r, d) not in pos_set:
        neg.append((r, d))

pov, pjac = scores(list(pos_set))
nov, njac = scores(neg)
y = np.r_[np.ones(len(pov)), np.zeros(len(nov))]
print(f"\n直接重叠(overlap计数): 正样本均值={pov.mean():.2f} 负样本均值={nov.mean():.2f}")
print(f"  正样本中 overlap>0 的比例: {(pov>0).mean()*100:.1f}%  (太低=需PPI/通路网络)")
print(f"  overlap-score  AUC = {roc_auc_score(y, np.r_[pov,nov]):.4f}")
print(f"  Jaccard-score  AUC = {roc_auc_score(y, np.r_[pjac,njac]):.4f}")
print("\n判读: AUC>0.6 → 机理直接汇聚有信号; ≈0.5 → 需接 PPI/通路网络做间接汇聚")

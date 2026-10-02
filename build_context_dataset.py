# -*- coding: utf-8 -*-
"""方向① 上下文数据集：从 NoncoRNA_raw 恢复 (ncRNA, 药, 癌种) → 耐药/敏感 + 机制/PMID。
体检: 癌种分布、类别均衡、与现有特征的名字重合率(可用子集), 并产出 context_data.csv。
用法: python build_context_dataset.py
"""
import os, sys, re
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

RAW = "D:/my_file/piRNA_new/原始数据库/NoncoRNA/NoncoRNA_raw.csv"


def norm_rna(n):
    """尽量归一到 features 里的名字风格; 返回若干候选。"""
    x = str(n).strip()
    cands = {x}
    if x.lower().startswith(("mir", "let")):
        cands.add("hsa-" + x)
    cands.add(re.sub(r"^hsa-", "", x))
    # miR-200b -> 加 -5p/-3p 候选
    base = "hsa-" + re.sub(r"^hsa-", "", x) if not x.lower().startswith("hsa") else x
    cands |= {base, base + "-5p", base + "-3p"}
    return cands


def main():
    raw = pd.read_csv(RAW, encoding="utf-8")
    raw["rel"] = raw["sensitive_or_resistant"].str.lower().str.strip()
    d = raw[raw["rel"].isin(["sensitive", "resistant"])].copy()
    d = d[d["cancer_type"].notna()]
    print(f"有效 (ncRNA,药,癌种,方向) 记录: {len(d)}")

    # 癌种分布
    cc = d["cancer_type"].value_counts()
    print(f"\n癌种数: {cc.size}; 样本≥100的癌种: {(cc>=100).sum()}个, ≥50的: {(cc>=50).sum()}个")
    print("Top10 癌种:")
    for c, n in cc.head(10).items():
        sub = d[d["cancer_type"] == c]["rel"].value_counts().to_dict()
        print(f"  {c:32s} {n:4d}  (耐药{sub.get('resistant',0)}/敏感{sub.get('sensitive',0)})")

    print(f"\n总体类别: 耐药{(d['rel']=='resistant').sum()} / 敏感{(d['rel']=='sensitive').sum()}")

    # 与现有特征名字重合(可用子集)
    rna_n2i = torch.load(os.path.join(C.FEAT_DIR, "rna_feats.pt"))["name2idx"]
    drug_n2i = torch.load(os.path.join(C.FEAT_DIR, "drug_feats.pt"))["name2idx"]
    rna_keys = set(rna_n2i)
    def match_rna(n):
        for c in norm_rna(n):
            if c in rna_keys:
                return c
        return None
    d["rna_matched"] = d["ncrna_name"].map(match_rna)
    d["drug_ok"] = d["drug_name"].isin(drug_n2i)
    usable = d[d["rna_matched"].notna() & d["drug_ok"]]
    print(f"\n可用子集(两边都有特征): {len(usable)}/{len(d)} = {len(usable)/len(d)*100:.0f}%")
    print(f"  可用里 癌种数={usable['cancer_type'].nunique()}, 耐药{(usable['rel']=='resistant').sum()}/敏感{(usable['rel']=='sensitive').sum()}")
    print(f"  有靶基因: {usable['target_gene'].notna().mean()*100:.0f}%, validated: {(usable['evidence_code'].str.lower()=='validated').mean()*100:.0f}%")

    out = usable[["rna_matched", "ncrna_type", "drug_name", "cancer_type", "rel",
                  "target_gene", "pmid", "evidence_code"]].rename(
        columns={"rna_matched": "ncRNA_Name", "ncrna_type": "ncRNA_Type", "drug_name": "Drug_Name"})
    out = out.drop_duplicates()
    path = os.path.join(C.DATA_DIR, "context_data.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n去重后 {len(out)} 条 -> {path}")


if __name__ == "__main__":
    main()

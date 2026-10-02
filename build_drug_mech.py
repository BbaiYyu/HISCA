# -*- coding: utf-8 -*-
"""机理分支地基②：药物→机理基因集。
有蛋白靶点 → 用 drug_targets.csv 的靶基因;
DNA损伤类药(铂类/烷化剂, 无蛋白靶点如Cisplatin占33%) → 映射到 DNA损伤应答/修复(DDR)基因集,
其它无靶点药 → 空(留给结构分支兜底)。
产出 data/drug_mech.json  {Drug_Name: {"genes":[...], "source":"target|ddr|none"}}
"""
import os, sys, csv, json, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT = os.path.join(C.DATA_DIR, "drug_mech.json")

# DNA 损伤应答/修复核心基因(用于铂类/烷化剂等直接损伤DNA、无蛋白靶点的药)
DDR_GENES = ["ERCC1", "ERCC2", "XPA", "XPC", "DDB1", "DDB2", "BRCA1", "BRCA2", "RAD51",
             "RAD51C", "PALB2", "ATM", "ATR", "CHEK1", "CHEK2", "TP53", "MDM2", "PARP1",
             "XRCC1", "XRCC3", "MLH1", "MSH2", "MSH6", "PMS2", "FANCD2", "FANCA",
             "POLB", "PCNA", "H2AX", "MGMT", "APEX1", "LIG1", "LIG4", "PRKDC"]
# 药名(小写包含)→ 走 DDR 的直接DNA损伤药
DNA_AGENTS = ["cisplatin", "carboplatin", "oxaliplatin", "nedaplatin", "lobaplatin",
              "cyclophosphamide", "ifosfamide", "temozolomide", "dacarbazine",
              "melphalan", "chlorambucil", "busulfan", "bendamustine", "mitomycin",
              "carmustine", "lomustine", "streptozocin", "thiotepa"]


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    # 建模用到的药
    edges = read_csv(C.EDGE_CSV)
    drugs = sorted({e["Drug_Name"] for e in edges
                    if e["rel"] in ("resistant", "sensitive")})
    # 药物靶点
    d2genes = {}
    for r in read_csv(C.DRUG_TGT_CSV):
        g = (r.get("Target_Gene", "") or "").replace(";", " ").split()
        if g:
            d2genes[r["Drug_Name"]] = sorted(set(g))

    out = {}
    n_tgt = n_ddr = n_none = 0
    for d in drugs:
        if d in d2genes and d2genes[d]:
            out[d] = {"genes": d2genes[d], "source": "target"}; n_tgt += 1
        elif any(a in d.lower() for a in DNA_AGENTS):
            out[d] = {"genes": DDR_GENES, "source": "ddr"}; n_ddr += 1
        else:
            out[d] = {"genes": [], "source": "none"}; n_none += 1
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    # 按关联加权覆盖
    import pandas as pd
    pos = pd.DataFrame(edges); pos = pos[pos["rel"].isin(["resistant", "sensitive"])]
    covd = {d for d in out if out[d]["genes"]}
    wcov = pos["Drug_Name"].isin(covd).mean() * 100
    print(f"药物机理基因集: 靶点{n_tgt} + DDR{n_ddr} + 无{n_none} = {len(drugs)}")
    print(f"  有机理基因的药占关联比例: {wcov:.1f}% (含救回的DNA损伤药)")
    for d in ["Cisplatin", "Oxaliplatin", "Gemcitabine", "Paclitaxel"]:
        if d in out:
            print(f"  {d}: source={out[d]['source']} n_genes={len(out[d]['genes'])}")
    print("->", OUT)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""机理分支地基①：抓全量 miRNA→靶基因(ENCORI/starBase, 高置信, 每miRNA取top-K)。
增量写 JSON 缓存, 可断点续跑。lncRNA/circRNA 暂不抓(数据稀疏, 后议)。
用法: python build_ncrna_targets.py
产出 data/ncrna_targets.json  {miRNA_name: [gene,...]}
"""
import os, sys, time, re, json
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT = os.path.join(C.DATA_DIR, "ncrna_targets.json")
TOPK = 150          # 每个 miRNA 保留支持度最高的前K个靶基因
CLIP = 2            # clipExpNum≥2: 至少2个CLIP实验支持(去噪)
API = ("https://rnasysu.com/encori/api/miRNATarget/?assembly=hg38&geneType=mRNA"
       "&miRNA={mir}&clipExpNum=" + str(CLIP) + "&degraExpNum=0&pancancerNum=0"
       "&programNum=1&program=None&target=all&cellType=all")


def variants(name):
    x = str(name).strip()
    if x.lower().startswith(("mir", "let")):
        x = "hsa-" + x
    x = x.replace("hsa-hsa-", "hsa-")
    cands = [x]
    if not re.search(r"-[35]p$", x):
        cands += [x + "-5p", x + "-3p"]
    return cands


def query(mir):
    for cand in variants(mir):
        try:
            req = urllib.request.Request(API.format(mir=cand), headers={"User-Agent": "Mozilla/5.0"})
            txt = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        except Exception:
            continue
        rows = [l.split("\t") for l in txt.splitlines()
                if l and not l.startswith("#") and not l.startswith("miRNAid")]
        if rows:
            # 列: miRNAid miRNAname geneID geneName geneType chromosome ... clipExpNum ...
            scored = []
            for p in rows:
                if len(p) > 6:
                    g = p[3]
                    try:
                        clip = int(p[6])
                    except Exception:
                        clip = 1
                    scored.append((g, clip))
            scored.sort(key=lambda x: -x[1])
            genes, seen = [], set()
            for g, _ in scored:
                if g not in seen:
                    seen.add(g); genes.append(g)
                if len(genes) >= TOPK:
                    break
            return genes
    return []


def main():
    m = pd.read_csv(C.EDGE_CSV)
    pos = m[m["rel"].isin(["resistant", "sensitive"])]
    mirs = [x for x in pos[pos["ncRNA_Type"] == "miRNA"]["ncRNA_Name"].unique()
            if str(x).lower().startswith(("hsa-", "mir", "let"))]
    cache = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    todo = [x for x in mirs if x not in cache]
    print(f"miRNA总={len(mirs)} 已缓存={len(cache)} 待抓={len(todo)}", flush=True)
    for i, mm in enumerate(todo):
        cache[mm] = query(mm)
        if (i + 1) % 50 == 0:
            json.dump(cache, open(OUT, "w", encoding="utf-8"))
            hit = sum(1 for v in cache.values() if v)
            print(f"  {i+1}/{len(todo)}  命中率={hit/len(cache)*100:.0f}%", flush=True)
        time.sleep(0.15)
    json.dump(cache, open(OUT, "w", encoding="utf-8"))
    hit = sum(1 for v in cache.values() if v)
    ng = [len(v) for v in cache.values() if v]
    import numpy as np
    print(f"完成: {len(cache)} miRNA, 命中 {hit} ({hit/len(cache)*100:.1f}%), "
          f"命中者靶基因数 中位={int(np.median(ng)) if ng else 0} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

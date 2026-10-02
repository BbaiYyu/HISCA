# -*- coding: utf-8 -*-
"""标准留一药 case study（对标 PLMF-MDA）：
把目标药的全部已知关联 mask 掉当新药 → 重训 → 对所有 ncRNA 打分排名 →
按规范名去重取 top-10 → 查 NoncoRNA_raw(PMID/癌种/靶点) → 数 confirmed。
协议 = 领域标准: 随机负样本 + 调好的模型(d_model256/lr1e-3, 用环境变量传)。
用法: NCDRUG_DMODEL=256 NCDRUG_LR=1e-3 NCDRUG_DROPOUT=0.2 python case_study_loo.py
"""
import os, sys, re, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils
from dataset import load_all, make_graph
from main import new_model

DRUGS = ["Temozolomide", "Taxol"]
EPOCHS = 50
RAW = "D:/my_file/piRNA_new/原始数据库/NoncoRNA/NoncoRNA_raw.csv"


def canon(name):
    """规范 miRNA 名: 去 hsa-/物种前缀、去 -5p/-3p、去*, 便于按同一 miRNA 去重/匹配。"""
    x = str(name).lower().strip().replace("*", "")
    x = re.sub(r"^[a-z]{3}-(?=let|mir)", "", x)  # 去物种前缀(hsa-/mmu-...)但只在其后是let/mir时,别误删mir-本身
    x = re.sub(r"-[35]p$", "", x)                # 去成熟体后缀
    return x


@torch.no_grad()
def score_all(model, data, graph, di, device, cache):
    N = data.N_rna
    sc = np.zeros(N)
    r_all = torch.arange(N)
    for b in range(0, N, C.BATCH_SIZE):
        r = r_all[b:b + C.BATCH_SIZE]
        d = torch.full((len(r),), di, dtype=torch.long)
        batch = utils.make_batch(data, r, d, device)
        out = model(batch, graph, z_net_cache=cache)
        sc[b:b + len(r)] = torch.sigmoid(out["assoc_logit"]).cpu().numpy()
    return sc


def train_leave_drug(data, di, device):
    """留一药: 训练集排除该药全部正样本, 随机负样本训练(领域标准协议)。"""
    keep = (data.pos.d != di)
    graph = make_graph(data, keep, device)
    model = new_model(data, device, arch="v2")
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    pr, pd_ = data.pos.r[keep], data.pos.d[keep]
    pos_set = set(zip(data.pos.r.tolist(), data.pos.d.tolist()))
    rng = random.Random(C.SEED)
    for ep in range(1, EPOCHS + 1):
        model.train()
        nr, nd = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, len(pr), rng)  # 随机负样本
        r = torch.cat([pr, nr]); d = torch.cat([pd_, nd])
        y = torch.cat([torch.ones(len(pr)), torch.zeros(len(nr))]).to(device)
        for idx in utils.iterate_minibatches(len(r), C.BATCH_SIZE, shuffle=True, seed=ep):
            it = torch.tensor(idx)
            batch = utils.make_batch(data, r[it], d[it], device)
            out = model(batch, graph)
            loss = F.binary_cross_entropy_with_logits(out["assoc_logit"], y[it])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    cache = None if model.arch == "v2" else model.compute_graph(graph)
    return model, graph, cache


def main():
    utils.set_seed(C.SEED)
    device = C.DEVICE
    print(f"加载数据 (d_model={C.D_MODEL}, lr={C.LR})...", flush=True)
    data = load_all()
    i2r = {i: n for n, i in data.r2i.items()}
    rtype = {0: "miRNA", 1: "lncRNA", 2: "circRNA"}
    tps = data.rna_type.numpy()
    pos_r, pos_d = data.pos.r.tolist(), data.pos.d.tolist()
    raw = pd.read_csv(RAW, encoding="utf-8")
    raw["c"] = raw["ncrna_name"].map(canon)
    raw["rel"] = raw["sensitive_or_resistant"].str.lower()

    rows = []
    for drug in DRUGS:
        if drug not in data.d2i:
            print(f"[{drug}] 不在数据, 跳过"); continue
        di = data.d2i[drug]
        known_canon = {canon(i2r[r]) for r, d in zip(pos_r, pos_d) if d == di}
        print(f"\n===== 留一药: {drug} (mask掉{sum(1 for d in pos_d if d==di)}条已知关联, 当新药) =====", flush=True)
        model, graph, cache = train_leave_drug(data, di, device)
        sc = score_all(model, data, graph, di, device, cache)
        # 按规范名去重, 每个 ncRNA 留最高分变体
        best = {}
        for r in range(data.N_rna):
            c = canon(i2r[r])
            if c not in best or sc[r] > best[c][0]:
                best[c] = (sc[r], i2r[r], tps[r])
        # 【分类型】每种 ncRNA 类型各取 top-5, 展示通用性
        for ti, tname in rtype.items():
            # 类型内 min-max 归一化: 用该类型【全部】去重 ncRNA 的分数范围
            type_vals = [v[0] for c, v in best.items() if v[2] == ti]
            if not type_vals:
                continue
            tmin, tmax = float(min(type_vals)), float(max(type_vals))
            rng_t = (tmax - tmin) or 1e-12
            ranked = sorted([(c, v) for c, v in best.items() if v[2] == ti],
                            key=lambda x: -x[1][0])[:5]
            conf = 0
            print(f"  --- {tname} top-5 (类型内 min={tmin:.4g}, max={tmax:.4g}) ---")
            for i, (c, (s, name, tp)) in enumerate(ranked, 1):
                is_known = c in known_canon
                conf += is_known
                snorm = (float(s) - tmin) / rng_t            # 类型内归一化分 [0,1]
                lit = raw[(raw["c"] == c) & (raw["drug_name"].str.contains(drug, case=False, na=False))]
                litstr = ""
                if len(lit):
                    h = lit.iloc[0]
                    litstr = f"{h['cancer_type']}|{h['rel']}|靶{h['target_gene']}|PMID{h['pmid']}|{h['evidence_code']}"
                mark = "✓已知" if is_known else "新预测"
                print(f"  {i:<3}{name:<20}raw={s:.5f}  norm={snorm:.3f}  {mark:<6}{litstr}")
                rows.append({"drug": drug, "type": tname, "rank": i, "ncRNA": name,
                             "score_raw": round(float(s), 6), "score_norm": round(snorm, 4),
                             "known": is_known, "literature": litstr})
            print(f"    → {tname}: {conf}/5 命中已知")

    out = os.path.join(C.DATA_DIR, "..", "case_study_loo.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已写出 -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

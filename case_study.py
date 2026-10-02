# -*- coding: utf-8 -*-
"""标准 case study（二分类关联）：
留出10%已知关联→用剩下训练→对重点药给所有ncRNA打分排名→
①看留出关联能否recover(证明抓真信号) ②列top预测里的新关联当假设(供查PMID)。
用法: python case_study.py
产出 case_study_top.csv + 终端报告。
"""
import os, sys, random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils
from dataset import load_all, make_graph
from main import new_model

DRUGS = ["Cisplatin", "Gemcitabine", "Doxorubicin", "Paclitaxel", "Sorafenib", "Gefitinib", "Tamoxifen"]
EPOCHS = 60
TOPN = 15


@torch.no_grad()
def score_drug(model, data, graph, d_idx, device, cache):
    """对固定药 d_idx, 给所有 ncRNA 打关联分。返回 [N_rna]。"""
    N = data.N_rna
    scores = np.zeros(N)
    r_all = torch.arange(N)
    for b in range(0, N, C.BATCH_SIZE):
        r = r_all[b:b + C.BATCH_SIZE]
        d = torch.full((len(r),), d_idx, dtype=torch.long)
        batch = utils.make_batch(data, r, d, device)
        out = model(batch, graph, z_net_cache=cache)
        scores[b:b + len(r)] = torch.sigmoid(out["assoc_logit"]).cpu().numpy()
    return scores


def main():
    utils.set_seed(C.SEED)
    device = C.DEVICE
    print("加载数据(二分类关联)...", flush=True)
    data = load_all()
    i2r = {i: n for n, i in data.r2i.items()}
    rna_type_name = {0: "miRNA", 1: "lncRNA", 2: "circRNA"}
    rtype = data.rna_type.numpy() if getattr(data, "rna_type", None) is not None else None

    # 留出 10% 正样本作测试(recover 验证)
    n_pos = len(data.pos.r)
    tr_mask, te_mask = utils.split_positives(n_pos, 0.1, C.SEED)
    train_pairs = set(zip(data.pos.r[tr_mask].tolist(), data.pos.d[tr_mask].tolist()))
    test_pairs = set(zip(data.pos.r[te_mask].tolist(), data.pos.d[te_mask].tolist()))
    all_pos = train_pairs | test_pairs
    print(f"训练关联={len(train_pairs)} 留出关联={len(test_pairs)}", flush=True)

    # 训练(仅二分类BCE, 用训练集正样本+随机负样本)
    graph = make_graph(data, tr_mask, device)
    model = new_model(data, device, ablation="none", tgt_feat_mode="real", arch="v2")
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    rng = random.Random(C.SEED)
    pr, pd = data.pos.r[tr_mask], data.pos.d[tr_mask]
    pos_set = set(zip(data.pos.r.tolist(), data.pos.d.tolist()))
    # 硬负样本(度数匹配): 逼模型学药物特异性, 不走"推热门miRNA"的度数捷径
    rp, dp = utils.build_neg_pools(pr, pd)
    for ep in range(1, EPOCHS + 1):
        model.train()
        nr, nd = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, len(pr), rng, rp, dp)
        r = torch.cat([pr, nr]); d = torch.cat([pd, nd])
        y = torch.cat([torch.ones(len(pr)), torch.zeros(len(nr))]).to(device)
        tot = 0.0
        for idx in utils.iterate_minibatches(len(r), C.BATCH_SIZE, shuffle=True, seed=ep):
            it = torch.tensor(idx)
            batch = utils.make_batch(data, r[it], d[it], device)
            out = model(batch, graph)
            loss = F.binary_cross_entropy_with_logits(out["assoc_logit"], y[it])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 15 == 0:
            print(f"  ep{ep} loss={tot:.1f}", flush=True)

    # 打分 + 报告
    model.eval()
    cache = None if model.arch == "v2" else model.compute_graph(graph)
    rows = []
    print("\n" + "=" * 70)
    for dn in DRUGS:
        if dn not in data.d2i:
            print(f"[{dn}] 不在数据里, 跳过"); continue
        di = data.d2i[dn]
        sc = score_drug(model, data, graph, di, device, cache)
        order = np.argsort(-sc)
        # recover: 该药的留出关联排名
        heldout_r = [r for (r, dd) in test_pairs if dd == di]
        rank = {int(r): int(np.where(order == r)[0][0]) for r in heldout_r}
        in_top = sum(1 for r in heldout_r if rank[r] < 50)
        print(f"\n【{dn}】留出关联{len(heldout_r)}条, 其中排进Top50的: {in_top}/{len(heldout_r)}"
              + (f" (中位排名{int(np.median(list(rank.values())))})" if rank else ""))
        shown = 0
        for r in order:
            if shown >= TOPN: break
            r = int(r)
            if (r, di) in train_pairs:
                flag = "已知(训练)"
            elif (r, di) in test_pairs:
                flag = "★留出-recover"
            else:
                flag = "新预测(假设)"
            tp = rna_type_name.get(rtype[r], "?") if rtype is not None else ""
            print(f"    {sc[r]:.3f}  {i2r[r]:<22s} [{tp}] {flag}")
            rows.append({"drug": dn, "ncRNA": i2r[r], "type": tp, "score": round(float(sc[r]), 4), "flag": flag})
            shown += 1
    import csv
    out = os.path.join(C.DATA_DIR, "..", "case_study_top.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["drug", "ncRNA", "type", "score", "flag"]); w.writeheader(); w.writerows(rows)
    print(f"\n已写出 -> {os.path.abspath(out)}")
    print("说明: ★留出-recover 排在前列=模型抓到真信号; '新预测'去查PMID即case study素材")


if __name__ == "__main__":
    main()

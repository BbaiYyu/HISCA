# -*- coding: utf-8 -*-
"""负样本难度 α 扫描: 每个 α 同时给 模型AUC 和 度数基线AUC(诚实尺子)。
度数基线高=该α下"节点热度"就能作弊, 模型AUC里的水分大。挑一个模型AUC够看、度数基线不虚高的点。
用法: python sweep_hardfrac.py
"""
import os, sys, random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NCDRUG_DMODEL", "256")
os.environ.setdefault("NCDRUG_LR", "1e-3")
os.environ.setdefault("NCDRUG_DROPOUT", "0.2")
import config as C
import utils
from dataset import load_all, make_graph
from main import new_model

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPOCHS = 55


def degree_baseline_auc(data, tr_mask, te_mask, hard_frac, rng):
    """在给定α的测试负样本上, 用 度数乘积 打分算AUC(平凡基线)。"""
    pr, pd = data.pos.r.numpy(), data.pos.d.numpy()
    degr = np.bincount(pr, minlength=data.N_rna).astype(float)
    degd = np.bincount(pd, minlength=data.N_drug).astype(float)
    pos_set = set(zip(pr.tolist(), pd.tolist()))
    rp, dp = utils.build_neg_pools(data.pos.r[tr_mask], data.pos.d[tr_mask])
    te = te_mask
    ptr, ptd = data.pos.r[te].numpy(), data.pos.d[te].numpy()
    nr, nd = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, len(ptr),
                                    random.Random(1), rp, dp, hard_frac=hard_frac)
    nr, nd = nr.numpy(), nd.numpy()
    y = np.r_[np.ones(len(ptr)), np.zeros(len(nr))]
    s = np.r_[degr[ptr] * degd[ptd], degr[nr] * degd[nd]]
    return roc_auc_score(y, s)


def run_alpha(data, device, alpha):
    os.environ["NCDRUG_HARD_FRAC"] = str(alpha)
    utils.set_seed(C.SEED)
    tr, te = utils.split_positives(len(data.pos.r), 0.2, C.SEED)
    inner, val = utils.split_train_val(tr, 0.15, C.SEED)
    graph = make_graph(data, inner, device)
    neg_pools = utils.build_neg_pools(data.pos.r, data.pos.d)
    model = new_model(data, device, arch="v2")
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    rng = random.Random(C.SEED)
    from main import build_pool, evaluate
    best_val, best_te = -1, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        r, d, y, rel = build_pool(data, inner, rng, C.NEG_RATIO, neg_pools)
        y = y.to(device)
        for idx in utils.iterate_minibatches(len(r), C.BATCH_SIZE, shuffle=True, seed=ep):
            it = torch.tensor(idx)
            batch = utils.make_batch(data, r[it], d[it], device)
            out = model(batch, graph)
            loss = F.binary_cross_entropy_with_logits(out["assoc_logit"], y[it])
            opt.zero_grad(); loss.backward(); opt.step()
        mv = evaluate(model, data, graph, val, device, rng, neg_pools)
        mt = evaluate(model, data, graph, te, device, rng, neg_pools)
        if mv["AUC"] > best_val:
            best_val, best_te = mv["AUC"], mt["AUC"]
    dbase = degree_baseline_auc(data, tr, te, alpha, rng)
    return best_te, dbase


def main():
    device = C.DEVICE
    data = load_all()
    print(f"{'α(硬度)':<10}{'模型AUC':<12}{'度数基线AUC':<14}{'净增(模型-基线)':<14}")
    for a in ALPHAS:
        mauc, dbase = run_alpha(data, device, a)
        tag = "随机" if a == 0 else ("全硬" if a == 1 else "混合")
        print(f"{a:<10.2f}{mauc:<12.4f}{dbase:<14.4f}{mauc-dbase:<+14.4f}  ({tag})", flush=True)
    print("\n判读: 度数基线越低越诚实(全随机~0.98虚高); 挑模型AUC够看、度数基线可接受、净增大的α。")


if __name__ == "__main__":
    main()

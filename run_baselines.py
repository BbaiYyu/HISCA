# -*- coding: utf-8 -*-
"""在【与主模型完全相同的协议】(5折 + 同难度负样本)下跑 4 个对比基线:
  RF        随机森林(拼接RNA-FM+ChemBERTa池化特征)
  XGB       XGBoost(同特征)
  GCN       2层图卷积链路预测(二部图, 训练集正样本建边, 点积打分)
  GCMF      图正则矩阵分解(GCMDR式: ncRNA/drug隐因子, 相似图拉普拉斯正则)
用法: NCDRUG_HARD_FRAC=0.5 python run_baselines.py
"""
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils
from dataset import load_all

HARD = float(os.environ.get("NCDRUG_HARD_FRAC", "0.5"))
DEV = C.DEVICE
K = 5


def pooled_feats(data):
    rna = data.rna_feat.mean(1)                                  # [N_rna,640]
    dm = data.drug_mask.unsqueeze(-1).float()
    drug = (data.drug_feat * dm).sum(1) / dm.sum(1).clamp(min=1)  # [N_drug,384]
    return rna.numpy(), drug.numpy()


def make_xy(pr, pd, nr, nd, rna, drug):
    r = np.r_[pr, nr]; d = np.r_[pd, nd]
    X = np.hstack([rna[r], drug[d]])
    y = np.r_[np.ones(len(pr)), np.zeros(len(nr))]
    return X, y


def gcn_fold(data, tr_r, tr_d, te_pr, te_pd, te_nr, te_nd, rna, drug, epochs=60, return_scores=False):
    """2层GCN链路预测: 二部图(训练正样本), 学节点嵌入, 点积打分。"""
    Nr, Nd = data.N_rna, data.N_drug
    N = Nr + Nd
    # 邻接(对称归一化), 训练正样本边 + 自环
    src = np.r_[tr_r, tr_d + Nr, np.arange(N)]
    dst = np.r_[tr_d + Nr, tr_r, np.arange(N)]
    idx = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=DEV)
    val = torch.ones(idx.shape[1], device=DEV)
    A = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
    dinv = deg.pow(-0.5)
    # 初始特征(投影到128)
    X0 = torch.tensor(np.vstack([np.hstack([rna, np.zeros((Nr, drug.shape[1]))]),
                                 np.hstack([np.zeros((Nd, rna.shape[1])), drug])]),
                      dtype=torch.float32, device=DEV)
    lin0 = nn.Linear(X0.shape[1], 128).to(DEV)
    W1 = nn.Linear(128, 128).to(DEV); W2 = nn.Linear(128, 128).to(DEV)
    opt = torch.optim.Adam(list(lin0.parameters()) + list(W1.parameters()) + list(W2.parameters()), lr=1e-2, weight_decay=1e-5)

    def propagate(H):
        H = dinv.unsqueeze(1) * H
        H = torch.sparse.mm(A, H)
        return dinv.unsqueeze(1) * H

    tr_r_t = torch.tensor(tr_r, device=DEV); tr_d_t = torch.tensor(tr_d + Nr, device=DEV)
    rng = np.random.RandomState(0)
    for ep in range(epochs):
        lin0.train()
        H = F.relu(lin0(X0)); H = F.relu(W1(propagate(H))); H = W2(propagate(H))
        # 训练负样本(随机)
        nnr = torch.tensor(rng.randint(0, Nr, len(tr_r)), device=DEV)
        nnd = torch.tensor(rng.randint(0, Nd, len(tr_r)) + Nr, device=DEV)
        pos = (H[tr_r_t] * H[tr_d_t]).sum(1)
        neg = (H[nnr] * H[nnd]).sum(1)
        loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + \
               F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        opt.zero_grad(); loss.backward(); opt.step()
    lin0.eval()
    with torch.no_grad():
        H = F.relu(lin0(X0)); H = F.relu(W1(propagate(H))); H = W2(propagate(H))
        pr = torch.tensor(te_pr, device=DEV); pdt = torch.tensor(te_pd + Nr, device=DEV)
        nr = torch.tensor(te_nr, device=DEV); ndt = torch.tensor(te_nd + Nr, device=DEV)
        sp = (H[pr] * H[pdt]).sum(1).cpu().numpy()
        sn = (H[nr] * H[ndt]).sum(1).cpu().numpy()
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]; s = np.r_[sp, sn]
    if return_scores: return s
    return roc_auc_score(y, s), average_precision_score(y, s)


def gcmf_fold(data, tr_r, tr_d, te_pr, te_pd, te_nr, te_nd, epochs=200, dim=64, return_scores=False):
    """图正则矩阵分解(GCMDR式): U[Nr,dim] V[Nd,dim], 训练正样本重建 + 相似图拉普拉斯正则。"""
    Nr, Nd = data.N_rna, data.N_drug
    U = nn.Parameter(torch.randn(Nr, dim, device=DEV) * 0.1)
    V = nn.Parameter(torch.randn(Nd, dim, device=DEV) * 0.1)
    opt = torch.optim.Adam([U, V], lr=0.05, weight_decay=1e-4)
    tr_r_t = torch.tensor(tr_r, device=DEV); tr_d_t = torch.tensor(tr_d, device=DEV)
    rng = np.random.RandomState(0)
    for ep in range(epochs):
        pos = (U[tr_r_t] * V[tr_d_t]).sum(1)
        nnr = torch.tensor(rng.randint(0, Nr, len(tr_r)), device=DEV)
        nnd = torch.tensor(rng.randint(0, Nd, len(tr_r)), device=DEV)
        neg = (U[nnr] * V[nnd]).sum(1)
        loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + \
               F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sp = (U[torch.tensor(te_pr, device=DEV)] * V[torch.tensor(te_pd, device=DEV)]).sum(1).cpu().numpy()
        sn = (U[torch.tensor(te_nr, device=DEV)] * V[torch.tensor(te_nd, device=DEV)]).sum(1).cpu().numpy()
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]; s = np.r_[sp, sn]
    if return_scores: return s
    return roc_auc_score(y, s), average_precision_score(y, s)


def main():
    data = load_all()
    rna, drug = pooled_feats(data)
    pos_r, pos_d = data.pos.r.numpy(), data.pos.d.numpy()
    pos_set = set(zip(pos_r.tolist(), pos_d.tolist()))
    rp, dp = utils.build_neg_pools(data.pos.r, data.pos.d)
    from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
    METS = ["AUC", "AUPR", "ACC", "Prec", "Recall", "F1", "MCC"]

    def metrics(y, s):
        thr = np.median(s)                                   # 1:1正负, 中位阈值二值化
        pred = (s > thr).astype(int)
        return {"AUC": roc_auc_score(y, s), "AUPR": average_precision_score(y, s),
                "ACC": accuracy_score(y, pred), "Prec": precision_score(y, pred, zero_division=0),
                "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
                "MCC": matthews_corrcoef(y, pred)}

    splits = utils.kfold_split(len(data.pos.r), K, C.SEED)
    res = {m: {k: [] for k in METS} for m in ["RF", "XGB", "GCN", "GCMF"]}
    for fi, (trm, tem) in enumerate(splits):
        rng = random.Random(100 + fi)
        tpr, tpd = pos_r[trm.numpy()], pos_d[trm.numpy()]
        epr, epd = pos_r[tem.numpy()], pos_d[tem.numpy()]
        tnr, tnd = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, len(tpr), rng, rp, dp, hard_frac=HARD)
        enr, end = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, len(epr), rng, rp, dp, hard_frac=HARD)
        tnr, tnd, enr, end = [x.numpy() for x in (tnr, tnd, enr, end)]
        Xtr, ytr = make_xy(tpr, tpd, tnr, tnd, rna, drug)
        Xte, yte = make_xy(epr, epd, enr, end, rna, drug)
        rf = RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1, random_state=0)
        rf.fit(Xtr, ytr); s_rf = rf.predict_proba(Xte)[:, 1]
        xg = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               subsample=0.8, n_jobs=-1, eval_metric="logloss", verbosity=0)
        xg.fit(Xtr, ytr); s_xg = xg.predict_proba(Xte)[:, 1]
        s_gcn = gcn_fold(data, tpr, tpd, epr, epd, enr, end, rna, drug, return_scores=True)
        s_gcmf = gcmf_fold(data, tpr, tpd, epr, epd, enr, end, return_scores=True)
        for m, s in [("RF", s_rf), ("XGB", s_xg), ("GCN", s_gcn), ("GCMF", s_gcmf)]:
            for k, v in metrics(yte, s).items():
                res[m][k].append(v)
        print(f"[fold {fi}] done", flush=True)
    print(f"\n=== 基线多指标 (α={HARD}, 5折) ===")
    print("Method  " + "  ".join(f"{k:>7s}" for k in METS))
    for m in res:
        print(f"{m:6s} " + "  ".join(f"{np.mean(res[m][k]):.4f}" for k in METS))


if __name__ == "__main__":
    main()

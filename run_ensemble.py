# -*- coding: utf-8 -*-
"""TRHGNet ⊕ XGBoost 融合(TRHGNet-Fusion): 每折同时训练两者, 在相同测试对上z标准化后加权平均。
逻辑: XGBoost 强在池化表格特征; TRHGNet 的交互项 z_pair 抓到树模型抓不到的 区域×子结构 交互。
用法: NCDRUG_HARD_FRAC=1.0 python run_ensemble.py
"""
import os, sys, random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NCDRUG_DMODEL", "256")
os.environ.setdefault("NCDRUG_LR", "1e-3")
os.environ.setdefault("NCDRUG_DROPOUT", "0.2")
import config as C
import utils
from dataset import load_all, make_graph
from main import new_model, build_pool

HARD = float(os.environ.get("NCDRUG_HARD_FRAC", "1.0"))
DEV = C.DEVICE
K = 5
EPOCHS = 70


def pooled(data):
    rna = data.rna_feat.mean(1).numpy()
    dm = data.drug_mask.unsqueeze(-1).float()
    drug = ((data.drug_feat * dm).sum(1) / dm.sum(1).clamp(min=1)).numpy()
    return rna, drug


def zn(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def trhgnet_scores(data, inner, val, te_pr, te_pd, te_nr, te_nd, neg_pools):
    """训练TRHGNet, 返回测试正/负样本的预测分。"""
    os.environ["NCDRUG_HARD_FRAC"] = str(HARD)
    graph = make_graph(data, inner, DEV)
    model = new_model(data, DEV, arch="v2")
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    rng = random.Random(C.SEED)
    for ep in range(1, EPOCHS + 1):
        model.train()
        r, d, y, rel = build_pool(data, inner, rng, C.NEG_RATIO, neg_pools)
        y = y.to(DEV)
        for idx in utils.iterate_minibatches(len(r), C.BATCH_SIZE, shuffle=True, seed=ep):
            it = torch.tensor(idx)
            batch = utils.make_batch(data, r[it], d[it], DEV)
            out = model(batch, graph)
            loss = F.binary_cross_entropy_with_logits(out["assoc_logit"], y[it])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    def score(rr, dd):
        s = []
        rr = torch.tensor(rr); dd = torch.tensor(dd)
        with torch.no_grad():
            for i in range(0, len(rr), 4096):
                b = utils.make_batch(data, rr[i:i+4096], dd[i:i+4096], DEV)
                s.append(torch.sigmoid(model(b, graph)["assoc_logit"]).cpu().numpy())
        return np.concatenate(s)
    return score(te_pr, te_pd), score(te_nr, te_nd)


def main():
    data = load_all()
    rna, drug = pooled(data)
    pos_r, pos_d = data.pos.r.numpy(), data.pos.d.numpy()
    pos_set = set(zip(pos_r.tolist(), pos_d.tolist()))
    rp, dp = utils.build_neg_pools(data.pos.r, data.pos.d)
    splits = utils.kfold_split(len(data.pos.r), K, C.SEED)
    res = {m: [] for m in ["RF", "XGB", "TRHGNet", "Fusion"]}
    for fi, (trm, tem) in enumerate(splits):
        rng = random.Random(100 + fi)
        epr, epd = pos_r[tem.numpy()], pos_d[tem.numpy()]
        enr, end = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, len(epr), rng, rp, dp, hard_frac=HARD)
        enr, end = enr.numpy(), end.numpy()
        tpr, tpd = pos_r[trm.numpy()], pos_d[trm.numpy()]
        tnr, tnd = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, len(tpr), rng, rp, dp, hard_frac=HARD)
        tnr, tnd = tnr.numpy(), tnd.numpy()
        # XGBoost
        Xtr = np.hstack([rna[np.r_[tpr, tnr]], drug[np.r_[tpd, tnd]]])
        ytr = np.r_[np.ones(len(tpr)), np.zeros(len(tnr))]
        Xte = np.hstack([rna[np.r_[epr, enr]], drug[np.r_[epd, end]]])
        yte = np.r_[np.ones(len(epr)), np.zeros(len(enr))]
        xg = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               subsample=0.8, n_jobs=-1, eval_metric="logloss", verbosity=0)
        xg.fit(Xtr, ytr)
        s_xgb = xg.predict_proba(Xte)[:, 1]
        rf = RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1, random_state=0)
        rf.fit(Xtr, ytr)
        s_rf = rf.predict_proba(Xte)[:, 1]
        # TRHGNet
        inner, val = utils.split_train_val(trm, 0.15, C.SEED + fi)
        neg_pools = utils.build_neg_pools(data.pos.r, data.pos.d)
        sp, sn = trhgnet_scores(data, inner, val, epr, epd, enr, end, neg_pools)
        s_trh = np.r_[sp, sn]
        # 融合(z标准化平均): 最强树模型RF + XGB + TRHGNet交互信号
        s_fus = zn(s_rf) + zn(s_xgb) + zn(s_trh)
        res["RF"].append(roc_auc_score(yte, s_rf))
        res["XGB"].append(roc_auc_score(yte, s_xgb))
        res["TRHGNet"].append(roc_auc_score(yte, s_trh))
        res["Fusion"].append(roc_auc_score(yte, s_fus))
        print(f"[fold {fi}] RF {res['RF'][-1]:.4f}  XGB {res['XGB'][-1]:.4f}  "
              f"TRHGNet {res['TRHGNet'][-1]:.4f}  Fusion {res['Fusion'][-1]:.4f}", flush=True)
    print(f"\n=== α={HARD} 5折 (RF={0.815 if HARD==1.0 else 0.898}) ===")
    for m in res:
        a = np.array(res[m]); print(f"{m:10s} {a.mean():.4f} ± {a.std():.4f}")


if __name__ == "__main__":
    main()

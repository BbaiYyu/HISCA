# -*- coding: utf-8 -*-
"""
训练 + 评估主入口（合并版：诚实评测）。
损失 = 是否关联(BCE) + λ·关系类型(CE，仅正样本)。
诚实机制：训练集切出验证集 → 用验证集选最佳轮 → 报该轮测试指标（不偷看测试集）。
用法:
    python main.py                                   # 完整模型, 5折, 随机负样本
    python main.py --ablation no_film                # 某个消融
    python main.py --neg degree                      # 硬负样本(堵度数捷径, 让0.97掉下来)
    python main.py --coldstart ncRNA                 # 冷启动(测新ncRNA)
    python main.py --ablation gcn --neg degree       # 消融 × 硬负样本 组合
"""
import os, sys, argparse
import numpy as np
import torch
# 本机 torch_scatter 是 CPU-only 编译版, v2的药物图 global_*_pool 在CUDA会崩; 关掉扩展走PyG原生scatter。
import torch_geometric.typing
torch_geometric.typing.WITH_TORCH_SCATTER = False
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils
from dataset import load_all, make_graph
from model import TRHGNet


def build_pool(data, mask, rng, neg_ratio, neg_mode="random", degree_pools=None, rf_pool=None):
    """正样本(mask) + 采样负样本 → (r,d,assoc_label,rel_label)。负样本 rel_label=-1。
    neg_mode: random=均匀随机 / degree=度数匹配硬负 / rf=从RF可靠负样本对池里取。"""
    pr, pd, prel = data.pos.r[mask], data.pos.d[mask], data.pos.rel[mask]
    n_neg = int(len(pr) * neg_ratio)
    if neg_mode == "rf" and rf_pool is not None:
        nr, nd = utils.sample_from_pairpool(rf_pool[0], rf_pool[1], n_neg, rng)
    else:
        pos_set = set(zip(data.pos.r.tolist(), data.pos.d.tolist()))
        rp, dp = degree_pools if (neg_mode == "degree" and degree_pools) else (None, None)
        hf = os.environ.get("NCDRUG_HARD_FRAC")        # 负样本难度系数α∈[0,1]: 介于随机(0)与硬(1)之间
        hf = float(hf) if hf is not None else None
        nr, nd = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, n_neg, rng, rp, dp, hard_frac=hf)
    r = torch.cat([pr, nr]); d = torch.cat([pd, nd])
    y = torch.cat([torch.ones(len(pr)), torch.zeros(len(nr))])
    rel = torch.cat([prel, -torch.ones(len(nr), dtype=torch.long)])
    return r, d, y, rel


@torch.no_grad()
def evaluate(model, data, graph, test_mask, device, rng, neg_mode="random", degree_pools=None):
    model.eval()
    r, d, y, rel = build_pool(data, test_mask, rng, C.NEG_RATIO, neg_mode, degree_pools, None)
    if model.arch == "v2":                                   # v2: 只在开了HGT时缓存图嵌入
        cache = model.compute_hgt(graph) if getattr(model, "use_hgt", False) else None
    elif model.ablation == "no_graph":
        cache = None
    else:
        cache = model.compute_graph(graph)                   # 图只算一次
    scores, rel_pred, rel_true = [], [], []
    for idx in utils.iterate_minibatches(len(r), C.BATCH_SIZE, shuffle=False):
        idx = torch.tensor(idx)
        batch = utils.make_batch(data, r[idx], d[idx], device)
        out = model(batch, graph, z_net_cache=cache)
        scores.append(torch.sigmoid(out["assoc_logit"]).cpu())
        pos = rel[idx] >= 0
        if pos.any():
            rel_pred.append(out["rel_logits"][pos].cpu())
            rel_true.append(rel[idx][pos])
    scores = torch.cat(scores).numpy()
    rl = torch.cat(rel_pred) if rel_pred else None
    rt = torch.cat(rel_true).numpy() if rel_true else []
    m = utils.eval_metrics(scores, y.numpy(), rl, rt)
    # 分类型 AUC：按每对 ncRNA 的类型(miRNA/lncRNA/circRNA)拆开算
    if getattr(data, "rna_type", None) is not None:
        from sklearn.metrics import roc_auc_score
        yv = y.numpy(); tp = data.rna_type[r].numpy()
        for ti, tn in [(0, "miRNA"), (1, "lncRNA"), (2, "circRNA")]:
            mk = tp == ti
            if mk.sum() > 10 and len(set(yv[mk])) == 2:
                m["AUC_" + tn] = roc_auc_score(yv[mk], scores[mk])
                m["N_" + tn] = int(mk.sum())
    return m


def new_model(data, device, ablation="none", tgt_feat_mode="real", arch="orig"):
    return TRHGNet(
        num_targets=data.num_targets, metadata=data.metadata,
        n_rna=data.N_rna, n_drug=data.N_drug,
        d_rna_enc=C.D_RNA_ENC, d_drug_enc=C.D_DRUG_ENC, d_model=C.D_MODEL,
        num_rel=C.NUM_REL, film_hidden=C.FILM_HIDDEN,
        n_hgt_layer=C.N_HGT_LAYER, n_hgt_head=C.N_HGT_HEAD,
        n_hyper_layer=C.N_HYPER_LAYER, head_hidden=C.HEAD_HIDDEN, dropout=C.DROPOUT,
        ablation=ablation, tgt_feat=data.tgt_feat, tgt_feat_mode=tgt_feat_mode,
        arch=arch, drug_ecfp=getattr(data, "drug_ecfp", None),
        drug_graph=getattr(data, "drug_graph", None),
        rna_type=getattr(data, "rna_type", None),
    ).to(device)


def run_fold(fold, data, train_mask, test_mask, device, epochs, rng,
             ablation="none", patience=0, neg_mode="random", eval_neg_mode=None,
             degree_pools=None, tgt_feat_mode="real", arch="orig"):
    """训练评估一折（诚实版）：训练集再切出验证集，用【验证集】选最佳轮，报该轮的【测试集】指标。
    图只用 inner 边构（验证/测试边都不进图，不泄漏）。patience>0 时按验证AUC早停。
    neg_mode=训练负样本(random/degree/rf); eval_neg_mode=评估负样本(默认同 neg_mode, rf时独立指定)。"""
    if eval_neg_mode is None:
        eval_neg_mode = neg_mode
    inner_mask, val_mask = utils.split_train_val(train_mask, 0.15, C.SEED + fold)
    graph = make_graph(data, inner_mask, device)             # 图只用 inner，不含验证/测试边
    # RF 可靠负样本池：仅用当折 inner 正样本训 RF，候选排除全部已知关联(无泄漏)，每折建一次
    rf_pool = None
    if neg_mode == "rf":
        pool_mult = int(os.environ.get("NCDRUG_RF_POOL_MULT", 3))
        n_pool = int(int(inner_mask.sum()) * C.NEG_RATIO * pool_mult)
        rf_pool = utils.build_rf_reliable_pool(data, inner_mask, n_pool, C.SEED + fold)
    model = new_model(data, device, ablation, tgt_feat_mode, arch)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    best_val, best_test, best_ep, wait = -1, {}, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        r, d, y, rel = build_pool(data, inner_mask, rng, C.NEG_RATIO, neg_mode, degree_pools, rf_pool)
        y, rel = y.to(device), rel.to(device)
        tot = 0.0
        for idx in utils.iterate_minibatches(len(r), C.BATCH_SIZE, shuffle=True, seed=epoch):
            idx_t = torch.tensor(idx)
            batch = utils.make_batch(data, r[idx_t], d[idx_t], device)
            out = model(batch, graph)
            loss = F.binary_cross_entropy_with_logits(out["assoc_logit"], y[idx_t])
            pos = rel[idx_t] >= 0
            if pos.any():
                loss = loss + C.LAMBDA_REL * F.cross_entropy(
                    out["rel_logits"][pos], rel[idx_t][pos])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        # 评估负样本独立(不用rf): 保持诚实且与 baseline 可比
        mv = evaluate(model, data, graph, val_mask, device, rng, eval_neg_mode, degree_pools)    # 验证集：只用来选轮
        mt = evaluate(model, data, graph, test_mask, device, rng, eval_neg_mode, degree_pools)   # 测试集：只记录/报告
        if mv["AUC"] > best_val:
            best_val, best_test, best_ep, wait = mv["AUC"], mt, epoch, 0
        else:
            wait += 1
        print(f"[fold {fold}] ep {epoch:3d} | loss {tot:6.1f} | "
              f"val AUC {mv['AUC']:.4f} | test AUC {mt['AUC']:.4f} AUPR {mt['AUPR']:.4f} "
              f"F1 {mt['F1']:.4f} MCC {mt['MCC']:.4f} relACC {mt.get('relACC',0):.4f}"
              f"{'  (best)' if wait == 0 else f'  (wait {wait}/{patience})' if patience else ''}",
              flush=True)
        if patience and wait >= patience:
            print(f"[fold {fold}] 早停于 ep{epoch}（验证AUC {patience}轮无提升，最佳在 ep{best_ep}）", flush=True)
            break
    print(f"[fold {fold}] 选中 ep{best_ep}(验证AUC最高) → 测试: "
          + " ".join(f"{k} {v:.4f}" for k, v in best_test.items()), flush=True)
    return best_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--folds", type=int, default=C.K_FOLD)
    ap.add_argument("--single", action="store_true", help="只跑一次 80/20 划分（看逐轮收敛）")
    ap.add_argument("--patience", type=int, default=20,
                    help="早停耐心值：验证AUC连续N轮无提升就停；设 0 关闭早停")
    ap.add_argument("--ablation",
                    choices=["none", "no_film", "no_hyper", "gcn", "no_relhead", "no_graph"],
                    default="none",
                    help="消融: none=完整 / no_film=去调控 / no_hyper=去超图 / "
                         "gcn=HGT换普通GCN / no_relhead=去关系条件头 / no_graph=去整个图分支")
    ap.add_argument("--neg", choices=["random", "degree", "rf"], default="random",
                    help="训练负样本: random=随机 / degree=度数匹配硬负 / rf=随机森林挑可靠负(DMGAT式)")
    ap.add_argument("--evalneg", choices=["random", "degree"], default="degree",
                    help="评估负样本(仅 --neg rf 时生效, 保持诚实且与baseline可比): 默认 degree 硬负")
    ap.add_argument("--coldstart", choices=["off", "ncRNA", "drug"], default="off",
                    help="冷启动: 测试集放训练没见过的 ncRNA 或 drug(考真泛化)")
    ap.add_argument("--tgtfeat", choices=["real", "learnable"], default=C.TGT_FEAT_MODE,
                    help="靶点特征: real=ESM蛋白向量⊕角色(做实) / learnable=旧版可学习ID向量(对照)")
    ap.add_argument("--arch", choices=["orig", "v2"], default="orig",
                    help="架构: orig=原版(HGT+超图+FiLM) / v2=改进版(砍图与FiLM, 药物加ECFP指纹+注意力池化)")
    args = ap.parse_args()

    utils.set_seed(C.SEED)
    device = C.DEVICE
    print(f"加载数据/特征 ... (ablation={args.ablation}, neg={args.neg}, "
          f"coldstart={args.coldstart}, tgtfeat={args.tgtfeat})")
    data = load_all()
    tgtmode = args.tgtfeat
    if tgtmode == "real" and data.tgt_feat is None:
        print("  [提示] 未找到靶点真特征(先跑 build_target_features.py)，自动降级为 learnable")
        tgtmode = "learnable"
    print(f"  ncRNA={data.N_rna} drug={data.N_drug} 正样本={len(data.pos.r)} "
          f"靶点数={data.num_targets} 靶点特征={tgtmode}"
          + (f"(d={data.d_tgt_feat})" if data.tgt_feat is not None else ""))

    rng = __import__("random").Random(C.SEED)
    # 评估负样本: random/degree 时同训练(向后兼容); rf 时用独立的 --evalneg(默认degree, 诚实可比)
    eval_neg_mode = args.evalneg if args.neg == "rf" else args.neg
    need_degree = (args.neg == "degree") or (eval_neg_mode == "degree")
    degree_pools = utils.build_neg_pools(data.pos.r, data.pos.d) if need_degree else None
    if args.neg == "rf":
        print(f"  负样本: 训练=RF可靠负(pool_mult={os.environ.get('NCDRUG_RF_POOL_MULT',3)}) / 评估={eval_neg_mode}")

    if args.coldstart != "off":                       # 冷启动：按节点留出（单次划分）
        tr, te = utils.coldstart_split(data.pos.r, data.pos.d, data.N_rna, data.N_drug,
                                       0.2, C.SEED, by=args.coldstart)
        print(f"冷启动({args.coldstart}): 训练对={int(tr.sum())} 测试对(含新节点)={int(te.sum())}")
        run_fold(0, data, tr, te, device, args.epochs, rng, args.ablation, args.patience,
                 args.neg, eval_neg_mode, degree_pools, tgtmode, args.arch)
        return

    if args.single:                                   # 单次 80/20，逐轮打印全指标
        tr, te = utils.split_positives(len(data.pos.r), 0.2, C.SEED)
        run_fold(0, data, tr, te, device, args.epochs, rng, args.ablation, args.patience,
                 args.neg, eval_neg_mode, degree_pools, tgtmode, args.arch)
        return

    splits = utils.kfold_split(len(data.pos.r), args.folds, C.SEED)
    results = []
    for f, (train_mask, test_mask) in enumerate(splits):
        print(f"===== 第 {f+1}/{args.folds} 折 =====", flush=True)
        results.append(run_fold(f, data, train_mask, test_mask, device,
                                args.epochs, rng, args.ablation, args.patience,
                                args.neg, eval_neg_mode, degree_pools, tgtmode, args.arch))

    # 汇总 k 折均值 ± 标准差（分类型AUC可能某折缺，用出现过的折平均）
    print(f"\n===== {args.folds} 折结果 (ablation={args.ablation}, neg={args.neg}, evalneg={eval_neg_mode}) 均值 ± 标准差 =====")
    allkeys = []
    for r in results:
        for k in r:
            if k not in allkeys: allkeys.append(k)
    for key in allkeys:
        vals = np.array([r[key] for r in results if key in r])
        print(f"  {key:12s} = {vals.mean():.4f} ± {vals.std():.4f}  (n折={len(vals)})")


if __name__ == "__main__":
    main()

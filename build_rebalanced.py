# -*- coding: utf-8 -*-
"""Path B 第一步：重平衡基准的参数体检。
看不同(枢纽药封顶 cap, ncRNA最小度数 mindeg)下: 关联数/ncRNA数/药数 + 度数基线AUC(越低越好=捷径被堵)。
帮用户拍板阈值。--build cap mindeg 则真正产出子集。
用法: python build_rebalanced.py            # 只看各方案对比
      python build_rebalanced.py --build 300 2   # 按 cap=300 mindeg=2 产出
"""
import os, sys, random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from sklearn.metrics import roc_auc_score


def rebalance(pos, cap, mindeg, seed=2024):
    """封顶每药最多cap条(随机下采样) + 只留度数>=mindeg的ncRNA(反复直到稳定)。"""
    rng = np.random.RandomState(seed)
    df = pos.copy()
    # 枢纽药封顶
    if cap and cap > 0:
        parts = []
        for d, g in df.groupby("Drug_Name"):
            parts.append(g.sample(min(len(g), cap), random_state=seed) if len(g) > cap else g)
        df = pd.concat(parts)
    # 迭代去低度数ncRNA(去掉后度数会变, 循环到稳定)
    while mindeg and mindeg > 1:
        vc = df["ncRNA_Name"].value_counts()
        keep = vc[vc >= mindeg].index
        if len(keep) == df["ncRNA_Name"].nunique():
            break
        df = df[df["ncRNA_Name"].isin(keep)]
    return df.reset_index(drop=True)


def degauc(df, seed=0):
    pr = df["r"].values; pd_ = df["d"].values
    Nr, Nd = pr.max() + 1, pd_.max() + 1
    dr = np.bincount(pr, minlength=Nr).astype(float); dd = np.bincount(pd_, minlength=Nd).astype(float)
    pos_set = set(zip(pr.tolist(), pd_.tolist()))
    rng = random.Random(seed); nr, nd = [], []
    while len(nr) < len(pr):
        a = pr[rng.randrange(len(pr))]; b = pd_[rng.randrange(len(pd_))]
        if (a, b) not in pos_set: nr.append(a); nd.append(b)
    ps = dr[pr] * dd[pd_]; ns = dr[np.array(nr)] * dd[np.array(nd)]
    y = np.r_[np.ones(len(ps)), np.zeros(len(ns))]
    return roc_auc_score(y, np.r_[ps, ns])


def main():
    m = pd.read_csv(C.EDGE_CSV)
    pos = m[m["rel"].isin(["resistant", "sensitive"])][["ncRNA_Name", "ncRNA_Type", "Drug_Name", "rel"]].drop_duplicates()

    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        cap, mindeg = int(sys.argv[2]), int(sys.argv[3])
        df = rebalance(pos, cap, mindeg)
        out = os.path.join(C.DATA_DIR, "merged_rebalanced.csv")
        df.to_csv(out, index=False)
        print(f"产出 cap={cap} mindeg={mindeg}: {len(df)}关联 {df['ncRNA_Name'].nunique()}ncRNA {df['Drug_Name'].nunique()}药 -> {out}")
        return

    print(f"{'方案':22s}{'关联':>8s}{'ncRNA':>8s}{'药':>6s}{'Cisplatin占比':>14s}{'度数基线AUC':>14s}")
    for cap, mindeg in [(0, 1), (0, 2), (0, 3), (500, 2), (300, 2), (300, 3), (200, 3), (150, 3)]:
        df = rebalance(pos, cap, mindeg)
        r2i = {n: i for i, n in enumerate(sorted(df["ncRNA_Name"].unique()))}
        d2i = {n: i for i, n in enumerate(sorted(df["Drug_Name"].unique()))}
        dd = pd.DataFrame({"r": df["ncRNA_Name"].map(r2i), "d": df["Drug_Name"].map(d2i)})
        cis = (df["Drug_Name"] == "Cisplatin").mean() * 100
        auc = degauc(dd)
        tag = f"cap={cap or '∞'},mindeg={mindeg}"
        print(f"{tag:22s}{len(df):>8d}{df['ncRNA_Name'].nunique():>8d}{df['Drug_Name'].nunique():>6d}{cis:>13.1f}%{auc:>14.4f}")
    print("\n度数基线AUC 越低=度数捷径被堵越狠(原始~0.98)。挑一个'关联够多 + 基线够低'的方案。")


if __name__ == "__main__":
    main()

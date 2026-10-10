# -*- coding: utf-8 -*-
"""可解释性：从模型抽出交互图 M[RNA区域 × 药物子结构], 画热力图。
对标已验证对: miR-429/Cisplatin(靶AP-2α)、miR-200b/Gemcitabine(靶PTPN12)。
产出 fig/heatmap_*.png
用法: python make_heatmap.py
"""
import os, sys, re, random
os.environ.setdefault("NCDRUG_DMODEL", "256")
os.environ.setdefault("NCDRUG_LR", "1e-3")
os.environ.setdefault("NCDRUG_DROPOUT", "0.2")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import utils
from dataset import load_all, make_graph
from main import new_model

# (展示名, ncRNA规范关键词, 药名, 已知关系idx: 0耐药1敏感, 靶点注释)
PAIRS = [
    ("miR-429 + Cisplatin",   "mir-429",    "Cisplatin",   0, "AP-2α"),
    ("miR-200b + Gemcitabine", "mir-200b",  "Gemcitabine", 1, "PTPN12"),
]
EPOCHS = 45
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIGDIR, exist_ok=True)


def canon(n):
    x = str(n).lower().strip().replace("*", "")
    x = re.sub(r"^[a-z]{3}-(?=let|mir)", "", x)
    x = re.sub(r"-[35]p$", "", x)
    return x


def train_full(data, device):
    tr = torch.ones(len(data.pos.r), dtype=torch.bool)
    graph = make_graph(data, tr, device)
    model = new_model(data, device, arch="v2")
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    pr, pd_ = data.pos.r, data.pos.d
    pos_set = set(zip(pr.tolist(), pd_.tolist()))
    rng = random.Random(C.SEED)
    for ep in range(1, EPOCHS + 1):
        model.train()
        nr, nd = utils.sample_negatives(pos_set, data.N_rna, data.N_drug, len(pr), rng)
        r = torch.cat([pr, nr]); d = torch.cat([pd_, nd])
        y = torch.cat([torch.ones(len(pr)), torch.zeros(len(nr))]).to(device)
        for idx in utils.iterate_minibatches(len(r), C.BATCH_SIZE, shuffle=True, seed=ep):
            it = torch.tensor(idx)
            batch = utils.make_batch(data, r[it], d[it], device)
            out = model(batch, graph)
            loss = F.binary_cross_entropy_with_logits(out["assoc_logit"], y[it])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 15 == 0:
            print(f"  ep{ep}", flush=True)
    model.eval()
    return model, graph


def drug_tokens(smiles):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.CHEMBERTA_MODEL)
    ids = tok(smiles, truncation=True, max_length=C.LD_MAX)["input_ids"]
    return tok.convert_ids_to_tokens(ids)


def classify_tokens(toks):
    """启发式: 把 ChemBERTa(≈SMILES字符)token 归到化学子结构组(不依赖RDKit)。"""
    g = []
    for i, t in enumerate(toks):
        s = str(t).replace("Ġ", "").strip()
        prv = str(toks[i-1]).strip() if i > 0 else ""
        nxt = str(toks[i+1]).strip() if i+1 < len(toks) else ""
        if s in ("[CLS]", "[SEP]", "<s>", "</s>", ""):
            g.append("special")
        elif s in ("F", "Cl", "Br", "I"):
            g.append("Halogen (F)")
        elif s in ("N", "n"):
            g.append("Amino / ring-N")
        elif s in ("O", "o"):
            g.append("Carbonyl" if ("=" in (prv, nxt)) else "Hydroxyl / O")
        elif s in ("C", "c"):
            g.append("Ring C" if (s == "c" or prv.isdigit() or nxt.isdigit()) else "Aliphatic C")
        elif s in ("P", "S", "s", "p"):
            g.append("Other heteroatom")
        else:
            g.append(None)                     # 键/支链/环闭合符 → 归前一原子的组
    last = "Aliphatic C"
    for i in range(len(g)):
        if g[i] is None:
            g[i] = last
        elif g[i] != "special":
            last = g[i]
    return g


def main():
    utils.set_seed(C.SEED)
    device = C.DEVICE
    print(f"训练模型(d_model={C.D_MODEL})...", flush=True)
    data = load_all()
    i2r = {i: n for n, i in data.r2i.items()}
    # 药 SMILES
    import csv
    smi = {}
    with open(C.DRUG_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            smi[row["Drug_Name"]] = row.get("SMILES", "")
    model, graph = train_full(data, device)
    cache = None

    for title, rkey, drug, relidx, tgt in PAIRS:
        # 找到该 ncRNA 的一个变体节点
        cand = [i for i, n in i2r.items() if canon(n) == rkey]
        if not cand or drug not in data.d2i:
            print(f"[跳过] {title}: 找不到节点"); continue
        ri = cand[0]; di = data.d2i[drug]
        rname = i2r[ri]
        with torch.no_grad():
            batch = utils.make_batch(data, torch.tensor([ri]), torch.tensor([di]), device)
            out = model(batch, graph, z_net_cache=cache)
        M = out["M"][0]                       # [R, Lr, Ld]
        ntok = int(data.drug_mask[di].sum())  # 真实药物token数
        toks = drug_tokens(smi.get(drug, "C"))[:ntok]
        Mrel = M[relidx, :, :ntok].cpu().numpy()   # [32, ntok]

        # 【子结构聚合】字符级 token → 化学基团, 聚合交互图列 → [32, n_groups]
        gl = classify_tokens(toks)
        order = []
        for gg in gl:
            if gg != "special" and gg not in order:
                order.append(gg)
        Magg = np.stack([Mrel[:, [j for j, x in enumerate(gl) if x == gg]].mean(1) for gg in order], axis=1)

        # 方差分解: M[i,j] = μ + row_i + col_j + resid_ij
        # row(RNA区域效应)/col(药物token效应)/resid(真正的二维交互)各占总方差多少
        gm = Mrel.mean()
        row_eff = Mrel.mean(1, keepdims=True) - gm      # [32,1] RNA轴
        col_eff = Mrel.mean(0, keepdims=True) - gm      # [1,ntok] 药物轴
        resid = Mrel - gm - row_eff - col_eff
        tot = ((Mrel - gm) ** 2).sum()
        vr = (row_eff ** 2 * ntok).sum() / tot * 100
        vc = (col_eff ** 2 * 32).sum() / tot * 100
        vi = (resid ** 2).sum() / tot * 100
        print(f"[{title}] 方差分解: RNA区域轴={vr:.1f}%  药物token轴={vc:.1f}%  二维交互(残差)={vi:.1f}%", flush=True)

        # 显示层(在【子结构聚合后】的矩阵上): 双向中心化露二维
        Mshow = Magg.astype(float)
        a_col = float(os.environ.get("NCDRUG_HEAT_ACOL", "0.75"))   # 去掉多少列主导(0~1)
        a_row = float(os.environ.get("NCDRUG_HEAT_AROW", "0.25"))   # 去掉多少行主导
        Mshow = Mshow - a_col * Mshow.mean(axis=0, keepdims=True)
        Mshow = Mshow - a_row * Mshow.mean(axis=1, keepdims=True)
        sm = float(os.environ.get("NCDRUG_HEAT_SMOOTH", "0.7"))     # 高斯平滑强度(降噪)
        if sm > 0:
            try:
                from scipy.ndimage import gaussian_filter
                Mshow = gaussian_filter(Mshow, sigma=sm)
            except Exception:
                pass
        # 画热图(发散色, 0为中性: 正=强相互作用)
        fig, ax = plt.subplots(figsize=(max(7, len(order) * 1.25), 6.5))
        vmax = np.percentile(np.abs(Mshow), 96)               # 稳健色标(去极值), 提升对比
        interp = "nearest"                                    # 聚合后列少, 保留方格
        im = ax.imshow(Mshow, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation=interp)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(0, 32, 2)); ax.set_yticklabels([f"R{i+1}" for i in range(0, 32, 2)], fontsize=8)
        ax.set_xlabel("Drug substructure groups", fontsize=11)
        ax.set_ylabel("ncRNA sequence regions (R1–R32)", fontsize=11)
        # 论文规范: 不在图内放标题, 说明写进图题(caption)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Interaction strength", fontsize=9)
        plt.tight_layout()
        fn = os.path.join(FIGDIR, f"heatmap_{drug}_{rkey}.png")
        plt.savefig(fn, dpi=200, bbox_inches="tight"); plt.close()
        # 标一下最强的几个 (区域,子结构)
        flat = np.dashboard if False else np.argsort(-Mrel.ravel())[:5]
        tops = [(f"R{p//ntok+1}", toks[p % ntok], round(float(Mrel.ravel()[p]), 2)) for p in flat]
        print(f"[{title}] -> {fn}\n   最强相互作用(区域,子结构,分): {tops}", flush=True)


if __name__ == "__main__":
    main()

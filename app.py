# -*- coding: utf-8 -*-
"""
HISCA interactive demo (Gradio).
两个功能:
  1) 预测并解释: 选 ncRNA + 药物 -> 关联概率 + 可解释"区域x子结构"热力图
  2) 药物 -> Top-K 预测 ncRNA 排行
模型在首次启动时用当前代码训练一次并缓存到 ckpt/hisca_demo.pt, 之后直接加载。
用法: D:/anaconda/envs/pytorch_gpu/python.exe app.py
"""
import os, csv, random
import numpy as np
import torch
import torch_geometric.typing
torch_geometric.typing.WITH_TORCH_SCATTER = False
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import utils
from dataset import load_all, make_graph
from main import new_model
import make_heatmap as mh   # 复用 canon / drug_tokens / classify_tokens

DEMO_EPOCHS = int(os.environ.get("NCDRUG_DEMO_EPOCHS", 80))
CKPT = os.path.join(C.CKPT_DIR, "hisca_demo.pt")
TYPE_NAME = {0: "miRNA", 1: "lncRNA", 2: "circRNA"}

print("加载数据/特征 ...", flush=True)
DEVICE = C.DEVICE
DATA = load_all()
I2R = {i: n for n, i in DATA.r2i.items()}
SMI = {}
with open(C.DRUG_CSV, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        SMI[row["Drug_Name"]] = row.get("SMILES", "")
GRAPH = make_graph(DATA, torch.ones(len(DATA.pos.r), dtype=torch.bool), DEVICE)  # 全图(所有已知关联)


def _train_full():
    model = new_model(DATA, DEVICE, arch="v2")
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    pr, pd_ = DATA.pos.r, DATA.pos.d
    pos_set = set(zip(pr.tolist(), pd_.tolist()))
    rng = random.Random(C.SEED)
    for ep in range(1, DEMO_EPOCHS + 1):
        model.train()
        nr, nd = utils.sample_negatives(pos_set, DATA.N_rna, DATA.N_drug, len(pr), rng)
        r = torch.cat([pr, nr]); d = torch.cat([pd_, nd])
        y = torch.cat([torch.ones(len(pr)), torch.zeros(len(nr))]).to(DEVICE)
        for idx in utils.iterate_minibatches(len(r), C.BATCH_SIZE, shuffle=True, seed=ep):
            it = torch.tensor(idx)
            batch = utils.make_batch(DATA, r[it], d[it], DEVICE)
            out = model(batch, GRAPH)
            loss = F.binary_cross_entropy_with_logits(out["assoc_logit"], y[it])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0:
            print(f"  demo train ep {ep}/{DEMO_EPOCHS} loss {loss.item():.3f}", flush=True)
    return model


def get_model():
    model = new_model(DATA, DEVICE, arch="v2")
    if os.path.exists(CKPT):
        try:
            model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
            print("已加载缓存模型", CKPT, flush=True); return model
        except Exception as e:
            print("缓存模型加载失败, 重新训练:", str(e)[:100], flush=True)
    print(f"训练 demo 模型 ({DEMO_EPOCHS} epoch) ...", flush=True)
    model = _train_full(); model.eval()
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CKPT); print("已保存", CKPT, flush=True)
    return model


MODEL = get_model()


@torch.no_grad()
def _score(ri, di):
    batch = utils.make_batch(DATA, torch.tensor([ri]), torch.tensor([di]), DEVICE)
    out = MODEL(batch, GRAPH)
    return float(torch.sigmoid(out["assoc_logit"])[0]), out


@torch.no_grad()
def predict_pair(rname, drug):
    if rname not in DATA.r2i or drug not in DATA.d2i:
        return "请选择有效的 ncRNA 和药物。", None
    ri, di = DATA.r2i[rname], DATA.d2i[drug]
    prob, out = _score(ri, di)
    fig = _heatmap_fig(out, di, rname, drug)
    txt = f"**{rname}**（{TYPE_NAME.get(int(DATA.rna_type[ri]),'?')}） × **{drug}**\n\n预测关联概率 = **{prob:.3f}**"
    return txt, fig


def _heatmap_fig(out, di, rname, drug):
    try:
        M = out["M"][0]
        ntok = int(DATA.drug_mask[di].sum())
        toks = mh.drug_tokens(SMI.get(drug, "C"))[:ntok]
        Mrel = M[0, :, :ntok].cpu().numpy()            # relation 0; [32, ntok]
        gl = mh.classify_tokens(toks)
        order = [gg for gg in dict.fromkeys(gl) if gg != "special"]
        if not order:
            return None
        Magg = np.stack([Mrel[:, [j for j, x in enumerate(gl) if x == gg]].mean(1) for gg in order], axis=1)
        # 完全双向去均值: 去掉药物子结构主效应(列)与 RNA 区域主效应(行),
        # 只留二维交互残差 -> 任意药对都自然居中于 0, 发散色图红蓝兼有(不再全红)
        Mshow = Magg.astype(float)
        Mshow = Mshow - Mshow.mean(axis=0, keepdims=True)
        Mshow = Mshow - Mshow.mean(axis=1, keepdims=True)
        try:
            from scipy.ndimage import gaussian_filter
            Mshow = gaussian_filter(Mshow, sigma=0.7)
        except Exception:
            pass
        # 归一化到 [-1,1] 相对强度(避免色标出现 1e-20 这种难看的指数)
        scale = np.percentile(np.abs(Mshow), 96) or 1e-12
        Mshow = np.clip(Mshow / scale, -1.0, 1.0)
        fig, ax = plt.subplots(figsize=(max(5, len(order) * 1.1), 5.2))
        im = ax.imshow(Mshow, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(0, 32, 2)); ax.set_yticklabels([f"R{i+1}" for i in range(0, 32, 2)], fontsize=8)
        ax.set_xlabel("Drug substructure groups"); ax.set_ylabel("ncRNA sequence regions (R1–R32)")
        ax.set_title(f"{rname} × {drug}", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Interaction strength (relative)", fontsize=9)
        plt.tight_layout()
        return fig
    except Exception as e:
        print("heatmap err:", str(e)[:120], flush=True)
        return None


@torch.no_grad()
def top_ncrna(drug, k, rtype):
    if drug not in DATA.d2i:
        return None
    di = DATA.d2i[drug]
    N = DATA.N_rna
    sc = np.zeros(N)
    r_all = torch.arange(N)
    for b in range(0, N, C.BATCH_SIZE):
        r = r_all[b:b + C.BATCH_SIZE]
        d = torch.full((len(r),), di, dtype=torch.long)
        batch = utils.make_batch(DATA, r, d, DEVICE)
        out = MODEL(batch, GRAPH)
        sc[b:b + len(r)] = torch.sigmoid(out["assoc_logit"]).cpu().numpy()
    tfilter = {"全部": None, "miRNA": 0, "lncRNA": 1, "circRNA": 2}[rtype]
    order = np.argsort(-sc)
    rows, seen = [], set()
    for i in order:
        ti = int(DATA.rna_type[i])
        if tfilter is not None and ti != tfilter:
            continue
        name = I2R[int(i)]
        if name in seen:
            continue
        seen.add(name)
        rows.append([len(rows) + 1, name, TYPE_NAME.get(ti, "?"), round(float(sc[i]), 4)])
        if len(rows) >= int(k):
            break
    return rows


RNAS = sorted(DATA.r2i.keys())
DRUGS = sorted(DATA.d2i.keys())

def build_demo():
    import gradio as gr
    with gr.Blocks(title="HISCA ncRNA–drug predictor") as demo:
        gr.Markdown("# HISCA\nInterpretable ncRNA–drug association prediction. "
                    "A single model unifies miRNA / lncRNA / circRNA and explains each prediction "
                    "with a region × substructure interaction map.")
        with gr.Tab("Predict & explain"):
            with gr.Row():
                rna_in = gr.Dropdown(RNAS, label="ncRNA", value=RNAS[0] if RNAS else None, filterable=True)
                drug_in = gr.Dropdown(DRUGS, label="Drug", value=DRUGS[0] if DRUGS else None, filterable=True)
            btn = gr.Button("Predict", variant="primary")
            out_txt = gr.Markdown()
            out_img = gr.Plot(label="Region × substructure interaction map")
            btn.click(predict_pair, [rna_in, drug_in], [out_txt, out_img])
        with gr.Tab("Drug → top ncRNAs"):
            with gr.Row():
                drug2 = gr.Dropdown(DRUGS, label="Drug", value=DRUGS[0] if DRUGS else None, filterable=True)
                k_in = gr.Slider(5, 30, value=10, step=1, label="Top-K")
                type_in = gr.Dropdown(["全部", "miRNA", "lncRNA", "circRNA"], value="全部", label="ncRNA type")
            btn2 = gr.Button("Rank", variant="primary")
            tbl = gr.Dataframe(headers=["Rank", "ncRNA", "Type", "Probability"], label="Top predicted ncRNAs")
            btn2.click(top_ncrna, [drug2, k_in, type_in], [tbl])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=int(os.environ.get("NCDRUG_PORT", 7860)))

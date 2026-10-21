

import os, csv, base64
import sys

_MODEL_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
if os.path.isdir(_MODEL_SOURCE):
    sys.path.insert(0, _MODEL_SOURCE)

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime", "gradio"))
import numpy as np
import torch
import torch_geometric.typing
torch_geometric.typing.WITH_TORCH_SCATTER = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import utils
from dataset import load_all, make_graph
from main import new_model
import interpretation as mh
from check_deployment import validate_runtime

CKPT = os.environ.get("NCDRUG_DEMO_CKPT", os.path.join(C.CKPT_DIR, "hisca_demo.pt"))
validate_runtime(data_dir=C.DATA_DIR, feat_dir=C.FEAT_DIR, checkpoint=CKPT, edge_csv=C.EDGE_CSV)
TYPE_NAME = {0: "miRNA", 1: "lncRNA", 2: "circRNA"}

print("加载数据/特征 ...", flush=True)
DEVICE = C.DEVICE
DATA = load_all()
I2R = {i: n for n, i in DATA.r2i.items()}
SMI = {}
with open(C.DRUG_CSV, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        SMI[row["Drug_Name"]] = row.get("SMILES", "")
GRAPH = make_graph(DATA, torch.ones(len(DATA.pos.r), dtype=torch.bool), DEVICE)


def get_model():
    if not os.path.isfile(CKPT):
        raise RuntimeError("Missing model checkpoint: " + CKPT)
    model = new_model(DATA, DEVICE, arch="v2")
    try:
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    except RuntimeError as exc:
        raise RuntimeError("Checkpoint does not match the features/model configuration. "
                           "Use matching data, features, and architecture settings; "
                           "see docs/INSTALL_zh.md. " + str(exc)) from exc
    model.eval()
    print("Loaded trained model: " + CKPT, flush=True)
    return model


MODEL = get_model()
with torch.no_grad():
    GRAPH_CACHE = MODEL.compute_hgt(GRAPH) if getattr(MODEL, "use_hgt", False) else None


@torch.no_grad()
def _score(ri, di):
    batch = utils.make_batch(DATA, torch.tensor([ri]), torch.tensor([di]), DEVICE)
    out = MODEL(batch, GRAPH, z_net_cache=GRAPH_CACHE)
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
        Mrel = M[0, :, :ntok].cpu().numpy()
        gl = mh.classify_tokens(toks)
        order = [gg for gg in dict.fromkeys(gl) if gg != "special"]
        if not order:
            return None
        Magg = np.stack([Mrel[:, [j for j, x in enumerate(gl) if x == gg]].mean(1) for gg in order], axis=1)


        Mshow = Magg.astype(float)
        Mshow = Mshow - Mshow.mean(axis=0, keepdims=True)
        Mshow = Mshow - Mshow.mean(axis=1, keepdims=True)
        try:
            from scipy.ndimage import gaussian_filter
            Mshow = gaussian_filter(Mshow, sigma=0.7)
        except Exception:
            pass

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
    for b in range(0, N, min(C.BATCH_SIZE, 128)):
        r = r_all[b:b + min(C.BATCH_SIZE, 128)]
        d = torch.full((len(r),), di, dtype=torch.long)
        batch = utils.make_batch(DATA, r, d, DEVICE)
        out = MODEL(batch, GRAPH, z_net_cache=GRAPH_CACHE)
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

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _data_uri(name):
    p = os.path.join(_ASSETS, name)
    try:
        with open(p, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def home_html():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "home.html"), encoding="utf-8") as f:
        from urllib.parse import quote
        image_url = "/file=" + quote(os.path.join(_ASSETS, "architecture_latest.png").replace("\\", "/"), safe="/:")
        return f.read().replace('href="__ARCH__"', 'href="' + image_url + '"').replace("__ARCH__", _data_uri("architecture_latest.png")).replace("__COVER__", _data_uri("cover_rna_drug.png"))


def build_demo():
    import gradio as gr
    navigation_js = """
    () => {
      const switchTab = (name) => {
        const labels = {home: 'Home', predict: 'Predict & explain', rank: 'Drug → top ncRNAs'};
        const tab = Array.from(document.querySelectorAll('#hisca-tabs button[role="tab"]'))
          .find(el => el.textContent.trim() === labels[name]);
        if (tab) {
          tab.click();
          document.getElementById('hisca-tabs').scrollIntoView({behavior: 'smooth', block: 'start'});
        }
      };
      if (!window.hiscaNavigationBound) {
        document.addEventListener('click', (event) => {
          const link = event.target.closest('[data-hisca-tab]');
          if (link) {
            event.preventDefault();
            switchTab(link.dataset.hiscaTab);
          }
        });
        window.hiscaNavigationBound = true;
      }
    }
    """
    css = """
    .gradio-container {max-width:1180px !important; margin:auto !important; font-family:Arial,sans-serif;}
    #hisca-tabs > .tab-nav {background:#263f57; padding:0 12px; border:0;}
    #hisca-tabs > .tab-nav button {color:#dae4ed; border:0; border-radius:0; padding:14px 22px;}
    #hisca-tabs > .tab-nav button.selected {background:#36566f; color:white; border-bottom:3px solid #cba56a;}
    #hisca-home .tool-action {display:inline-block;margin-top:12px;padding:7px 14px;background:#294b66;color:white;font-size:12px;text-decoration:none;}
    #hisca-home .tool-action:hover {background:#3b6686;}
    #hisca-home .cover-cta {display:inline-block;margin-top:15px;padding:7px 16px;background:#294b66;color:white;font-size:12px;text-decoration:none;}
    """
    theme = gr.themes.Base(primary_hue="slate", font=["Arial", "sans-serif"])
    with gr.Blocks(title="HISCA | ncRNA–drug prediction", theme=theme, css=css,
                   js=navigation_js, analytics_enabled=False) as demo:
        with gr.Tabs(selected="home", elem_id="hisca-tabs") as tabs:
            with gr.Tab("Home", id="home"):
                gr.HTML(home_html())
                with gr.Row():
                    to_predict = gr.Button("Open prediction / 进入配对预测", variant="primary")
                    to_rank = gr.Button("Open ranking / 进入候选排序")
            with gr.Tab("Predict & explain", id="predict"):
                gr.Markdown("## ncRNA–drug association prediction\n选择已有 ncRNA 和药物，点击 Predict 查看关联分数和交互热图。")
                with gr.Row():
                    rna_in = gr.Dropdown(RNAS, label="ncRNA", value=RNAS[0], filterable=True)
                    drug_in = gr.Dropdown(DRUGS, label="Drug", value=DRUGS[0], filterable=True)
                with gr.Row():
                    predict_btn = gr.Button("Predict / 开始预测", variant="primary")
                    back_predict = gr.Button("Back to Home / 返回首页")
                result = gr.Markdown()
                heatmap = gr.Plot(label="Region × substructure interaction map")
                gr.Markdown("研究用途：分数为模型输出，交互热图表示相对模式，不代表已经验证的生物学机制。")
                predict_btn.click(predict_pair, [rna_in, drug_in], [result, heatmap],
                                  api_name="predict_pair", concurrency_limit=1, concurrency_id="model")
            with gr.Tab("Drug → top ncRNAs", id="rank"):
                gr.Markdown("## Drug-based candidate ranking\n选择药物、候选数量及 ncRNA 类型，点击 Rank 获取真实模型排序。")
                with gr.Row():
                    drug_rank = gr.Dropdown(DRUGS, label="Drug", value=DRUGS[0], filterable=True)
                    k = gr.Slider(5, 30, value=10, step=1, label="Top-K")
                    rtype = gr.Dropdown(["全部", "miRNA", "lncRNA", "circRNA"], value="全部", label="ncRNA type")
                with gr.Row():
                    rank_btn = gr.Button("Rank / 开始排序", variant="primary")
                    back_rank = gr.Button("Back to Home / 返回首页")
                table = gr.Dataframe(headers=["Rank", "ncRNA", "Type", "Probability"],
                                     datatype=["number", "str", "str", "number"],
                                     interactive=False, value=[], label="Top predicted ncRNAs")
                rank_btn.click(top_ncrna, [drug_rank, k, rtype], table,
                               api_name="top_ncrna", concurrency_limit=1, concurrency_id="model")
        to_predict.click(lambda: gr.Tabs(selected="predict"), outputs=tabs, queue=False, api_name=False)
        to_rank.click(lambda: gr.Tabs(selected="rank"), outputs=tabs, queue=False, api_name=False)
        back_predict.click(lambda: gr.Tabs(selected="home"), outputs=tabs, queue=False, api_name=False)
        back_rank.click(lambda: gr.Tabs(selected="home"), outputs=tabs, queue=False, api_name=False)
    return demo


if __name__ == "__main__":
    build_demo().queue(max_size=16).launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("NCDRUG_PORT", 7860)),
        show_error=True,
        allowed_paths=[_ASSETS],
    )
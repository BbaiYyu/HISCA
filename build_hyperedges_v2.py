

import os, sys, json, csv, re
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch_geometric.typing
torch_geometric.typing.WITH_TORCH_SCATTER = False
import config as C
from dataset import load_all

R2_MAX = int(os.environ.get("HYPER_R2_MAX", 10))
X1_MAX = int(os.environ.get("HYPER_X1_MAX", 20))
CHEMBL_CACHE = os.environ.get("NCDRUG_CHEMBL_CACHE", os.path.join(C.DATA_DIR, "chembl_target_cache.json"))


def canon_mir(n):
    x = str(n).lower().strip().replace("*", "")
    x = re.sub(r"^[a-z]{3}-(?=let|mir)", "", x)
    x = re.sub(r"-[35]p$", "", x)
    return x


def build():
    DATA = load_all()
    N_rna, N_drug = DATA.N_rna, DATA.N_drug
    rtype = DATA.rna_type.numpy()
    i2r = {i: n for n, i in DATA.r2i.items()}
    i2d = {i: n for n, i in DATA.d2i.items()}
    tgt_ids, tgt_mask = DATA.tgt_ids.numpy(), DATA.tgt_mask.numpy()
    G = {}


    d = defaultdict(set)
    for di in range(N_drug):
        for t, m in zip(tgt_ids[di], tgt_mask[di]):
            if m: d[int(t)].add(N_rna + di)
    G["D1"] = {k: v for k, v in d.items() if len(v) >= 2}


    smi = {}
    with open(C.DRUG_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f): smi[row["Drug_Name"]] = row.get("SMILES", "")
    d = defaultdict(set)
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        for di in range(N_drug):
            s = smi.get(i2d[di], "")
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s) if s else ""
                if sc: d[sc].add(N_rna + di)
            except Exception: pass
    except Exception as e:
        print("[D2] RDKit 不可用, 跳过:", str(e)[:60])
    G["D2"] = {k: v for k, v in d.items() if len(v) >= 2}


    seq = {}
    with open(os.path.join(C.DATA_DIR, "ncRNA.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seq[row["ncRNA_Name"]] = (row.get("Sequence", "") or "").upper().replace("T", "U")
    d = defaultdict(set)
    for ri in range(N_rna):
        if rtype[ri] != 0: continue
        s = seq.get(i2r[ri], "")
        if len(s) >= 8: d[s[1:8]].add(ri)
    G["R1"] = {k: v for k, v in d.items() if len(v) >= 2}


    nt = json.load(open(os.path.join(C.DATA_DIR, "ncrna_targets.json"), encoding="utf-8"))
    nt_canon = {canon_mir(k): set(str(x).upper() for x in v) for k, v in nt.items() if v}
    mir_gene = defaultdict(set)
    for ri in range(N_rna):
        if rtype[ri] != 0: continue
        gs = nt_canon.get(canon_mir(i2r[ri]))
        if gs:
            for g in gs: mir_gene[g].add(ri)

    G["R2"] = {g: v for g, v in mir_gene.items() if 2 <= len(v) <= R2_MAX}


    chembl = json.load(open(CHEMBL_CACHE, encoding="utf-8"))
    uni2gene = {}
    for _, pairs in chembl.items():
        for p in pairs:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                uni2gene[str(p[0]).upper()] = str(p[1]).upper()
    tgt = torch.load(os.path.join(C.FEAT_DIR, "target_feats.pt"), map_location="cpu")
    id2uni = {v: str(k).upper() for k, v in tgt["vocab"].items()}
    drug_gene = defaultdict(set)
    for di in range(N_drug):
        for t, m in zip(tgt_ids[di], tgt_mask[di]):
            if m:
                uni = id2uni.get(int(t)); g = uni2gene.get(uni) if uni else None
                if g: drug_gene[g].add(N_rna + di)

    d = {}
    for g in (set(drug_gene) & set(mir_gene)):
        if len(mir_gene[g]) <= X1_MAX:
            d[g] = drug_gene[g] | mir_gene[g]
    G["X1"] = {k: v for k, v in d.items() if len(v) >= 2}


    TYPES = ["D1", "D2", "R1", "R2", "X1"]
    rows, cols, etype = [], [], []
    he = 0
    for ti, t in enumerate(TYPES):
        for _, nodes in sorted(G[t].items(), key=lambda kv: -len(kv[1])):
            for nid in nodes:
                rows.append(int(nid)); cols.append(he)
            etype.append(ti); he += 1
    hyperedge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    return DATA, G, TYPES, hyperedge_index, edge_type


def main():
    DATA, G, TYPES, HEI, ET = build()
    N_rna, N_drug = DATA.N_rna, DATA.N_drug
    NAME = {"D1": "Drug: shared target", "D2": "Drug: shared scaffold",
            "R1": "miRNA: seed family", "R2": "miRNA: shared target gene",
            "X1": "Cross: shared-gene bridge"}

    out = os.path.join(C.DATA_DIR, "hyperedges_v2.pt")
    meta = {t: {"n_edges": len(G[t]),
                "n_nodes": len(set().union(*G[t].values())) if G[t] else 0,
                "sizes_mean": float(np.mean([len(v) for v in G[t].values()])) if G[t] else 0,
                "sizes_max": int(np.max([len(v) for v in G[t].values()])) if G[t] else 0}
            for t in TYPES}
    torch.save({"hyperedge_index": HEI, "edge_type": ET, "type_names": TYPES,
                "n_rna": N_rna, "n_drug": N_drug, "meta": meta}, out)
    tot_e = sum(m["n_edges"] for m in meta.values())
    print(f"\n已保存 -> {out}")
    print(f"总超边 {tot_e} 条, 关联(非零) {HEI.shape[1]}, 节点空间 {N_rna + N_drug}\n")
    for t in TYPES:
        m = meta[t]
        print(f"  {t} {NAME[t]:<28} 超边{m['n_edges']:>5}  覆盖{m['n_nodes']:>5}  均{m['sizes_mean']:.1f}/最大{m['sizes_max']}")


    print("\nX1 桥接示例:")
    i2r = {i: n for n, i in DATA.r2i.items()}; i2d = {i: n for n, i in DATA.d2i.items()}
    shown = 0
    for g, nodes in sorted(G["X1"].items(), key=lambda kv: -len(kv[1])):
        drugs = [i2d[n - N_rna] for n in nodes if n >= N_rna][:4]
        mirs = [i2r[n] for n in nodes if n < N_rna][:4]
        if g in ("EGFR", "TP53", "PTEN", "BCL2", "VEGFA", "KRAS", "AKT1", "MYC") or shown < 2:
            print(f"   基因 {g}: 药{drugs} + miRNA{mirs}")
            shown += 1
        if shown >= 4: break


    palette = {"D1": "#335C81", "D2": "#4A7FB0", "R1": "#5B8A72",
               "R2": "#8DB48E", "X1": "#C98C4A"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    xs = range(len(TYPES)); cols = [palette[t] for t in TYPES]


    ne = [meta[t]["n_edges"] for t in TYPES]
    b = axes[0].bar(xs, ne, color=cols)
    for bi, v in zip(b, ne): axes[0].text(bi.get_x()+bi.get_width()/2, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(list(xs)); axes[0].set_xticklabels(TYPES, fontsize=10)
    axes[0].set_ylabel("Number of hyperedges", fontsize=11)
    axes[0].set_title("(a) Hyperedge count by type", fontsize=11)


    nn = [meta[t]["n_nodes"] for t in TYPES]
    b = axes[1].bar(xs, nn, color=cols)
    for bi, v in zip(b, nn): axes[1].text(bi.get_x()+bi.get_width()/2, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(list(xs)); axes[1].set_xticklabels(TYPES, fontsize=10)
    axes[1].set_ylabel("Nodes covered", fontsize=11)
    axes[1].set_title("(b) Node coverage by type", fontsize=11)


    sizes = [[len(v) for v in G[t].values()] or [0] for t in TYPES]
    bp = axes[2].boxplot(sizes, patch_artist=True, showfliers=True,
                         flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, t in zip(bp["boxes"], TYPES): patch.set_facecolor(palette[t]); patch.set_alpha(0.8)
    for med in bp["medians"]: med.set_color("black")
    axes[2].set_yscale("log")
    axes[2].set_xticks(range(1, len(TYPES)+1)); axes[2].set_xticklabels(TYPES, fontsize=9)
    axes[2].set_ylabel("Hyperedge size (nodes, log)", fontsize=11)
    axes[2].set_title("(c) Hyperedge size distribution", fontsize=11)
    axes[2].grid(axis="y", ls=":", alpha=0.4)

    fig.suptitle(f"High-order hyperedges for HISCA  (total {sum(ne)} hyperedges over {N_rna+N_drug} nodes)", fontsize=12)
    fig.tight_layout()
    figout = os.environ.get("NCDRUG_HYPER_FIGURE", os.path.join(os.path.dirname(C.DATA_DIR), "fig", "hyperedge_stats.png"))
    figout = os.path.abspath(figout)
    os.makedirs(os.path.dirname(figout), exist_ok=True)
    fig.savefig(figout, dpi=200, bbox_inches="tight")
    print(f"\n统计图 -> {figout}")


if __name__ == "__main__":
    main()

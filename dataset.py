# -*- coding: utf-8 -*-
"""
数据装配：读阶段1 特征 + merged_final 关联 → 建异质图 + 超边 + 正样本。
关键：assoc（ncRNA—drug 关联）边只用【训练集】构图，避免测试边泄漏；
      相似边、共享靶点超边来自特征/靶点本身，不含标签，可一次建好。
"""
import os, csv, sys
from types import SimpleNamespace
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

csv.field_size_limit(10 ** 8)

NODE_TYPES = ["ncRNA", "drug"]
EDGE_TYPES = []
for _rel in C.RELATIONS:                              # 关系边由 config.RELATIONS 推导（现为2关系）
    EDGE_TYPES += [("ncRNA", _rel, "drug"), ("drug", "rev_" + _rel, "ncRNA")]
EDGE_TYPES += [("ncRNA", "sim", "ncRNA"), ("drug", "sim", "drug")]
METADATA = (NODE_TYPES, EDGE_TYPES)
REL2IDX = {r: i for i, r in enumerate(C.RELATIONS)}   # resistant/sensitive -> 0/1


def _read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def topk_cosine_edges(feat, k):
    """[N,d] 特征 → 每个节点连 top-k 余弦相似邻居，返回 edge_index [2,E]（无向、含双向）。"""
    x = torch.nn.functional.normalize(feat, dim=1)
    N = x.shape[0]
    src, dst = [], []
    step = 2048
    for b in range(0, N, step):
        sim = x[b:b + step] @ x.t()                      # [chunk,N]
        sim[torch.arange(sim.shape[0]), torch.arange(b, b + sim.shape[0])] = -2  # 去自环
        idx = sim.topk(min(k, N - 1), dim=1).indices     # [chunk,k]
        for i in range(idx.shape[0]):
            u = b + i
            for v in idx[i].tolist():
                src += [u, v]; dst += [v, u]              # 双向
    return torch.tensor([src, dst], dtype=torch.long)


def load_all():
    """一次性装配：特征 + 节点映射 + 正样本 + 相似边 + 靶点分组。"""
    F = C.FEAT_DIR
    rna = torch.load(os.path.join(F, "rna_feats.pt"))
    drug = torch.load(os.path.join(F, "drug_feats.pt"))
    tgt = torch.load(os.path.join(F, "target_feats.pt"))

    rna_feat = rna["feat"]                    # [N_rna,K,d_rna]
    drug_feat, drug_mask = drug["feat"], drug["mask"]   # [N_drug,Ld,d_drug]
    r2i, d2i = rna["name2idx"], drug["name2idx"]
    N_rna, N_drug = len(r2i), len(d2i)

    # ---- RNA k-mer 区域特征(交互图可解释性用, 位置有区分度)：[N,32,64], 缺则 None
    rna_kmer = None
    _kp = os.path.join(F, "rna_kmer.pt")
    if os.path.exists(_kp):
        kk = torch.load(_kp)
        assert kk["name2idx"] == r2i, "rna_kmer.pt 顺序与 rna_feats.pt 不一致，请重跑 build_kmer_feats.py"
        rna_kmer = kk["feat"]

    # ---- ncRNA 类型索引(创新②类型调制用)：miRNA/lncRNA/circRNA → 0/1/2, 其它→0
    TYPE2I = {"miRNA": 0, "lncRNA": 1, "circRNA": 2}
    name2type = {}
    for row in _read_csv(C.EDGE_CSV):
        name2type.setdefault(row["ncRNA_Name"], row.get("ncRNA_Type", "miRNA"))
    rna_type = torch.zeros(N_rna, dtype=torch.long)
    for name, i in r2i.items():
        rna_type[i] = TYPE2I.get(name2type.get(name, "miRNA"), 0)

    # 节点初始特征（阶段3 图用）：区域/ token 平均池化
    rna_node = rna_feat.mean(dim=1)                                   # [N_rna,d_rna]
    dm = drug_mask.unsqueeze(-1).float()
    drug_node = (drug_feat * dm).sum(1) / dm.sum(1).clamp(min=1)      # [N_drug,d_drug]

    # ---- 正样本：清洗后每对一行，rel ∈ resistant/sensitive/conflict
    pos_r, pos_d, pos_rel = [], [], []
    for e in _read_csv(C.EDGE_CSV):
        rn, dn, rel = e["ncRNA_Name"], e["Drug_Name"], e["rel"]
        if rn not in r2i or dn not in d2i:
            continue
        if rel == "conflict":                          # 冲突对按配置处理
            if C.CONFLICT_POLICY == "drop":
                continue
            ridx = -1                                  # mask: 仍是关联正样本, 但无关系标签(不进CE)
        else:
            ridx = REL2IDX[rel]
        pos_r.append(r2i[rn]); pos_d.append(d2i[dn]); pos_rel.append(ridx)
    pos = SimpleNamespace(r=torch.tensor(pos_r), d=torch.tensor(pos_d),
                          rel=torch.tensor(pos_rel))

    # ---- 相似边（一次建好，不含标签）
    # ncRNA 相似边: 优先用【可区分的 k-mer 特征】(RNA-FM池化区域雷同→相似图近乎无信息)。默认开。
    _sim_feat = (rna_kmer.mean(1) if (rna_kmer is not None and os.environ.get("NCDRUG_SIM_KMER", "1") == "1")
                 else rna_node)
    sim_rna  = topk_cosine_edges(_sim_feat, C.SIM_K)
    sim_drug = topk_cosine_edges(drug_node, C.SIM_K)

    # ---- 共享靶点分组（用于超边）：target_idx -> [drug 节点id]
    tgt_ids, tgt_mask = tgt["tgt_ids"], tgt["mask"]      # [N_drug,T]
    tid2drugs = {}
    for d in range(N_drug):
        for j in range(tgt_ids.shape[1]):
            if tgt_mask[d, j]:
                tid2drugs.setdefault(int(tgt_ids[d, j]), []).append(d)

    # ---- 靶点真特征（ESM⊕角色）：build_target_features.py 生成的 'feat' [num_tgt+1, d_feat]
    tgt_feat = tgt.get("feat")                            # 没跑过构建脚本时为 None → 模型自动降级用可学习ID向量
    d_tgt_feat = int(tgt.get("d_feat", 0)) if tgt_feat is not None else 0

    # ---- 药物分子指纹（ECFP4, build_drug_extra.py 生成）：v2 架构用, 缺则 None
    drug_ecfp = None
    _ep = os.path.join(F, "drug_extra.pt")
    if os.path.exists(_ep):
        de = torch.load(_ep)
        assert de["name2idx"] == d2i, "drug_extra.pt 与 drug_feats.pt 的药物顺序不一致，请重跑 build_drug_extra.py"
        drug_ecfp = de["ecfp"]                            # [N_drug, 2048]

    # ---- 药物原子分子图（build_drug_graphs.py 生成）：v2 架构的可训练GCN用, 缺则 None
    drug_graph = None
    _gp = os.path.join(F, "drug_graphs.pt")
    if os.path.exists(_gp):
        dg = torch.load(_gp)
        assert dg["name2idx"] == d2i, "drug_graphs.pt 与 drug_feats.pt 顺序不一致，请重跑 build_drug_graphs.py"
        # 拼成一张大图: x[所有原子,69], edge_index[2,E](偏移), batch[所有原子]
        xs, eis, bt, off = [], [], [], 0
        for gi, (x, ei) in enumerate(dg["graphs"]):
            xs.append(x); eis.append(ei + off); bt.append(torch.full((x.shape[0],), gi, dtype=torch.long))
            off += x.shape[0]
        drug_graph = {"x": torch.cat(xs), "edge_index": torch.cat(eis, dim=1),
                      "batch": torch.cat(bt), "atom_dim": dg.get("atom_dim", 69)}

    return SimpleNamespace(
        rna_feat=rna_feat, drug_feat=drug_feat, drug_mask=drug_mask,
        rna_node=rna_node, drug_node=drug_node,
        tgt_ids=tgt_ids, tgt_mask=tgt_mask, num_targets=tgt["num_targets"],
        tgt_feat=tgt_feat, d_tgt_feat=d_tgt_feat, drug_ecfp=drug_ecfp, drug_graph=drug_graph,
        rna_type=rna_type, rna_kmer=rna_kmer,
        r2i=r2i, d2i=d2i, N_rna=N_rna, N_drug=N_drug,
        pos=pos, sim_rna=sim_rna, sim_drug=sim_drug, tid2drugs=tid2drugs,
        metadata=METADATA,
    )


def build_hyperedges(data):
    """超边：① 共享靶点高阶群（组内药）② 可选·相似 ego。统一 id：drug 偏移 N_rna。
    防过拟合旋钮见 config：HYPER_GROUP_MAX 丢超大hub组 / HYPER_MAX_EDGES 只留前N条 /
    HYPER_USE_SIM_EGO 是否加相似ego超边（默认关，避免与HGT的sim边重复+爆量9000+）。"""
    N_rna = data.N_rna
    rows, cols = [], []
    he = 0
    # ① 共享靶点：按组大小过滤 + 排序 + 截前N
    groups = [drugs for drugs in data.tid2drugs.values()
              if len(drugs) >= C.HYPER_MIN_GROUP
              and (C.HYPER_GROUP_MAX <= 0 or len(drugs) <= C.HYPER_GROUP_MAX)]
    groups.sort(key=len, reverse=True)                       # 大组在前
    if C.HYPER_MAX_EDGES > 0:
        groups = groups[: C.HYPER_MAX_EDGES]                 # 只融合前N条最大的群
    for drugs in groups:
        for d in drugs:
            rows.append(N_rna + d); cols.append(he)
        he += 1
    # ② 相似 ego（每节点+其相似邻居）——默认关闭
    if C.HYPER_USE_SIM_EGO:
        def ego(edge_index, offset):
            nonlocal he
            nbr = {}
            s, t = edge_index
            for u, v in zip(s.tolist(), t.tolist()):
                nbr.setdefault(u, set()).add(v)
            for u, vs in nbr.items():
                for m in ({u} | vs):
                    rows.append(offset + m); cols.append(he)
                he += 1
        ego(data.sim_rna, 0)
        ego(data.sim_drug, N_rna)
    if he == 0:                                              # 极端配置下无超边：空占位避免报错
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def make_graph(data, train_mask, device):
    """按【训练集】正样本建 assoc 边 + 常驻相似边 + 超边，组装模型要的 graph dict。"""
    N_rna = data.N_rna
    r = data.pos.r[train_mask]; d = data.pos.d[train_mask]; rel = data.pos.rel[train_mask]

    edge_index_dict = {}
    for ri, rname in enumerate(C.RELATIONS):
        sel = rel == ri
        rr, dd = r[sel], d[sel]
        edge_index_dict[("ncRNA", rname, "drug")]     = torch.stack([rr, dd]).to(device)
        edge_index_dict[("drug", "rev_" + rname, "ncRNA")] = torch.stack([dd, rr]).to(device)
    edge_index_dict[("ncRNA", "sim", "ncRNA")] = data.sim_rna.to(device)
    edge_index_dict[("drug", "sim", "drug")]   = data.sim_drug.to(device)

    x_dict = {"ncRNA": data.rna_node.to(device), "drug": data.drug_node.to(device)}
    hyperedge_index = build_hyperedges(data).to(device)

    # 同质图边（仅供 gcn 消融用）：统一 id 空间 drug 偏移 N_rna，所有边不分类型、双向
    a_src = torch.cat([r, d + N_rna])                      # 关联边（训练集）双向
    a_dst = torch.cat([d + N_rna, r])
    s_r, s_d = data.sim_rna, data.sim_drug                 # 相似边（sim_drug 需偏移到统一空间）
    homo_src = torch.cat([a_src, s_r[0], s_d[0] + N_rna])
    homo_dst = torch.cat([a_dst, s_r[1], s_d[1] + N_rna])
    homo_edge_index = torch.stack([homo_src, homo_dst]).to(device)

    return {"x_dict": x_dict, "edge_index_dict": edge_index_dict,
            "hyperedge_index": hyperedge_index, "homo_edge_index": homo_edge_index}

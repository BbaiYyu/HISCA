

import os, csv, sys
from types import SimpleNamespace
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

csv.field_size_limit(10 ** 8)

NODE_TYPES = ["ncRNA", "drug"]
EDGE_TYPES = []
for _rel in C.RELATIONS:
    EDGE_TYPES += [("ncRNA", _rel, "drug"), ("drug", "rev_" + _rel, "ncRNA")]
EDGE_TYPES += [("ncRNA", "sim", "ncRNA"), ("drug", "sim", "drug")]
METADATA = (NODE_TYPES, EDGE_TYPES)
REL2IDX = {r: i for i, r in enumerate(C.RELATIONS)}


def _read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def topk_cosine_edges(feat, k):

    x = torch.nn.functional.normalize(feat, dim=1)
    N = x.shape[0]
    src, dst = [], []
    step = 2048
    for b in range(0, N, step):
        sim = x[b:b + step] @ x.t()
        sim[torch.arange(sim.shape[0]), torch.arange(b, b + sim.shape[0])] = -2
        idx = sim.topk(min(k, N - 1), dim=1).indices
        for i in range(idx.shape[0]):
            u = b + i
            for v in idx[i].tolist():
                src += [u, v]; dst += [v, u]
    return torch.tensor([src, dst], dtype=torch.long)


def load_all():

    F = C.FEAT_DIR
    rna = torch.load(os.path.join(F, "rna_feats.pt"))
    drug = torch.load(os.path.join(F, "drug_feats.pt"))
    tgt = torch.load(os.path.join(F, "target_feats.pt"))

    rna_feat = rna["feat"]
    drug_feat, drug_mask = drug["feat"], drug["mask"]
    r2i, d2i = rna["name2idx"], drug["name2idx"]
    N_rna, N_drug = len(r2i), len(d2i)


    rna_kmer = None
    _kp = os.path.join(F, "rna_kmer.pt")
    if os.path.exists(_kp):
        kk = torch.load(_kp)
        assert kk["name2idx"] == r2i, "rna_kmer.pt 顺序与 rna_feats.pt 不一致，请重跑 build_kmer_feats.py"
        rna_kmer = kk["feat"]


    TYPE2I = {"miRNA": 0, "lncRNA": 1, "circRNA": 2}
    name2type = {}
    for row in _read_csv(C.EDGE_CSV):
        name2type.setdefault(row["ncRNA_Name"], row.get("ncRNA_Type", "miRNA"))
    rna_type = torch.zeros(N_rna, dtype=torch.long)
    for name, i in r2i.items():
        rna_type[i] = TYPE2I.get(name2type.get(name, "miRNA"), 0)


    rna_node = rna_feat.mean(dim=1)
    dm = drug_mask.unsqueeze(-1).float()
    drug_node = (drug_feat * dm).sum(1) / dm.sum(1).clamp(min=1)


    pos_r, pos_d, pos_rel = [], [], []
    for e in _read_csv(C.EDGE_CSV):
        rn, dn, rel = e["ncRNA_Name"], e["Drug_Name"], e["rel"]
        if rn not in r2i or dn not in d2i:
            continue
        if rel == "conflict":
            if C.CONFLICT_POLICY == "drop":
                continue
            ridx = -1
        else:
            ridx = REL2IDX[rel]
        pos_r.append(r2i[rn]); pos_d.append(d2i[dn]); pos_rel.append(ridx)
    pos = SimpleNamespace(r=torch.tensor(pos_r), d=torch.tensor(pos_d),
                          rel=torch.tensor(pos_rel))


    _sim_feat = (rna_kmer.mean(1) if (rna_kmer is not None and os.environ.get("NCDRUG_SIM_KMER", "1") == "1")
                 else rna_node)
    sim_rna  = topk_cosine_edges(_sim_feat, C.SIM_K)
    sim_drug = topk_cosine_edges(drug_node, C.SIM_K)


    tgt_ids, tgt_mask = tgt["tgt_ids"], tgt["mask"]
    tid2drugs = {}
    for d in range(N_drug):
        for j in range(tgt_ids.shape[1]):
            if tgt_mask[d, j]:
                tid2drugs.setdefault(int(tgt_ids[d, j]), []).append(d)


    tgt_feat = tgt.get("feat")
    d_tgt_feat = int(tgt.get("d_feat", 0)) if tgt_feat is not None else 0


    drug_ecfp = None
    _ep = os.path.join(F, "drug_extra.pt")
    if os.path.exists(_ep):
        de = torch.load(_ep)
        assert de["name2idx"] == d2i, "drug_extra.pt 与 drug_feats.pt 的药物顺序不一致，请重跑 build_drug_extra.py"
        drug_ecfp = de["ecfp"]


    drug_graph = None
    _gp = os.path.join(F, "drug_graphs.pt")
    if os.path.exists(_gp):
        dg = torch.load(_gp)
        assert dg["name2idx"] == d2i, "drug_graphs.pt 与 drug_feats.pt 顺序不一致，请重跑 build_drug_graphs.py"

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


    N_rna = data.N_rna
    rows, cols = [], []
    he = 0

    groups = [drugs for drugs in data.tid2drugs.values()
              if len(drugs) >= C.HYPER_MIN_GROUP
              and (C.HYPER_GROUP_MAX <= 0 or len(drugs) <= C.HYPER_GROUP_MAX)]
    groups.sort(key=len, reverse=True)
    if C.HYPER_MAX_EDGES > 0:
        groups = groups[: C.HYPER_MAX_EDGES]
    for drugs in groups:
        for d in drugs:
            rows.append(N_rna + d); cols.append(he)
        he += 1

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
    if he == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def make_graph(data, train_mask, device):

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

    _hev2 = os.path.join(C.DATA_DIR, "hyperedges_v2.pt")
    if os.environ.get("NCDRUG_V2_HYPER", "0") == "1" and os.path.exists(_hev2):
        hyperedge_index = torch.load(_hev2)["hyperedge_index"].to(device)
    else:
        hyperedge_index = build_hyperedges(data).to(device)


    a_src = torch.cat([r, d + N_rna])
    a_dst = torch.cat([d + N_rna, r])
    s_r, s_d = data.sim_rna, data.sim_drug
    homo_src = torch.cat([a_src, s_r[0], s_d[0] + N_rna])
    homo_dst = torch.cat([a_dst, s_r[1], s_d[1] + N_rna])
    homo_edge_index = torch.stack([homo_src, homo_dst]).to(device)

    return {"x_dict": x_dict, "edge_index_dict": edge_index_dict,
            "hyperedge_index": hyperedge_index, "homo_edge_index": homo_edge_index}

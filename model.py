# -*- coding: utf-8 -*-
"""
完整模型：把 5 个零件按框架图串起来。

  阶段1 编码器输出（Hr / Hd / 靶点id）在外部预计算，本模型消费它们。
  阶段2  TargetRoutedFiLM → InteractionModule           (z_pair, 交互图M)  〔创新②〕
  阶段3  HGTBranch  ⊕  HyperBranch  →  门控融合          (z_net)
  阶段4  RelationCondHead                                (是否关联 + 关系类型)〔创新①〕
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (TargetRoutedFiLM, InteractionModule,
                    HGTBranch, HyperBranch, RelationCondHead,
                    GCNBranch, SharedHead, AttnPool, DrugGraphEncoder, TypeFiLM)


class TRHGNet(nn.Module):
    """
    Target-Routed Heterogeneous-Hypergraph Net（名字可改）。
    参数:
        num_targets : 蛋白靶点总数（靶点 embedding 查表大小）
        metadata    : (node_types, edge_types)，PyG 异质图元信息
        n_rna,n_drug: 图里 ncRNA / drug 节点数（用于统一 id 映射）
    """
    def __init__(self, num_targets, metadata, n_rna, n_drug,
                 d_rna_enc, d_drug_enc, d_model,
                 num_rel, film_hidden, n_hgt_layer, n_hgt_head,
                 n_hyper_layer, head_hidden, dropout=0.2, ablation="none",
                 tgt_feat=None, tgt_feat_mode="learnable",
                 arch="orig", drug_ecfp=None, drug_graph=None, rna_type=None):
        super().__init__()
        self.n_rna = n_rna
        self.d_model = d_model
        # 消融开关: none / no_film / no_hyper / gcn / no_relhead / no_graph
        self.ablation = ablation
        # 架构: orig=原版(HGT+超图+FiLM) / v2=改进版(砍图与FiLM, 药物加ECFP指纹+注意力池化)
        self.arch = arch

        # ---- 阶段1 输出的 token 级特征投到统一维度
        self.rna_in  = nn.Linear(d_rna_enc,  d_model)   # 每个核苷酸区域
        self.drug_in = nn.Linear(d_drug_enc, d_model)   # 每个药物子结构
        # ---- 靶点表示（创新②的输入）
        # real: 用真蛋白特征(ESM⊕角色)冻结做底 + 线性投影(+可学习残差); learnable: 旧版可学习ID向量
        self.tgt_mode = tgt_feat_mode if tgt_feat is not None else "learnable"
        if self.tgt_mode == "real":
            self.register_buffer("tgt_feat", tgt_feat)              # [num_targets+1, d_feat] 冻结
            self.tgt_proj = nn.Linear(tgt_feat.shape[1], d_model)   # 真特征 → 统一维
            self.tgt_res  = nn.Embedding(num_targets + 1, d_model, padding_idx=0)  # 小可学习残差
            nn.init.zeros_(self.tgt_res.weight)                    # 从0起，纯靠真特征打底
        else:
            self.tgt_emb = nn.Embedding(num_targets + 1, d_model, padding_idx=0)

        # ---- 阶段2（创新②）
        self.film     = TargetRoutedFiLM(d_model, film_hidden, dropout)
        self.interact = InteractionModule(d_model, num_rel, dropout)

        # ---- 阶段3a：HGT（用节点“池化后”的编码器特征当初始特征）
        self.hgt = HGTBranch(metadata,
                             d_in_dict={"ncRNA": d_rna_enc, "drug": d_drug_enc},
                             d_model=d_model, n_layer=n_hgt_layer,
                             n_head=n_hgt_head, dropout=dropout)
        # ---- 阶段3b：超图（先把两类节点投到统一维，再拼成一个 id 空间）
        self.hyper_rna_in  = nn.Linear(d_rna_enc,  d_model)
        self.hyper_drug_in = nn.Linear(d_drug_enc, d_model)
        self.hyper = HyperBranch(d_model, d_model, n_hyper_layer, dropout)
        # ---- 两分支门控融合 → z_net
        self.gate_rna  = nn.Linear(2 * d_model, d_model)
        self.gate_drug = nn.Linear(2 * d_model, d_model)

        # ---- 阶段4（创新①）
        self.head = RelationCondHead(d_model, num_rel, head_hidden, dropout)

        # ---- 消融专用模块（按需构建）
        if ablation == "gcn":        # 阶段3 换普通同质 GCN
            self.gcn = GCNBranch(d_rna_enc, d_drug_enc, d_model, n_hgt_layer, dropout)
        if ablation == "no_relhead": # 阶段4 换共享头（消融创新①）
            self.shared_head = SharedHead(d_model, num_rel, head_hidden, dropout)

        # ---- v2 架构专用：注意力池化 + 药物分子指纹融合（替掉没用的图分支给头提供节点表示）
        if arch == "v2":
            self.attn_rna  = AttnPool(d_model)                 # 32个RNA区域 → 注意力汇总
            self.attn_drug = AttnPool(d_model)                 # 药物子结构 token → 注意力汇总
            d_ecfp = drug_ecfp.shape[1] if drug_ecfp is not None else 2048
            self.register_buffer("drug_ecfp",
                                 drug_ecfp if drug_ecfp is not None
                                 else torch.zeros(n_drug, d_ecfp))
            self.ecfp_proj = nn.Sequential(nn.Linear(d_ecfp, d_model), nn.ReLU(),
                                           nn.Dropout(dropout))
            # 可训练药物原子图 GCN（有 drug_graph 才建）——第三个药物视角
            self.has_dgnn = drug_graph is not None
            if self.has_dgnn:
                self.drug_gnn = DrugGraphEncoder(drug_graph["atom_dim"], d_model,
                                                 n_layer=3, dropout=dropout)
                self.register_buffer("dg_x", drug_graph["x"])
                self.register_buffer("dg_ei", drug_graph["edge_index"])
                self.register_buffer("dg_batch", drug_graph["batch"])
            n_view = 3 if self.has_dgnn else 2                 # ChemBERTa池化 ⊕ 指纹 (⊕ 原子图)
            self.drug_fuse = nn.Linear(n_view * d_model, d_model)
            # 类型感知机制调制(救活的创新②): 按 ncRNA 类型给不同 γ/β
            self.type_film = TypeFiLM(num_types=3, d_model=d_model)
            self.register_buffer("rna_type",
                                 rna_type if rna_type is not None else torch.zeros(n_rna, dtype=torch.long))
            # k-mer 区域特征投影(交互可解释性): 给RNA区域注入位置区分度(RNA-FM区域太雷同,交互图分不开)
            # 默认开启: 实测对AUC无损(0.7844→0.7852)且让交互图从条纹变真二维
            self.use_kmer = os.environ.get("NCDRUG_KMER_INT", "1") == "1"
            if self.use_kmer:
                self.kmer_proj = nn.Sequential(nn.Linear(64, d_model), nn.ReLU(), nn.Dropout(dropout))
            # 留一消融开关(v2真实组件): 去类型FiLM / 去药物多视角(只留ChemBERTa)
            self.no_typefilm = os.environ.get("NCDRUG_NO_TYPEFILM", "0") == "1"
            self.no_multiview = os.environ.get("NCDRUG_NO_MULTIVIEW", "0") == "1"
            # HGT 异质图分支(真接入v2): 图嵌入与内容表示融合后进头。NCDRUG_NO_HGT=1 关闭(留一消融)
            self.use_hgt = os.environ.get("NCDRUG_NO_HGT", "0") != "1"
            if self.use_hgt:
                # 门控残差融合: 图信号无用时门→0(退回内容表示, 不伤), 有用时门>0才加
                self.hgt_gate_r = nn.Linear(2 * d_model, d_model)
                self.hgt_gate_d = nn.Linear(2 * d_model, d_model)
                self.hgt_proj_r = nn.Linear(d_model, d_model)
                self.hgt_proj_d = nn.Linear(d_model, d_model)
            # 原始池化特征 skip 连接: 让预测头直接拿到与RF/XGBoost相同的1024维原始信息
            # (RNA-FM 640 + ChemBERTa 384 池化), 加到关联logit上。NCDRUG_SKIP 开关, 默认开。
            self.use_skip = os.environ.get("NCDRUG_SKIP", "1") == "1"
            if self.use_skip:
                self.skip_head = nn.Sequential(
                    nn.Linear(d_rna_enc + d_drug_enc, 512), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    # ---------------------------------------------------------------- 阶段3 (v2用: 纯HGT)
    def compute_hgt(self, graph):
        """v2 图分支: 只跑 HGT 异质图注意力, 返回 (ncRNA, drug) 图嵌入。"""
        h = self.hgt(graph["x_dict"], graph["edge_index_dict"])
        return h["ncRNA"], h["drug"]

    # ---------------------------------------------------------------- 阶段3
    def compute_graph(self, graph):
        """跑一次整张图，得到每个 ncRNA / drug 的 z_net。可每个 epoch 缓存复用。"""
        # 〔消融 gcn〕普通同质 GCN，不分关系/类型
        if self.ablation == "gcn":
            return self.gcn(graph["x_dict"]["ncRNA"], graph["x_dict"]["drug"],
                            graph["homo_edge_index"], self.n_rna)

        # 3a HGT（关系/类型感知注意力）
        h_hgt = self.hgt(graph["x_dict"], graph["edge_index_dict"])
        h_hgt_r, h_hgt_d = h_hgt["ncRNA"], h_hgt["drug"]

        # 〔消融 no_hyper〕只留 HGT，不要超图高阶分支
        if self.ablation == "no_hyper":
            return h_hgt_r, h_hgt_d

        # 3b 超图（统一 id 空间：前 n_rna 个是 ncRNA，后面是 drug）
        x_r = F.relu(self.hyper_rna_in(graph["x_dict"]["ncRNA"]))
        x_d = F.relu(self.hyper_drug_in(graph["x_dict"]["drug"]))
        x_all = torch.cat([x_r, x_d], dim=0)
        h_hyp = self.hyper(x_all, graph["hyperedge_index"])
        h_hyp_r, h_hyp_d = h_hyp[: self.n_rna], h_hyp[self.n_rna:]

        # 门控融合：模型自己学每个节点该多信 HGT / 多信 超图
        g_r = torch.sigmoid(self.gate_rna(torch.cat([h_hgt_r, h_hyp_r], dim=-1)))
        z_net_r = g_r * h_hgt_r + (1 - g_r) * h_hyp_r
        g_d = torch.sigmoid(self.gate_drug(torch.cat([h_hgt_d, h_hyp_d], dim=-1)))
        z_net_d = g_d * h_hgt_d + (1 - g_d) * h_hyp_d
        return z_net_r, z_net_d

    # ---------------------------------------------------------------- 阶段2
    def compute_pair(self, batch):
        """对一批 (ncRNA, drug) 对，跑阶段2 得到分关系的 z_pair 和交互图 M。"""
        Hr = self.rna_in(batch["Hr"])            # [B,Lr,d]
        if self.arch == "v2" and not getattr(self, "no_typefilm", False):  # 类型感知调制(创新②)
            Hr = self.type_film(Hr, self.rna_type[batch["rna_nid"]])
        if getattr(self, "use_kmer", False) and "Hr_kmer" in batch:
            Hr = Hr + self.kmer_proj(batch["Hr_kmer"])   # 叠加k-mer位置特征(破区域雷同)
        Hd = self.drug_in(batch["Hd"])           # [B,Ld,d]

        # 〔消融 no_film / v2架构〕不做靶点路由 FiLM 调制，药物子结构保持原样
        if self.ablation == "no_film" or self.arch == "v2":
            Hd_tilde, route = Hd, None
        else:
            ids = batch["tgt_ids"]
            if self.tgt_mode == "real":          # 真蛋白特征(ESM⊕角色)投影 + 可学习残差
                P = self.tgt_proj(self.tgt_feat[ids]) + self.tgt_res(ids)   # [B,T,d]
            else:
                P = self.tgt_emb(ids)            # 旧版可学习ID向量
            Hd_tilde, route = self.film(Hd, batch["Hd_mask"], P, batch["tgt_mask"])   # 创新②
        z_pair, M = self.interact(Hr, batch["Hr_mask"], Hd_tilde, batch["Hd_mask"])
        return z_pair, M, route

    # ---------------------------------------------------------------- 组装
    def forward(self, batch, graph, z_net_cache=None):
        """
        batch : dict(Hr,Hr_mask,Hd,Hd_mask,tgt_ids,tgt_mask,rna_nid,drug_nid)
        graph : dict(x_dict, edge_index_dict, hyperedge_index)
        z_net_cache : (z_net_r, z_net_d) 若已算好可传入，省去重复跑图
        """
        z_pair, M, route = self.compute_pair(batch)              # 阶段2

        # 〔v2 架构〕节点表示不再来自图，而是 序列注意力池化 + 药物分子指纹（有效信号替掉没用的图）
        if self.arch == "v2":
            Hr = self.rna_in(batch["Hr"])                        # [B,Lr,d]
            if not getattr(self, "no_typefilm", False):
                Hr = self.type_film(Hr, self.rna_type[batch["rna_nid"]])   # 类型调制(创新②)
            Hd = self.drug_in(batch["Hd"])                       # [B,Ld,d]
            zr = self.attn_rna(Hr, batch["Hr_mask"])             # [B,d] ncRNA序列表示
            zd_seq = self.attn_drug(Hd, batch["Hd_mask"])        # [B,d] 药物语言模型表示
            zd_fp = self.ecfp_proj(self.drug_ecfp[batch["drug_nid"]])  # [B,d] 分子指纹
            nomv = getattr(self, "no_multiview", False)          # 去多视角: 只留ChemBERTa, 其余视角置零(保持drug_fuse维度)
            if nomv: zd_fp = torch.zeros_like(zd_fp)
            views = [zd_seq, zd_fp]
            if self.has_dgnn:                                    # [B,d] 原子图GCN(可训练结构)
                dg_all = self.drug_gnn(self.dg_x, self.dg_ei, self.dg_batch, self.drug_ecfp.shape[0])
                dg_v = dg_all[batch["drug_nid"]]
                views.append(torch.zeros_like(dg_v) if nomv else dg_v)
            zd = F.relu(self.drug_fuse(torch.cat(views, dim=-1)))  # [B,d] 药物多视角
            # 只用交互项(堵退化): 砍掉跟药无关的 zr(和 zd 单点项), 逼预测靠药物特异的 z_pair
            if os.environ.get("NCDRUG_INT_ONLY", "0") == "1":
                zr = torch.zeros_like(zr); zd = torch.zeros_like(zd)
            # HGT 图分支: 关联图上的协同信号, 与内容表示融合(训练每minibatch重算, 评估传cache)
            if getattr(self, "use_hgt", False):
                if z_net_cache is not None:
                    g_r, g_d = z_net_cache
                else:
                    g_r, g_d = self.compute_hgt(graph)
                gr = g_r[batch["rna_nid"]]; gd = g_d[batch["drug_nid"]]
                zr = zr + torch.sigmoid(self.hgt_gate_r(torch.cat([zr, gr], -1))) * self.hgt_proj_r(gr)
                zd = zd + torch.sigmoid(self.hgt_gate_d(torch.cat([zd, gd], -1))) * self.hgt_proj_d(gd)
            head = self.head
            assoc_logit, rel_logits = head(z_pair, zr, zd)
            # 原始池化特征 skip: 直接给关联logit补上RF式的原始特征通路
            if getattr(self, "use_skip", False):
                raw_r = batch["Hr"].mean(1)                          # [B,640]
                dm = batch["Hd_mask"].unsqueeze(-1).float()
                raw_d = (batch["Hd"] * dm).sum(1) / dm.sum(1).clamp(min=1)  # [B,384]
                assoc_logit = assoc_logit + self.skip_head(torch.cat([raw_r, raw_d], -1)).squeeze(-1)
            return {"assoc_logit": assoc_logit, "rel_logits": rel_logits, "M": M, "route": route}

        # 〔消融 no_graph〕整个图分支不要，z_net 置零，只靠 pair 交互
        if self.ablation == "no_graph":
            B = z_pair.shape[0]
            zr = torch.zeros(B, self.d_model, device=z_pair.device)
            zd = torch.zeros(B, self.d_model, device=z_pair.device)
        else:
            if z_net_cache is None:
                z_net_r, z_net_d = self.compute_graph(graph)     # 阶段3
            else:
                z_net_r, z_net_d = z_net_cache
            zr = z_net_r[batch["rna_nid"]]                       # 取出这批对应的节点表示
            zd = z_net_d[batch["drug_nid"]]

        head = self.shared_head if self.ablation == "no_relhead" else self.head
        assoc_logit, rel_logits = head(z_pair, zr, zd)           # 阶段4（创新①）
        return {
            "assoc_logit": assoc_logit,   # [B]   是否关联
            "rel_logits":  rel_logits,    # [B,R] 关系类型打分
            "M": M,                       # [B,R,Lr,Ld] 分关系交互图（可解释①）
            "route": route,               # [B,Ld,T]    靶点路由权重（可解释②）
        }

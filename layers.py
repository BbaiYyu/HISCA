# -*- coding: utf-8 -*-
"""
5 个零件的积木层。每个类对应模型框架图里的一块：
  1) TargetRoutedFiLM   —— 阶段2 · 靶点路由 + FiLM 调制        〔创新②〕
  2) InteractionModule  —— 阶段2 · 交互图 M → z_pair（分关系）  〔配合创新①〕
  3) HGTBranch          —— 阶段3 · HGT 关系/类型感知注意力
  4) HyperBranch        —— 阶段3 · 超图高阶（拉群借信息）
  5) RelationCondHead   —— 阶段4 · 关系条件预测头               〔创新①〕
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, HypergraphConv, GCNConv


def masked_softmax(scores, mask, dim=-1):
    """在 mask=False 的位置填 -inf 再 softmax，避免 padding 干扰。"""
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=dim)
    # 某一行全被 mask（例如药物无靶点）时 softmax 会得到 nan，置 0
    return torch.nan_to_num(out, nan=0.0)


class TypeFiLM(nn.Module):
    """类型感知机制调制(救活的创新②)：按 ncRNA 类型(miRNA/lncRNA/circRNA)给不同的 γ、β。
    生物动机：三类 ncRNA 机制不同(miRNA种子靶向 / lncRNA·circRNA ceRNA海绵), 该分开调制。
    类型100%可得, 无靶点稀疏/冷启动问题。输入 X[B,L,d] 或 [B,d], type_idx[B]。"""
    def __init__(self, num_types, d_model):
        super().__init__()
        self.gamma = nn.Embedding(num_types, d_model)
        self.beta = nn.Embedding(num_types, d_model)
        nn.init.ones_(self.gamma.weight)      # 从恒等变换起步(γ=1,β=0), 稳
        nn.init.zeros_(self.beta.weight)

    def forward(self, X, type_idx):
        g, b = self.gamma(type_idx), self.beta(type_idx)   # [B,d]
        if X.dim() == 3:                                    # [B,L,d]: 在token维广播
            g, b = g.unsqueeze(1), b.unsqueeze(1)
        return g * X + b


class AttnPool(nn.Module):
    """可学习 query 的注意力池化：[B,L,d] + mask[B,L] → [B,d]。
    比 mean-pool 好——让模型自己选序列里重要的区域/子结构(保住长ncRNA的信息)。"""
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, X, mask):
        s = (torch.tanh(self.proj(X)) @ self.q)          # [B,L]
        s = s.masked_fill(~mask, float("-inf"))
        w = torch.nan_to_num(torch.softmax(s, dim=1), nan=0.0)  # [B,L]
        return (w.unsqueeze(-1) * X).sum(dim=1)          # [B,d]


class DrugGraphEncoder(nn.Module):
    """药物【原子级分子图】GCN 编码器(可训练)——比静态指纹更强的结构表征(PLMF的杀器)。
    对全部药物的分子图跑一次 GCN + 全局池化 → [N_drug, d_model], 再按 drug_nid 查表。
    输入: x[所有原子,69], edge_index[2,E], batch[所有原子](属于哪个药)。"""
    def __init__(self, atom_dim, d_model, n_layer=3, dropout=0.2):
        super().__init__()
        dims = [atom_dim] + [d_model] * n_layer
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(n_layer)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, n_drug):
        from torch_geometric.nn import global_mean_pool, global_max_pool
        for i, conv in enumerate(self.convs):
            x = self.bns[i](conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = self.dropout(F.relu(x))
        return global_mean_pool(x, batch, size=n_drug) + global_max_pool(x, batch, size=n_drug)


class MLP(nn.Module):
    """通用两层小网络。"""
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================= 零件 1（创新②）
class TargetRoutedFiLM(nn.Module):
    """
    阶段2 · 靶点路由 + FiLM 调制。
    输入:
        Hd   [B, Ld, d]  药物每个子结构的特征
        Hd_mask [B, Ld]  哪些子结构是真的（非 padding）
        P    [B, T, d]   这个药的靶点集合特征（不池化，全留着）
        P_mask [B, T]    哪些靶点是真的
    输出:
        Hd_tilde [B, Ld, d]  带上"靶点味道"的药物子结构
        beta [B, Ld, T]      路由权重（每块结构冲着哪个靶点，可解释）
    """
    def __init__(self, d_model, film_hidden, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)   # 药物子结构 → query
        self.k = nn.Linear(d_model, d_model)   # 靶点 → key
        self.scale = d_model ** 0.5
        # FiLM 的两个"旋钮网络"：读靶点上下文，输出 γ(放大) 和 β(偏移)
        self.gamma = MLP(d_model, film_hidden, d_model, dropout)
        self.beta  = MLP(d_model, film_hidden, d_model, dropout)

    def forward(self, Hd, Hd_mask, P, P_mask):
        # 第1步：每块结构对每个靶点打分  score[b,j,k]
        score = torch.matmul(self.q(Hd), self.k(P).transpose(1, 2)) / self.scale  # [B,Ld,T]
        # 第2步：softmax 变成百分比（路由权重），对无效靶点做 mask
        route = masked_softmax(score, P_mask.unsqueeze(1).expand_as(score), dim=-1)  # [B,Ld,T]
        # 第3步：按百分比把靶点揉成"这块结构专属的靶点上下文"
        p_ctx = torch.matmul(route, P)  # [B,Ld,d]
        # 第4步：FiLM —— 两个旋钮 γ、β 由靶点上下文算出，再改造子结构
        gamma = self.gamma(p_ctx)
        beta  = self.beta(p_ctx)
        Hd_tilde = gamma * Hd + beta                       # 逐特征：乘一个 + 加一个
        Hd_tilde = torch.where(Hd_mask.unsqueeze(-1), Hd_tilde, Hd)  # padding 位置保持原样
        return Hd_tilde, route


# ============================================================= 零件 2
class InteractionModule(nn.Module):
    """
    阶段2 · 交互图 M → z_pair。为每个关系各产出一套（配合创新①"分关系可解释"）。
    输入:
        Hr [B, Lr, d]        RNA 每个区域的特征
        Hr_mask [B, Lr]
        Hd_tilde [B, Ld, d]  零件1 输出（带靶点味道的药物子结构）
        Hd_mask [B, Ld]
    输出:
        z_pair [B, R, d]         每个关系一个 pair 向量
        M      [B, R, Lr, Ld]    每个关系一张交互图（可解释①）
    """
    def __init__(self, d_model, num_rel, dropout=0.0):
        super().__init__()
        self.num_rel = num_rel
        # 每个关系一套投影 —— 这就是"分关系给不同交互图"的来源
        self.r_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_rel)])
        self.d_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_rel)])
        self.joint  = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(num_rel)])
        self.scale = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, Hr, Hr_mask, Hd_tilde, Hd_mask):
        B, Lr, d = Hr.shape
        Ld = Hd_tilde.shape[1]
        cell_mask = Hr_mask.unsqueeze(2) & Hd_mask.unsqueeze(1)      # [B,Lr,Ld] 有效格子
        z_list, M_list = [], []
        for r in range(self.num_rel):
            rr = self.r_proj[r](Hr)                                  # [B,Lr,d]
            dd = self.d_proj[r](Hd_tilde)                            # [B,Ld,d]
            # 中心化: RNA-FM各区域几乎=公共向量c+微小残差δ, 内积被c主导→条纹。
            # 沿序列轴减去均值(去掉公共成分c), 只留区域/token特异残差做内积, 逼出二维结构。
            if os.environ.get("NCDRUG_INT_CENTER", "0") == "1":
                rr = rr - rr.mean(dim=1, keepdim=True)
                dd = dd - dd.mean(dim=1, keepdim=True)
            # L2归一化: RNA侧/药物侧投影特征幅度(量纲)不同, 内积被幅度大的一侧主导→条纹。
            # 沿特征维归一化 → 内积变余弦, 只比方向/模式, 去掉幅度主导, 两轴才平权。
            if os.environ.get("NCDRUG_INT_NORM", "0") == "1":
                rr = F.normalize(rr, dim=-1)
                dd = F.normalize(dd, dim=-1)
            # 交互图 M[b,i,j] = RNA区域 i 与 药物子结构 j 的匹配分
            H = int(os.environ.get("NCDRUG_INT_HEADS", 1))
            if H > 1:                                                # 多头双线性: 每头一套秩1模式,相加成秩>1(破条纹)
                dh = d // H
                rrh = rr.view(B, Lr, H, dh).transpose(1, 2)          # [B,H,Lr,dh]
                ddh = dd.view(B, Ld, H, dh).transpose(1, 2)          # [B,H,Ld,dh]
                Mh = torch.matmul(rrh, ddh.transpose(-1, -2)) / (dh ** 0.5)  # [B,H,Lr,Ld]
                M = Mh.sum(dim=1)                                    # [B,Lr,Ld] 各头相加
            else:
                M = torch.matmul(rr, dd.transpose(1, 2)) / self.scale    # [B,Lr,Ld]
            # 注意力池化（可分离：避免 materialize [B,Lr,Ld,d] 大张量，省显存）
            attn = M.masked_fill(~cell_mask, float("-inf"))
            attn = torch.softmax(attn.flatten(1), dim=1).view(B, Lr, Ld)  # 对整张图归一
            attn = torch.nan_to_num(attn, nan=0.0)
            row_w = attn.sum(dim=2)                                  # [B,Lr] 每个RNA区域的总权重
            col_w = attn.sum(dim=1)                                  # [B,Ld] 每个药物子结构的总权重
            z_r = (row_w.unsqueeze(-1) * rr).sum(dim=1)              # [B,d] RNA侧加权汇总
            z_d = (col_w.unsqueeze(-1) * dd).sum(dim=1)              # [B,d] 药物侧加权汇总
            z = self.joint[r](torch.cat([z_r, z_d], dim=-1))        # [B,d]
            z_list.append(self.dropout(z))
            M_list.append(M)
        z_pair = torch.stack(z_list, dim=1)                          # [B,R,d]
        M = torch.stack(M_list, dim=1)                               # [B,R,Lr,Ld]
        return z_pair, M


# ============================================================= 零件 3
class HGTBranch(nn.Module):
    """
    阶段3 · HGT 关系/类型感知注意力（换掉 R-GCN）。升级版（优化A）:
      · 每层【残差 + LayerNorm】—— 让它能安全加深(2→3/4层)不过平滑, 传到 2 跳(ncRNA→drug→ncRNA), 救冷启动。
      · 【JumpingKnowledge】跨层聚合(cat/max/last) —— 浅层身份 + 深层结构都用上。
      · 训练期对【关联边 DropEdge】—— 随机丢一部分 ncRNA↔drug 边, 直击"度捷径"过拟合(相似边不动)。
    metadata = (node_types, edge_types)，PyG 的标准异质图元信息。
    env 旋钮(默认即开, 消融时可关):
        NCDRUG_HGT_JK       = cat(默认) | max | last
        NCDRUG_HGT_DROPEDGE = 0.2(默认), 0=关
    输入:
        x_dict          {节点类型: [N, d_in]}  初始特征（= 编码器输出）
        edge_index_dict {边类型: [2, E]}
    输出:
        {节点类型: [N, d]}  吸饱邻居信息的节点表示
    """
    def __init__(self, metadata, d_in_dict, d_model, n_layer, n_head, dropout=0.0):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.jk = os.environ.get("NCDRUG_HGT_JK", "cat")
        self.drop_edge = float(os.environ.get("NCDRUG_HGT_DROPEDGE", "0.2"))
        self.assoc_ets = {et for et in edge_types if et[0] != et[2]}   # 跨类型=关联边(相似边同类型, 不丢)
        # 先把各类型节点投到统一维度
        self.lin_in = nn.ModuleDict({nt: nn.Linear(d_in_dict[nt], d_model) for nt in node_types})
        self.convs = nn.ModuleList([
            HGTConv(d_model, d_model, metadata, heads=n_head) for _ in range(n_layer)
        ])
        # 每层每类型一个 LayerNorm（配合残差, 稳住深层）
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(d_model) for nt in node_types}) for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(dropout)
        if self.jk == "cat":                                          # JK-cat 需把 n_layer*d 投回 d
            self.jk_lin = nn.ModuleDict({nt: nn.Linear(n_layer * d_model, d_model) for nt in node_types})

    def _dropedge(self, edge_index_dict):
        """训练期随机丢弃一部分关联边(不动相似边); 评估期原样返回。"""
        if not self.training or self.drop_edge <= 0:
            return edge_index_dict
        out = {}
        for et, ei in edge_index_dict.items():
            if et in self.assoc_ets and ei.shape[1] > 0:
                keep = torch.rand(ei.shape[1], device=ei.device) >= self.drop_edge
                out[et] = ei[:, keep]
            else:
                out[et] = ei
        return out

    def forward(self, x_dict, edge_index_dict):
        edge_index_dict = self._dropedge(edge_index_dict)
        h = {nt: F.relu(self.lin_in[nt](x)) for nt, x in x_dict.items()}
        outs = {nt: [] for nt in h}                                   # JK: 收集每层输出
        for li, conv in enumerate(self.convs):
            m = conv(h, edge_index_dict)                              # 消息传递
            new = {}
            for nt in h:
                upd = m[nt] if nt in m else h[nt]                     # 该类型无入边时保留原值
                new[nt] = self.norms[li][nt](h[nt] + self.dropout(F.relu(upd)))  # 残差 + LN
            h = new
            for nt in h:
                outs[nt].append(h[nt])
        if self.jk == "cat":
            h = {nt: self.jk_lin[nt](torch.cat(outs[nt], dim=-1)) for nt in h}
        elif self.jk == "max":
            h = {nt: torch.stack(outs[nt], dim=0).max(dim=0).values for nt in h}
        return h                                                      # jk="last": 已是最后一层


# ============================================================= 零件 4
class HyperBranch(nn.Module):
    """
    阶段3 · 超图高阶（拉群借信息）。所有节点放进统一 id 空间。
    输入:
        x               [N_all, d_in]   统一后的节点初始特征
        hyperedge_index [2, nnz]        row0=节点 id, row1=超边 id（共享靶点/相似成群）
    输出:
        [N_all, d]  经"点→群→点"两跳更新后的节点表示
    """
    def __init__(self, d_in, d_model, n_layer, dropout=0.0):
        super().__init__()
        self.lin_in = nn.Linear(d_in, d_model)
        self.convs = nn.ModuleList([HypergraphConv(d_model, d_model) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hyperedge_index):
        h = F.relu(self.lin_in(x))
        for conv in self.convs:
            h = conv(h, hyperedge_index)     # 内部就是 点→超边→点 两跳
            h = self.dropout(F.relu(h))
        return h


# ============================================================= 零件 5（创新①）
class RelationCondHead(nn.Module):
    """
    阶段4 · 关系条件预测头。三个"专科判官"，各判各的关系。
    输入:
        z_pair   [B, R, d]  分关系的细粒度作用向量（零件2）
        z_net_r  [B, d]     该 ncRNA 的图表示
        z_net_d  [B, d]     该 drug 的图表示
    输出:
        assoc_logit [B]     是否关联（二分类）
        rel_logits  [B, R]  关系类型打分（耐药/敏感/靶向）
    """
    def __init__(self, d_model, num_rel, hidden, dropout=0.0):
        super().__init__()
        self.num_rel = num_rel
        # 融合 z_net 的一对（ncRNA + drug）→ pair 级图表示
        self.net_fuse = nn.Linear(2 * d_model, d_model)
        # 每个关系一个"判官"：读 [分关系z_pair ⊕ z_net_pair] → 打一个分
        self.scorers = nn.ModuleList([MLP(2 * d_model, hidden, 1, dropout) for _ in range(num_rel)])
        # 是否关联的二分类头（读所有关系的融合表示）
        self.assoc = MLP(num_rel * (2 * d_model), hidden, 1, dropout)

    def forward(self, z_pair, z_net_r, z_net_d):
        z_net_pair = F.relu(self.net_fuse(torch.cat([z_net_r, z_net_d], dim=-1)))  # [B,d]
        fused, scores = [], []
        for r in range(self.num_rel):
            f = torch.cat([z_pair[:, r, :], z_net_pair], dim=-1)   # [B,2d]
            fused.append(f)
            scores.append(self.scorers[r](f).squeeze(-1))          # [B]
        rel_logits = torch.stack(scores, dim=1)                    # [B,R]
        assoc_logit = self.assoc(torch.cat(fused, dim=-1)).squeeze(-1)  # [B]
        return assoc_logit, rel_logits


# ============================================================= 消融零件 A（替换 HGT）
class GCNBranch(nn.Module):
    """
    消融用：普通同质 GCN，替换 HGTBranch。
    把异质图拍平成一张同质图（ncRNA 0..N_rna-1，drug 接在后面），
    所有边（关系边 + 相似边）不分类型一视同仁 —— 故意丢掉"关系/类型感知"。
    输入:
        x_rna [N_rna,d_rna], x_drug [N_drug,d_drug]
        edge_index [2,E]  同质 id 空间的边（drug 已 +N_rna 偏移）
        n_rna 用于拆回两类
    输出:
        z_r [N_rna,d], z_d [N_drug,d]
    """
    def __init__(self, d_in_rna, d_in_drug, d_model, n_layer, dropout=0.0):
        super().__init__()
        self.rna_in  = nn.Linear(d_in_rna,  d_model)
        self.drug_in = nn.Linear(d_in_drug, d_model)
        self.convs = nn.ModuleList([GCNConv(d_model, d_model) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_rna, x_drug, edge_index, n_rna):
        x = torch.cat([F.relu(self.rna_in(x_rna)), F.relu(self.drug_in(x_drug))], dim=0)
        for conv in self.convs:
            x = self.dropout(F.relu(conv(x, edge_index)))
        return x[:n_rna], x[n_rna:]


# ============================================================= 消融零件 B（替换关系条件头）
class SharedHead(nn.Module):
    """
    消融用：共享预测头，替换 RelationCondHead（消融创新①）。
    不再"每个关系一个专科判官"，而是把分关系的 z_pair 平均池化成一个向量，
    过一个共享主干，再线性输出 assoc + 各关系分。
    """
    def __init__(self, d_model, num_rel, hidden, dropout=0.0):
        super().__init__()
        self.net_fuse = nn.Linear(2 * d_model, d_model)
        self.trunk = MLP(2 * d_model, hidden, hidden, dropout)
        self.assoc = nn.Linear(hidden, 1)
        self.rel   = nn.Linear(hidden, num_rel)

    def forward(self, z_pair, z_net_r, z_net_d):
        z_net_pair = F.relu(self.net_fuse(torch.cat([z_net_r, z_net_d], dim=-1)))  # [B,d]
        z = z_pair.mean(dim=1)                                  # [B,d] 关系维度池化（丢掉分关系条件）
        h = self.trunk(torch.cat([z, z_net_pair], dim=-1))      # [B,hidden]
        return self.assoc(h).squeeze(-1), self.rel(h)

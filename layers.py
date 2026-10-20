

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, HypergraphConv, GCNConv


def masked_softmax(scores, mask, dim=-1):

    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=dim)

    return torch.nan_to_num(out, nan=0.0)


class TypeFiLM(nn.Module):


    def __init__(self, num_types, d_model):
        super().__init__()
        self.gamma = nn.Embedding(num_types, d_model)
        self.beta = nn.Embedding(num_types, d_model)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, X, type_idx):
        g, b = self.gamma(type_idx), self.beta(type_idx)
        if X.dim() == 3:
            g, b = g.unsqueeze(1), b.unsqueeze(1)
        return g * X + b


class AttnPool(nn.Module):


    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, X, mask):
        s = (torch.tanh(self.proj(X)) @ self.q)
        s = s.masked_fill(~mask, float("-inf"))
        w = torch.nan_to_num(torch.softmax(s, dim=1), nan=0.0)
        return (w.unsqueeze(-1) * X).sum(dim=1)


class DrugGraphEncoder(nn.Module):


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

    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TargetRoutedFiLM(nn.Module):


    def __init__(self, d_model, film_hidden, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

        self.gamma = MLP(d_model, film_hidden, d_model, dropout)
        self.beta  = MLP(d_model, film_hidden, d_model, dropout)

    def forward(self, Hd, Hd_mask, P, P_mask):

        score = torch.matmul(self.q(Hd), self.k(P).transpose(1, 2)) / self.scale

        route = masked_softmax(score, P_mask.unsqueeze(1).expand_as(score), dim=-1)

        p_ctx = torch.matmul(route, P)

        gamma = self.gamma(p_ctx)
        beta  = self.beta(p_ctx)
        Hd_tilde = gamma * Hd + beta
        Hd_tilde = torch.where(Hd_mask.unsqueeze(-1), Hd_tilde, Hd)
        return Hd_tilde, route


class InteractionModule(nn.Module):


    def __init__(self, d_model, num_rel, dropout=0.0):
        super().__init__()
        self.num_rel = num_rel

        self.r_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_rel)])
        self.d_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_rel)])
        self.joint  = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(num_rel)])
        self.scale = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, Hr, Hr_mask, Hd_tilde, Hd_mask):
        B, Lr, d = Hr.shape
        Ld = Hd_tilde.shape[1]
        cell_mask = Hr_mask.unsqueeze(2) & Hd_mask.unsqueeze(1)
        z_list, M_list = [], []
        for r in range(self.num_rel):
            rr = self.r_proj[r](Hr)
            dd = self.d_proj[r](Hd_tilde)


            if os.environ.get("NCDRUG_INT_CENTER", "0") == "1":
                rr = rr - rr.mean(dim=1, keepdim=True)
                dd = dd - dd.mean(dim=1, keepdim=True)


            if os.environ.get("NCDRUG_INT_NORM", "0") == "1":
                rr = F.normalize(rr, dim=-1)
                dd = F.normalize(dd, dim=-1)

            M = torch.matmul(rr, dd.transpose(1, 2)) / self.scale

            attn = M.masked_fill(~cell_mask, float("-inf"))
            attn = torch.softmax(attn.flatten(1), dim=1).view(B, Lr, Ld)
            attn = torch.nan_to_num(attn, nan=0.0)
            row_w = attn.sum(dim=2)
            col_w = attn.sum(dim=1)
            z_r = (row_w.unsqueeze(-1) * rr).sum(dim=1)
            z_d = (col_w.unsqueeze(-1) * dd).sum(dim=1)
            z = self.joint[r](torch.cat([z_r, z_d], dim=-1))
            z_list.append(self.dropout(z))
            M_list.append(M)
        z_pair = torch.stack(z_list, dim=1)
        M = torch.stack(M_list, dim=1)
        return z_pair, M


class HGTBranch(nn.Module):


    def __init__(self, metadata, d_in_dict, d_model, n_layer, n_head, dropout=0.0):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.jk = os.environ.get("NCDRUG_HGT_JK", "cat")
        self.drop_edge = float(os.environ.get("NCDRUG_HGT_DROPEDGE", "0.2"))
        self.assoc_ets = {et for et in edge_types if et[0] != et[2]}

        self.lin_in = nn.ModuleDict({nt: nn.Linear(d_in_dict[nt], d_model) for nt in node_types})
        self.convs = nn.ModuleList([
            HGTConv(d_model, d_model, metadata, heads=n_head) for _ in range(n_layer)
        ])

        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(d_model) for nt in node_types}) for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(dropout)
        if self.jk == "cat":
            self.jk_lin = nn.ModuleDict({nt: nn.Linear(n_layer * d_model, d_model) for nt in node_types})

    def _dropedge(self, edge_index_dict):

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
        outs = {nt: [] for nt in h}
        for li, conv in enumerate(self.convs):
            m = conv(h, edge_index_dict)
            new = {}
            for nt in h:
                upd = m[nt] if nt in m else h[nt]
                new[nt] = self.norms[li][nt](h[nt] + self.dropout(F.relu(upd)))
            h = new
            for nt in h:
                outs[nt].append(h[nt])
        if self.jk == "cat":
            h = {nt: self.jk_lin[nt](torch.cat(outs[nt], dim=-1)) for nt in h}
        elif self.jk == "max":
            h = {nt: torch.stack(outs[nt], dim=0).max(dim=0).values for nt in h}
        return h


class HyperBranch(nn.Module):


    def __init__(self, d_in, d_model, n_layer, dropout=0.0):
        super().__init__()
        self.lin_in = nn.Linear(d_in, d_model)
        self.convs = nn.ModuleList([HypergraphConv(d_model, d_model) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hyperedge_index):
        h = F.relu(self.lin_in(x))
        for conv in self.convs:
            h = conv(h, hyperedge_index)
            h = self.dropout(F.relu(h))
        return h


class RelationCondHead(nn.Module):


    def __init__(self, d_model, num_rel, hidden, dropout=0.0):
        super().__init__()
        self.num_rel = num_rel

        self.net_fuse = nn.Linear(2 * d_model, d_model)

        self.scorers = nn.ModuleList([MLP(2 * d_model, hidden, 1, dropout) for _ in range(num_rel)])

        self.assoc = MLP(num_rel * (2 * d_model), hidden, 1, dropout)

    def forward(self, z_pair, z_net_r, z_net_d):
        z_net_pair = F.relu(self.net_fuse(torch.cat([z_net_r, z_net_d], dim=-1)))
        fused, scores = [], []
        for r in range(self.num_rel):
            f = torch.cat([z_pair[:, r, :], z_net_pair], dim=-1)
            fused.append(f)
            scores.append(self.scorers[r](f).squeeze(-1))
        rel_logits = torch.stack(scores, dim=1)
        assoc_logit = self.assoc(torch.cat(fused, dim=-1)).squeeze(-1)
        return assoc_logit, rel_logits


class GCNBranch(nn.Module):


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


class SharedHead(nn.Module):


    def __init__(self, d_model, num_rel, hidden, dropout=0.0):
        super().__init__()
        self.net_fuse = nn.Linear(2 * d_model, d_model)
        self.trunk = MLP(2 * d_model, hidden, hidden, dropout)
        self.assoc = nn.Linear(hidden, 1)
        self.rel   = nn.Linear(hidden, num_rel)

    def forward(self, z_pair, z_net_r, z_net_d):
        z_net_pair = F.relu(self.net_fuse(torch.cat([z_net_r, z_net_d], dim=-1)))
        z = z_pair.mean(dim=1)
        h = self.trunk(torch.cat([z, z_net_pair], dim=-1))
        return self.assoc(h).squeeze(-1), self.rel(h)

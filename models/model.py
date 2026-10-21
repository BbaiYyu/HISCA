

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (TargetRoutedFiLM, InteractionModule,
                    HGTBranch, HyperBranch, RelationCondHead,
                    GCNBranch, SharedHead, AttnPool, DrugGraphEncoder, TypeFiLM)


class TRHGNet(nn.Module):


    def __init__(self, num_targets, metadata, n_rna, n_drug,
                 d_rna_enc, d_drug_enc, d_model,
                 num_rel, film_hidden, n_hgt_layer, n_hgt_head,
                 n_hyper_layer, head_hidden, dropout=0.2, ablation="none",
                 tgt_feat=None, tgt_feat_mode="learnable",
                 arch="orig", drug_ecfp=None, drug_graph=None, rna_type=None):
        super().__init__()
        self.n_rna = n_rna
        self.d_model = d_model

        self.ablation = ablation

        self.arch = arch


        self.rna_in  = nn.Linear(d_rna_enc,  d_model)
        self.drug_in = nn.Linear(d_drug_enc, d_model)


        self.tgt_mode = tgt_feat_mode if tgt_feat is not None else "learnable"
        if self.tgt_mode == "real":
            self.register_buffer("tgt_feat", tgt_feat)
            self.tgt_proj = nn.Linear(tgt_feat.shape[1], d_model)
            self.tgt_res  = nn.Embedding(num_targets + 1, d_model, padding_idx=0)
            nn.init.zeros_(self.tgt_res.weight)
        else:
            self.tgt_emb = nn.Embedding(num_targets + 1, d_model, padding_idx=0)


        self.film     = TargetRoutedFiLM(d_model, film_hidden, dropout)
        self.interact = InteractionModule(d_model, num_rel, dropout)


        self.hgt = HGTBranch(metadata,
                             d_in_dict={"ncRNA": d_rna_enc, "drug": d_drug_enc},
                             d_model=d_model, n_layer=n_hgt_layer,
                             n_head=n_hgt_head, dropout=dropout)

        self.hyper_rna_in  = nn.Linear(d_rna_enc,  d_model)
        self.hyper_drug_in = nn.Linear(d_drug_enc, d_model)
        self.hyper = HyperBranch(d_model, d_model, n_hyper_layer, dropout)

        self.gate_rna  = nn.Linear(2 * d_model, d_model)
        self.gate_drug = nn.Linear(2 * d_model, d_model)


        self.head = RelationCondHead(d_model, num_rel, head_hidden, dropout)


        if ablation == "gcn":
            self.gcn = GCNBranch(d_rna_enc, d_drug_enc, d_model, n_hgt_layer, dropout)
        if ablation == "no_relhead":
            self.shared_head = SharedHead(d_model, num_rel, head_hidden, dropout)


        if arch == "v2":
            self.attn_rna  = AttnPool(d_model)
            self.attn_drug = AttnPool(d_model)
            d_ecfp = drug_ecfp.shape[1] if drug_ecfp is not None else 2048
            self.register_buffer("drug_ecfp",
                                 drug_ecfp if drug_ecfp is not None
                                 else torch.zeros(n_drug, d_ecfp))
            self.ecfp_proj = nn.Sequential(nn.Linear(d_ecfp, d_model), nn.ReLU(),
                                           nn.Dropout(dropout))

            self.has_dgnn = drug_graph is not None
            if self.has_dgnn:
                self.drug_gnn = DrugGraphEncoder(drug_graph["atom_dim"], d_model,
                                                 n_layer=3, dropout=dropout)
                self.register_buffer("dg_x", drug_graph["x"])
                self.register_buffer("dg_ei", drug_graph["edge_index"])
                self.register_buffer("dg_batch", drug_graph["batch"])
            n_view = 3 if self.has_dgnn else 2
            self.drug_fuse = nn.Linear(n_view * d_model, d_model)

            self.type_film = TypeFiLM(num_types=3, d_model=d_model)
            self.register_buffer("rna_type",
                                 rna_type if rna_type is not None else torch.zeros(n_rna, dtype=torch.long))


            self.use_kmer = os.environ.get("NCDRUG_KMER_INT", "1") == "1"
            if self.use_kmer:
                self.kmer_proj = nn.Sequential(nn.Linear(64, d_model), nn.ReLU(), nn.Dropout(dropout))

            self.no_typefilm = os.environ.get("NCDRUG_NO_TYPEFILM", "0") == "1"
            self.no_multiview = os.environ.get("NCDRUG_NO_MULTIVIEW", "0") == "1"

            self.use_hgt = os.environ.get("NCDRUG_NO_HGT", "0") != "1"
            if self.use_hgt:

                self.hgt_gate_r = nn.Linear(2 * d_model, d_model)
                self.hgt_gate_d = nn.Linear(2 * d_model, d_model)
                self.hgt_proj_r = nn.Linear(d_model, d_model)
                self.hgt_proj_d = nn.Linear(d_model, d_model)


            self.use_skip = os.environ.get("NCDRUG_SKIP", "1") == "1"
            if self.use_skip:
                self.skip_head = nn.Sequential(
                    nn.Linear(d_rna_enc + d_drug_enc, 512), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))


    def compute_hgt(self, graph):

        h = self.hgt(graph["x_dict"], graph["edge_index_dict"])
        hgt_r, hgt_d = h["ncRNA"], h["drug"]
        if (os.environ.get("NCDRUG_V2_HYPER", "0") == "1"
                and graph.get("hyperedge_index") is not None
                and graph["hyperedge_index"].shape[1] > 0):
            x_r = F.relu(self.hyper_rna_in(graph["x_dict"]["ncRNA"]))
            x_d = F.relu(self.hyper_drug_in(graph["x_dict"]["drug"]))
            h_hyp = self.hyper(torch.cat([x_r, x_d], dim=0), graph["hyperedge_index"])
            hyp_r, hyp_d = h_hyp[: self.n_rna], h_hyp[self.n_rna:]
            g_r = torch.sigmoid(self.gate_rna(torch.cat([hgt_r, hyp_r], -1)))
            hgt_r = g_r * hgt_r + (1 - g_r) * hyp_r
            g_d = torch.sigmoid(self.gate_drug(torch.cat([hgt_d, hyp_d], -1)))
            hgt_d = g_d * hgt_d + (1 - g_d) * hyp_d
        return hgt_r, hgt_d


    def compute_graph(self, graph):


        if self.ablation == "gcn":
            return self.gcn(graph["x_dict"]["ncRNA"], graph["x_dict"]["drug"],
                            graph["homo_edge_index"], self.n_rna)


        h_hgt = self.hgt(graph["x_dict"], graph["edge_index_dict"])
        h_hgt_r, h_hgt_d = h_hgt["ncRNA"], h_hgt["drug"]


        if self.ablation == "no_hyper":
            return h_hgt_r, h_hgt_d


        x_r = F.relu(self.hyper_rna_in(graph["x_dict"]["ncRNA"]))
        x_d = F.relu(self.hyper_drug_in(graph["x_dict"]["drug"]))
        x_all = torch.cat([x_r, x_d], dim=0)
        h_hyp = self.hyper(x_all, graph["hyperedge_index"])
        h_hyp_r, h_hyp_d = h_hyp[: self.n_rna], h_hyp[self.n_rna:]


        g_r = torch.sigmoid(self.gate_rna(torch.cat([h_hgt_r, h_hyp_r], dim=-1)))
        z_net_r = g_r * h_hgt_r + (1 - g_r) * h_hyp_r
        g_d = torch.sigmoid(self.gate_drug(torch.cat([h_hgt_d, h_hyp_d], dim=-1)))
        z_net_d = g_d * h_hgt_d + (1 - g_d) * h_hyp_d
        return z_net_r, z_net_d


    def compute_pair(self, batch):

        Hr = self.rna_in(batch["Hr"])
        if self.arch == "v2" and not getattr(self, "no_typefilm", False):
            Hr = self.type_film(Hr, self.rna_type[batch["rna_nid"]])
        if getattr(self, "use_kmer", False) and "Hr_kmer" in batch:
            Hr = Hr + self.kmer_proj(batch["Hr_kmer"])
        Hd = self.drug_in(batch["Hd"])


        if self.ablation == "no_film" or self.arch == "v2":
            Hd_tilde, route = Hd, None
        else:
            ids = batch["tgt_ids"]
            if self.tgt_mode == "real":
                P = self.tgt_proj(self.tgt_feat[ids]) + self.tgt_res(ids)
            else:
                P = self.tgt_emb(ids)
            Hd_tilde, route = self.film(Hd, batch["Hd_mask"], P, batch["tgt_mask"])
        z_pair, M = self.interact(Hr, batch["Hr_mask"], Hd_tilde, batch["Hd_mask"])
        return z_pair, M, route


    def forward(self, batch, graph, z_net_cache=None):


        z_pair, M, route = self.compute_pair(batch)


        if self.arch == "v2":
            Hr = self.rna_in(batch["Hr"])
            if not getattr(self, "no_typefilm", False):
                Hr = self.type_film(Hr, self.rna_type[batch["rna_nid"]])
            Hd = self.drug_in(batch["Hd"])
            zr = self.attn_rna(Hr, batch["Hr_mask"])
            zd_seq = self.attn_drug(Hd, batch["Hd_mask"])
            zd_fp = self.ecfp_proj(self.drug_ecfp[batch["drug_nid"]])
            nomv = getattr(self, "no_multiview", False)
            if nomv: zd_fp = torch.zeros_like(zd_fp)
            views = [zd_seq, zd_fp]
            if self.has_dgnn:
                dg_all = self.drug_gnn(self.dg_x, self.dg_ei, self.dg_batch, self.drug_ecfp.shape[0])
                dg_v = dg_all[batch["drug_nid"]]
                views.append(torch.zeros_like(dg_v) if nomv else dg_v)
            zd = F.relu(self.drug_fuse(torch.cat(views, dim=-1)))

            if os.environ.get("NCDRUG_INT_ONLY", "0") == "1":
                zr = torch.zeros_like(zr); zd = torch.zeros_like(zd)

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

            if getattr(self, "use_skip", False):
                raw_r = batch["Hr"].mean(1)
                dm = batch["Hd_mask"].unsqueeze(-1).float()
                raw_d = (batch["Hd"] * dm).sum(1) / dm.sum(1).clamp(min=1)
                assoc_logit = assoc_logit + self.skip_head(torch.cat([raw_r, raw_d], -1)).squeeze(-1)
            return {"assoc_logit": assoc_logit, "rel_logits": rel_logits, "M": M, "route": route}


        if self.ablation == "no_graph":
            B = z_pair.shape[0]
            zr = torch.zeros(B, self.d_model, device=z_pair.device)
            zd = torch.zeros(B, self.d_model, device=z_pair.device)
        else:
            if z_net_cache is None:
                z_net_r, z_net_d = self.compute_graph(graph)
            else:
                z_net_r, z_net_d = z_net_cache
            zr = z_net_r[batch["rna_nid"]]
            zd = z_net_d[batch["drug_nid"]]

        head = self.shared_head if self.ablation == "no_relhead" else self.head
        assoc_logit, rel_logits = head(z_pair, zr, zd)
        return {
            "assoc_logit": assoc_logit,
            "rel_logits":  rel_logits,
            "M": M,
            "route": route,
        }

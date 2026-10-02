# -*- coding: utf-8 -*-
"""
阶段1 · 编码器：把 序列 / SMILES / 靶点 变成模型能吃的数字，预计算并存盘。
产出（存到 config.FEAT_DIR）：
    rna_feats.pt   {'feat':[N_rna,K,D_RNA_ENC], 'name2idx':{...}}       每条 RNA → K 个区域特征
    drug_feats.pt  {'feat':[N_drug,Ld,D_DRUG_ENC], 'mask':[N_drug,Ld], 'name2idx':{...}}
    target_feats.pt{'tgt_ids':[N_drug,T], 'mask':[N_drug,T], 'vocab':{uniprot:idx}, 'num_targets':int}
说明：
    - RNA 逐核苷酸特征太大，池化成固定 K 个"区域"（对应交互图 M 的行）。
    - 药物分子小，按 token 存（M 的列）。
    - 靶点集合"不池化"，每药存一串靶点 id（padding 到 T）。
用法：
    python encoders.py            # 用外部基础模型（RNA-FM / ChemBERTa）
    python encoders.py --fallback # 离线占位编码器，先把管线跑通
"""
import os, csv, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

csv.field_size_limit(10 ** 8)   # ncRNA.csv 有超长序列


# ============================================================ RNA 编码器
def _load_rnafm():
    """稳健加载 RNA-FM 预训练权重。
    本地 multimolecule 0.0.7 直接 from_pretrained 会因 checkpoint 键前缀不匹配
    （ckpt 是 `model.*`/`encoder.layer_norm`，旧类要 `*`/`encoder.emb_layer_norm_after`）
    导致 backbone 全部随机初始化。这里手动重映射后灌入，保证用的是真预训练权重。"""
    import glob
    from safetensors.torch import load_file
    from multimolecule import RnaFmModel, RnaFmConfig
    cfg = RnaFmConfig.from_pretrained("multimolecule/rnafm")
    model = RnaFmModel(cfg)
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    cands = glob.glob(os.path.join(cache, "**", "*rnafm*", "**", "*.safetensors"), recursive=True)
    if not cands:
        raise FileNotFoundError("找不到 rnafm 的 safetensors 权重，请先联网下载一次 multimolecule/rnafm")
    sd = load_file(cands[0])
    new = {}
    for k, v in sd.items():
        if k.startswith("lm_head") or k.startswith("ss_head") or k.startswith("pretrain"):
            continue
        nk = k[len("model."):] if k.startswith("model.") else k
        if nk == "encoder.layer_norm.weight": nk = "encoder.emb_layer_norm_after.weight"
        if nk == "encoder.layer_norm.bias":   nk = "encoder.emb_layer_norm_after.bias"
        new[nk] = v
    res = model.load_state_dict(new, strict=False)
    still = [k for k in res.missing_keys if not k.startswith("pooler")]
    if still:
        raise RuntimeError(f"RNA-FM 权重未完整加载，仍随机的非pooler键: {still}")
    print(f"[RNA-FM] 权重已正确加载（仅 pooler 未用，随机；backbone 全部命中）")
    return model


class RNAEncoder:
    """ncRNA 序列 → 逐核苷酸向量。优先 RNA-FM，失败则用离线占位。"""
    def __init__(self, backend="rnafm", device="cpu"):
        self.device = device
        self.backend = backend
        self.dim = C.D_RNA_ENC
        if backend == "rnafm":
            # RNA-FM 的 HF 官方移植版（官网 .pth 下载被封，改用 multimolecule）
            from multimolecule import RnaTokenizer
            self.tok = RnaTokenizer.from_pretrained("multimolecule/rnafm")
            self.model = _load_rnafm().to(device).eval()
        else:  # fallback：每个核苷酸一个固定随机向量（占位，仅供跑通管线）
            g = torch.Generator().manual_seed(0)
            self.emb = torch.randn(16, self.dim, generator=g)   # A/U/G/C/... 索引
            self.vocab = {c: i for i, c in enumerate("AUGCNTaugcnt")}

    @torch.no_grad()
    def encode(self, seqs):
        """seqs: list[str] -> list[Tensor[Li, dim]]（去掉首尾特殊符）。"""
        seqs = [(s or "N").upper().replace("T", "U")[: C.RNA_MAX_LEN] for s in seqs]
        if self.backend == "rnafm":
            # 分块编码：长序列切成 ≤RNA_WINDOW 的窗口，逐段过 RNA-FM 再拼回，一段不丢
            W = C.RNA_WINDOW
            wins = []                                          # (seq_idx, subseq)
            for i, s in enumerate(seqs):
                s = s or "N"
                for b in range(0, len(s), W):
                    wins.append((i, s[b:b + W]))
            per_seq = [[] for _ in seqs]
            for b in range(0, len(wins), C.ENC_BATCH):         # 窗口按 batch 编码（显存可控）
                grp = wins[b:b + C.ENC_BATCH]
                texts = [w for _, w in grp]
                enc = self.tok(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=W + 2)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc).last_hidden_state      # [G,l+2,640]（首cls尾eos）
                for k, (si, w) in enumerate(grp):
                    per_seq[si].append(out[k, 1:len(w) + 1].cpu())
            return [torch.cat(ch, dim=0) if ch else torch.zeros(1, self.dim) for ch in per_seq]
        else:
            res = []
            for s in seqs:
                idx = torch.tensor([self.vocab.get(c, 4) for c in s.replace("T", "U")])
                res.append(self.emb[idx])
            return res


# ============================================================ 药物编码器
class DrugEncoder:
    """药物 SMILES → 逐 token 向量。优先 ChemBERTa，失败则离线占位。"""
    def __init__(self, backend="chemberta", device="cpu"):
        self.device = device
        self.backend = backend
        self.dim = C.D_DRUG_ENC
        if backend == "chemberta":
            from transformers import AutoTokenizer, AutoModel
            self.tok = AutoTokenizer.from_pretrained(C.CHEMBERTA_MODEL)
            self.model = AutoModel.from_pretrained(C.CHEMBERTA_MODEL).to(device).eval()
        else:
            g = torch.Generator().manual_seed(1)
            self.emb = torch.randn(128, self.dim, generator=g)

    @torch.no_grad()
    def encode(self, smiles):
        """smiles: list[str] -> list[Tensor[Li, dim]]。"""
        smiles = [s or "C" for s in smiles]
        if self.backend == "chemberta":
            enc = self.tok(smiles, return_tensors="pt", padding=True,
                           truncation=True, max_length=C.LD_MAX)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc).last_hidden_state  # [B,L,d]
            res = []
            for i in range(len(smiles)):
                n = int(enc["attention_mask"][i].sum())
                res.append(out[i, :n].cpu())
            return res
        else:
            res = []
            for s in smiles[:]:
                idx = torch.tensor([ord(c) % 128 for c in s[: C.LD_MAX]])
                res.append(self.emb[idx])
            return res


# ============================================================ 工具
def region_pool(x, K):
    """[L, d] 逐核苷酸 → [K, d] 固定区域（自适应平均池化）。"""
    if x.shape[0] == 0:
        return torch.zeros(K, x.shape[1])
    xp = x.t().unsqueeze(0)                       # [1,d,L]
    xp = F.adaptive_avg_pool1d(xp, K)             # [1,d,K]
    return xp.squeeze(0).t()                      # [K,d]


def pad_stack(tensors, L_max, dim):
    """list[Tensor[Li,d]] -> ([N,L_max,d], mask[N,L_max])。"""
    N = len(tensors)
    out = torch.zeros(N, L_max, dim)
    mask = torch.zeros(N, L_max, dtype=torch.bool)
    for i, t in enumerate(tensors):
        n = min(t.shape[0], L_max)
        out[i, :n] = t[:n]
        mask[i, :n] = True
    return out, mask


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ============================================================ 主流程
def precompute(backend_override=None):
    os.makedirs(C.FEAT_DIR, exist_ok=True)
    dev = C.DEVICE
    rna_backend  = backend_override or C.RNA_BACKEND
    drug_backend = backend_override or C.DRUG_BACKEND

    # ---- 只对建模用到的 ncRNA / drug 编码（以 merged_final 出现的为准）
    edges = read_csv(C.EDGE_CSV)
    rna_names  = sorted({r["ncRNA_Name"] for r in edges})
    drug_names = sorted({r["Drug_Name"] for r in edges})

    rna_seq  = {r["ncRNA_Name"]: r.get("Sequence", "") for r in read_csv(C.NCRNA_CSV)}
    drug_smi = {r["Drug_Name"]: r.get("SMILES", "")   for r in read_csv(C.DRUG_CSV)}

    # ---------- RNA → K 区域
    print(f"[RNA] {len(rna_names)} 条，backend={rna_backend}")
    renc = RNAEncoder(rna_backend, dev)
    rna_feat = torch.zeros(len(rna_names), C.K_RNA_REGION, C.D_RNA_ENC)
    for b in range(0, len(rna_names), C.ENC_BATCH):
        chunk = rna_names[b:b + C.ENC_BATCH]
        embs = renc.encode([rna_seq.get(n, "") for n in chunk])
        for j, e in enumerate(embs):
            rna_feat[b + j] = region_pool(e, C.K_RNA_REGION)
        if b % (C.ENC_BATCH * 20) == 0:
            print(f"  RNA {b}/{len(rna_names)}", flush=True)
    torch.save({"feat": rna_feat, "name2idx": {n: i for i, n in enumerate(rna_names)}},
               os.path.join(C.FEAT_DIR, "rna_feats.pt"))

    # ---------- 药物 → token
    print(f"[DRUG] {len(drug_names)} 个，backend={drug_backend}")
    denc = DrugEncoder(drug_backend, dev)
    drug_embs = []
    for b in range(0, len(drug_names), C.ENC_BATCH):
        chunk = drug_names[b:b + C.ENC_BATCH]
        drug_embs += denc.encode([drug_smi.get(n, "") for n in chunk])
    drug_feat, drug_mask = pad_stack(drug_embs, C.LD_MAX, C.D_DRUG_ENC)
    torch.save({"feat": drug_feat, "mask": drug_mask,
                "name2idx": {n: i for i, n in enumerate(drug_names)}},
               os.path.join(C.FEAT_DIR, "drug_feats.pt"))

    # ---------- 靶点集合（不池化）
    build_target_features(drug_names)
    print("阶段1 预计算完成 ->", C.FEAT_DIR)


def build_target_features(drug_names):
    """读 drug_targets.csv → 每药一串靶点 id（padding 到 T_MAX）。缺文件则全空（自动降级）。"""
    drug2tgt = {}
    if os.path.exists(C.DRUG_TGT_CSV):
        for r in read_csv(C.DRUG_TGT_CSV):
            ups = [x for x in (r.get("Target_UniProt", "") or "").split(";") if x]
            if ups:
                drug2tgt[r["Drug_Name"]] = ups
    else:
        print(f"  [警告] 缺 {C.DRUG_TGT_CSV}，靶点特征全空 → 阶段2 走降级")

    vocab = {}                                    # uniprot -> idx（从 1 开始，0 留给 padding）
    for ups in drug2tgt.values():
        for u in ups:
            vocab.setdefault(u, len(vocab) + 1)

    N = len(drug_names)
    tgt_ids = torch.zeros(N, C.T_MAX, dtype=torch.long)
    mask = torch.zeros(N, C.T_MAX, dtype=torch.bool)
    for i, name in enumerate(drug_names):
        ups = drug2tgt.get(name, [])[: C.T_MAX]
        for j, u in enumerate(ups):
            tgt_ids[i, j] = vocab[u]
            mask[i, j] = True
    torch.save({"tgt_ids": tgt_ids, "mask": mask, "vocab": vocab, "num_targets": len(vocab)},
               os.path.join(C.FEAT_DIR, "target_feats.pt"))
    cov = sum(1 for n in drug_names if n in drug2tgt)
    print(f"  靶点：{len(vocab)} 个蛋白，{cov}/{N} 个药有靶点")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", action="store_true", help="用离线占位编码器跑通管线")
    args = ap.parse_args()
    precompute(backend_override="fallback" if args.fallback else None)

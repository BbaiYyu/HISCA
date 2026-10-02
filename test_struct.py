# -*- coding: utf-8 -*-
"""可行性测试: RNA二级结构特征能否让区域区分度从(k-mer)0.52再降。
对比 区域相似度: k-mer-only / struct-only / kmer+struct。
"""
import os, sys, csv, subprocess, itertools
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
csv.field_size_limit(10**8)
RNAFOLD = "C:/Program Files (x86)/ViennaRNA Package/RNAfold.exe"
NREG = 32
BASES = "AUGC"
KMERS = {"".join(p): i for i, p in enumerate(itertools.product(BASES, repeat=3))}  # 64


def rd(p):
    with open(p, encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def clean(seq):
    return "".join(c for c in str(seq).upper().replace("T", "U") if c in BASES)


def fold(seq):
    """调 RNAfold 得点括号结构。长序列截断到1500(测试用)。"""
    seq = seq[:1500]
    try:
        out = subprocess.run([RNAFOLD, "--noPS"], input=seq+"\n", capture_output=True, text=True, timeout=60)
        lines = out.stdout.strip().splitlines()
        if len(lines) >= 2:
            return lines[1].split()[0]  # 点括号
    except Exception:
        pass
    return "." * len(seq)


def pool(mat):  # [L,D] -> [NREG,D]
    x = torch.tensor(mat, dtype=torch.float32).t().unsqueeze(0)  # [1,D,L]
    return F.adaptive_avg_pool1d(x, min(NREG, mat.shape[0])).squeeze(0).t()  # [nreg,D]


def kmer_feat(seq):
    L = len(seq) - 2
    m = np.zeros((max(L, 1), 64), dtype=np.float32)
    for i in range(L):
        j = KMERS.get(seq[i:i+3])
        if j is not None: m[i, j] = 1
    return m


def struct_feat(seq, db):
    """每位置结构特征: [是否配对, '(', ')', '.', 局部GC配对倾向] 等 -> 用更丰富的窗口结构熵"""
    L = len(seq)
    db = (db + "." * L)[:L]
    feat = np.zeros((L, 6), dtype=np.float32)
    for i in range(L):
        c = db[i]
        feat[i, 0] = 1.0 if c in "()" else 0.0      # 配对
        feat[i, 1] = 1.0 if c == "(" else 0.0
        feat[i, 2] = 1.0 if c == ")" else 0.0
        feat[i, 3] = 1.0 if c == "." else 0.0
        # 局部窗口配对密度(±5)
        w = db[max(0, i-5):i+6]
        feat[i, 4] = sum(ch in "()" for ch in w) / max(len(w), 1)
        feat[i, 5] = float(seq[i] in "GC")           # GC含量(结构相关)
    return feat


def region_sim(reg):
    xn = F.normalize(reg, dim=1); S = xn @ xn.t(); k = reg.shape[0]
    if k < 2: return None
    iu = torch.triu_indices(k, k, 1); return S[iu[0], iu[1]].mean().item()


def main():
    smap = {r["ncRNA_Name"]: r.get("Sequence", "") for r in rd(C.NCRNA_CSV)}
    names = [n for n in smap if len(clean(smap[n])) >= 30][:40]
    sk, ss, sks = [], [], []
    for n in names:
        seq = clean(smap[n])
        db = fold(seq)[:len(seq)]
        km = pool(kmer_feat(seq))                    # 区域kmer
        st = pool(struct_feat(seq, db))              # 区域结构
        both = torch.cat([F.normalize(km, dim=1), F.normalize(st, dim=1)], dim=1)
        for lst, reg in [(sk, km), (ss, st), (sks, both)]:
            v = region_sim(reg)
            if v is not None: lst.append(v)
    print(f"样本数: {len(sk)}")
    print(f"k-mer only     区域相似度: {np.mean(sk):.3f}")
    print(f"struct only    区域相似度: {np.mean(ss):.3f}")
    print(f"kmer + struct  区域相似度: {np.mean(sks):.3f}  (目标: 比0.52更低更好)")


if __name__ == "__main__":
    main()

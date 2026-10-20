

import os, sys, csv, json, argparse, time, urllib.request
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(C.FEAT_DIR, "target_feats.pt")
DRUG_TGT = os.path.join(C.DATA_DIR, "drug_targets.csv")
REF = os.path.join(C.DATA_DIR, "..", "原始数据库", "ref")
SEQ_CACHE = os.path.join(C.FEAT_DIR, "_uniprot_seq_cache.json")


def find_ref(name):

    for d in (os.path.join(C.DATA_DIR, "ref"), REF):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def fetch_sequences(accs):

    cache = {}
    if os.path.exists(SEQ_CACHE):
        cache = json.load(open(SEQ_CACHE, encoding="utf-8"))
    todo = [a for a in accs if a not in cache]
    print(f"序列：已缓存 {len(cache)}，待抓 {len(todo)}", flush=True)
    for i, a in enumerate(todo):
        seq = ""
        for _ in range(3):
            try:
                url = f"https://rest.uniprot.org/uniprotkb/{a}.fasta"
                txt = urllib.request.urlopen(url, timeout=30).read().decode()
                seq = "".join(txt.splitlines()[1:]) if txt.startswith(">") else ""
                break
            except Exception:
                time.sleep(2)
        cache[a] = seq
        if (i + 1) % 50 == 0:
            json.dump(cache, open(SEQ_CACHE, "w", encoding="utf-8"))
            print(f"  {i+1}/{len(todo)} 抓完", flush=True)
        time.sleep(0.1)
    json.dump(cache, open(SEQ_CACHE, "w", encoding="utf-8"))
    got = sum(1 for a in accs if cache.get(a))
    print(f"序列抓取完成：{got}/{len(accs)} 有序列", flush=True)
    return cache


def esm_embed(seqs, model_name, device, max_len=1022, batch=8):

    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = EsmModel.from_pretrained(model_name).to(device).eval()
    d = mdl.config.hidden_size
    print(f"ESM {model_name} 加载完成，维度={d}", flush=True)
    items = [(a, s[:max_len]) for a, s in seqs.items() if s]
    out = {}
    with torch.no_grad():
        for b in range(0, len(items), batch):
            chunk = items[b:b + batch]
            enc = tok([s for _, s in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_len + 2).to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()

            vec = (h * m).sum(1) / m.sum(1).clamp(min=1)
            for (a, _), v in zip(chunk, vec.cpu()):
                out[a] = v
            if (b + batch) % 80 == 0:
                print(f"  ESM {b+batch}/{len(items)}", flush=True)
    return out, d


def build_roles(id2uni):

    u2g = {}
    for r in csv.DictReader(open(DRUG_TGT, encoding="utf-8-sig")):
        up, g = r.get("Target_UniProt", "").strip(), r.get("Target_Gene", "").strip()
        for a, b in zip(up.split(";"), g.split(";")):
            if a.strip() and b.strip():
                u2g[a.strip()] = b.strip().upper()
    onco, tsg = set(), set()
    okb = find_ref("oncokb.tsv")
    if okb:
        for r in csv.DictReader(open(okb, encoding="utf-8", errors="ignore"), delimiter="\t"):
            g, t = r["Hugo Symbol"].upper(), r["Gene Type"].strip()
            if "ONCOGENE" in t:
                onco.add(g)
            if "TSG" in t:
                tsg.add(g)
    tsf = find_ref("tsgene.txt")
    if tsf:
        for r in csv.DictReader(open(tsf, encoding="utf-8", errors="ignore"), delimiter="\t"):
            tsg.add(r["GeneSymbol"].upper())
    roles = {}
    for tid, uni in id2uni.items():
        g = u2g.get(uni)
        roles[tid] = torch.tensor([1.0 if g in onco else 0.0,
                                   1.0 if g in tsg else 0.0])
    ncov = sum(1 for v in roles.values() if v.sum() > 0)
    print(f"角色：{ncov}/{len(roles)} 靶点有 onco/TSG 标注", flush=True)
    return roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", default="facebook/esm2_t12_35M_UR50D",
                    help="ESM 模型；更强用 facebook/esm2_t30_150M_UR50D(640维)")
    args = ap.parse_args()
    device = C.DEVICE

    t = torch.load(FEAT)
    vocab = t["vocab"]
    num = t["num_targets"]
    id2uni = {int(i): str(u) for u, i in vocab.items()}
    accs = sorted(set(id2uni.values()))
    print(f"靶点总数={num}  唯一UniProt={len(accs)}", flush=True)

    seqs = fetch_sequences(accs)
    emb, d_esm = esm_embed(seqs, args.esm, device)
    roles = build_roles(id2uni)
    d_role = 2
    d_feat = d_esm + d_role

    feat = torch.zeros(num + 1, d_feat)
    miss = 0
    for tid in range(1, num + 1):
        uni = id2uni.get(tid)
        v = emb.get(uni)
        if v is None:
            miss += 1
        else:
            feat[tid, :d_esm] = v
        feat[tid, d_esm:] = roles.get(tid, torch.zeros(d_role))
    print(f"无ESM序列(用零向量兜底)的靶点：{miss}/{num}", flush=True)

    t["feat"] = feat
    t["d_feat"] = d_feat
    t["esm_model"] = args.esm
    torch.save(t, FEAT)
    print(f"已写回 {FEAT}：feat {tuple(feat.shape)} (ESM {d_esm} + 角色 {d_role})", flush=True)


if __name__ == "__main__":
    main()



import os, sys, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def split_positives(n_pos, test_ratio=0.2, seed=C.SEED):

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_pos, generator=g)
    n_test = int(n_pos * test_ratio)
    test_mask = torch.zeros(n_pos, dtype=torch.bool); test_mask[perm[:n_test]] = True
    return ~test_mask, test_mask


def kfold_split(n_pos, k=C.K_FOLD, seed=C.SEED):

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_pos, generator=g)
    folds = torch.chunk(perm, k)
    splits = []
    for i in range(k):
        test_mask = torch.zeros(n_pos, dtype=torch.bool)
        test_mask[folds[i]] = True
        splits.append((~test_mask, test_mask))
    return splits


def split_train_val(train_mask, val_frac=0.15, seed=C.SEED):

    idx = torch.nonzero(train_mask).squeeze(1)
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(len(idx), generator=g)]
    n_val = int(len(idx) * val_frac)
    val_mask = torch.zeros_like(train_mask); val_mask[perm[:n_val]] = True
    inner_mask = train_mask & ~val_mask
    return inner_mask, val_mask


def sample_negatives(pos_set, n_rna, n_drug, n_neg, rng, rna_pool=None, drug_pool=None, hard_frac=None):


    negs_r, negs_d = [], []
    has_pool = rna_pool is not None
    if hard_frac is None:
        hard_frac = 1.0 if has_pool else 0.0
    while len(negs_r) < n_neg:
        use_hard = has_pool and (rng.random() < hard_frac)
        if use_hard:
            r = rna_pool[rng.randint(0, len(rna_pool) - 1)]
            d = drug_pool[rng.randint(0, len(drug_pool) - 1)]
        else:
            r = rng.randint(0, n_rna - 1); d = rng.randint(0, n_drug - 1)
        if (r, d) not in pos_set:
            negs_r.append(r); negs_d.append(d)
    return torch.tensor(negs_r), torch.tensor(negs_d)


def build_neg_pools(pos_r, pos_d):


    return pos_r.tolist(), pos_d.tolist()


def _bulk_unlabeled(n, N_rna, N_drug, pos_set, g, over=1.4):

    got_r, got_d = [], []
    while len(got_r) < n:
        m = int((n - len(got_r)) * over) + 32
        rr = g.randint(0, N_rna, m); dd = g.randint(0, N_drug, m)
        for a, b in zip(rr.tolist(), dd.tolist()):
            if (a, b) not in pos_set:
                got_r.append(a); got_d.append(b)
                if len(got_r) >= n:
                    break
    return np.array(got_r[:n]), np.array(got_d[:n])


def build_rf_reliable_pool(data, train_mask, n_pool, seed, cand_mult=5,
                           n_estimators=150, min_leaf=5):


    from sklearn.ensemble import RandomForestClassifier
    g = np.random.RandomState(seed)
    rna = data.rna_node.cpu().numpy()
    drg = data.drug_node.cpu().numpy()
    all_pos = set(zip(data.pos.r.tolist(), data.pos.d.tolist()))

    tr_r = data.pos.r[train_mask].numpy(); tr_d = data.pos.d[train_mask].numpy()
    n_pos = len(tr_r)
    Xpos = np.concatenate([rna[tr_r], drg[tr_d]], axis=1)
    ur, ud = _bulk_unlabeled(n_pos, data.N_rna, data.N_drug, all_pos, g)
    Xneg = np.concatenate([rna[ur], drg[ud]], axis=1)
    X = np.concatenate([Xpos, Xneg], 0)
    y = np.concatenate([np.ones(n_pos), np.zeros(len(ur))])

    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_leaf,
                                max_features="sqrt", n_jobs=-1, random_state=seed)
    rf.fit(X, y)

    n_cand = min(n_pool * cand_mult, data.N_rna * data.N_drug - len(all_pos))
    cr, cd = _bulk_unlabeled(n_cand, data.N_rna, data.N_drug, all_pos, g)
    Xc = np.concatenate([rna[cr], drg[cd]], axis=1)
    p = rf.predict_proba(Xc)[:, 1]
    order = np.argsort(p)[:n_pool]
    print(f"    [RF可靠负] 正{n_pos} 候选{n_cand} → 选{len(order)}条; "
          f"候选P(关联) 均值{p.mean():.3f} 选中均值{p[order].mean():.3f}", flush=True)
    return torch.tensor(cr[order], dtype=torch.long), torch.tensor(cd[order], dtype=torch.long)


def sample_from_pairpool(pool_r, pool_d, n_neg, rng):

    n = len(pool_r)
    if n == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    idx = torch.tensor([rng.randint(0, n - 1) for _ in range(n_neg)])
    return pool_r[idx], pool_d[idx]


def coldstart_split(pos_r, pos_d, n_rna, n_drug, test_frac=0.2, seed=C.SEED, by="ncRNA"):

    g = np.random.RandomState(seed)
    ids = (pos_r.numpy() if by == "ncRNA" else pos_d.numpy())
    n = (n_rna if by == "ncRNA" else n_drug)
    held = set(g.choice(n, int(n * test_frac), replace=False).tolist())
    test_mask = torch.tensor([int(i) in held for i in ids])
    return ~test_mask, test_mask


def make_batch(data, r_ids, d_ids, device):

    B = r_ids.shape[0]
    Hr = data.rna_feat[r_ids].to(device)
    batch = {
        "Hr": Hr,
        "Hr_mask": torch.ones(B, Hr.shape[1], dtype=torch.bool, device=device),
        "Hd": data.drug_feat[d_ids].to(device),
        "Hd_mask": data.drug_mask[d_ids].to(device),
        "tgt_ids": data.tgt_ids[d_ids].to(device),
        "tgt_mask": data.tgt_mask[d_ids].to(device),
        "rna_nid": r_ids.to(device),
        "drug_nid": d_ids.to(device),
    }
    if getattr(data, "rna_kmer", None) is not None:
        batch["Hr_kmer"] = data.rna_kmer[r_ids].to(device)
    return batch


def iterate_minibatches(n, batch_size, shuffle=True, seed=0):
    idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    for b in range(0, n, batch_size):
        yield idx[b:b + batch_size]


def eval_metrics(assoc_score, assoc_label, rel_logits=None, rel_label=None):

    import numpy as np
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 accuracy_score, precision_score, recall_score,
                                 matthews_corrcoef, confusion_matrix)
    s = np.asarray(assoc_score); y = np.asarray(assoc_label).astype(int)
    pred = (s >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {}
    out["AUC"]    = roc_auc_score(y, s)
    out["AUPR"]   = average_precision_score(y, s)
    out["ACC"]    = accuracy_score(y, pred)
    out["Prec"]   = precision_score(y, pred, zero_division=0)
    out["Recall"] = recall_score(y, pred, zero_division=0)
    out["Spec"]   = tn / (tn + fp) if (tn + fp) else 0.0
    out["F1"]     = f1_score(y, pred, zero_division=0)
    out["MCC"]    = matthews_corrcoef(y, pred)
    if rel_logits is not None and len(rel_label) > 0:
        rp = rel_logits.argmax(1)
        out["relACC"] = accuracy_score(rel_label, rp)
        out["relF1"]  = f1_score(rel_label, rp, average="macro", zero_division=0)
    return out

"""Token groups for the local web explanation plot (heuristic, not atom attribution)."""
import os
from functools import lru_cache
import config as C

@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.environ.get("NCDRUG_TOKENIZER_DIR", C.CHEMBERTA_MODEL))

def drug_tokens(smiles):
    tok = _tokenizer()
    ids = tok(smiles, truncation=True, max_length=C.LD_MAX)["input_ids"]
    return tok.convert_ids_to_tokens(ids)

def classify_tokens(toks):
    """启发式: 把 ChemBERTa(≈SMILES字符)token 归到化学子结构组(不依赖RDKit)。"""
    g = []
    for i, t in enumerate(toks):
        s = str(t).replace("Ġ", "").strip()
        prv = str(toks[i-1]).strip() if i > 0 else ""
        nxt = str(toks[i+1]).strip() if i+1 < len(toks) else ""
        if s in ("[CLS]", "[SEP]", "<s>", "</s>", ""):
            g.append("special")
        elif s in ("F", "Cl", "Br", "I"):
            g.append("Halogen (F)")
        elif s in ("N", "n"):
            g.append("Amino / ring-N")
        elif s in ("O", "o"):
            g.append("Carbonyl" if ("=" in (prv, nxt)) else "Hydroxyl / O")
        elif s in ("C", "c"):
            g.append("Ring C" if (s == "c" or prv.isdigit() or nxt.isdigit()) else "Aliphatic C")
        elif s in ("P", "S", "s", "p"):
            g.append("Other heteroatom")
        else:
            g.append(None)                     # 键/支链/环闭合符 → 归前一原子的组
    last = "Aliphatic C"
    for i in range(len(g)):
        if g[i] is None:
            g[i] = last
        elif g[i] != "special":
            last = g[i]
    return g

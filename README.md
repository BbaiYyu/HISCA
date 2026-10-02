# HISCA

**Heterogeneous-graph and Interpretable Substructure Co-Attention** for ncRNA–drug association prediction.

HISCA is a general ncRNA–drug association model that unifies miRNAs, lncRNAs and circRNAs in a single framework and produces an interpretable region–substructure interaction map. It combines:

- an **ncRNA encoder** — a pre-trained RNA language model (RNA-FM) enriched with *k*-mer composition features and adapted to the ncRNA subtype by a type-conditioned FiLM layer;
- a **multi-view drug encoder** — a molecular language model (ChemBERTa), an atom-level graph network, and an ECFP fingerprint;
- a **multi-head bilinear co-attention** module that forms an explicit affinity map between ncRNA sequence regions and drug substructures;
- a **heterogeneous graph transformer (HGT)** branch over the ncRNA–drug association graph.

## Installation

```bash
conda create -n hisca python=3.10
conda activate hisca
pip install -r requirements.txt
```

Tested with PyTorch 2.x + PyTorch Geometric. A CUDA GPU is recommended.

## Data

Place the following tables under `data/` (not distributed here; build them from the public databases cited in the paper):

| File | Columns |
|---|---|
| `ncRNA.csv` | `ncRNA_Name, ncRNA_Type, Sequence` |
| `drugs.csv` | `Drug_Name, SMILES` |
| `merged_clean.csv` | `ncRNA_Name, ncRNA_Type, Drug_Name, rel` |
| `drug_targets.csv` | drug → protein target (optional) |

Paths are resolved relative to the repository and can be overridden with the
`NCDRUG_DATA_DIR`, `NCDRUG_FEAT_DIR` and `NCDRUG_CKPT_DIR` environment variables.

## Usage

**1. Pre-compute features** (RNA-FM / ChemBERTa embeddings, *k*-mer, fingerprints, graphs):

```bash
python encoders.py              # rna_feats.pt / drug_feats.pt
python build_kmer_feats.py      # rna_kmer.pt
python build_drug_extra.py      # ECFP fingerprints
python build_drug_graphs.py     # atom graphs
python build_target_features.py # ESM target features (optional)
```

**2. Train and evaluate** (5-fold cross-validation):

```bash
python main.py --arch v2 --neg degree --folds 5
```

Useful flags: `--ablation {none,no_film,no_hyper,gcn,no_relhead,no_graph}`,
`--neg {random,degree,rf}` (training negatives), `--evalneg {random,degree}`,
`--coldstart {off,ncRNA,drug}`, `--tgtfeat {real,learnable}`.

**3. Leave-one-drug-out case study**:

```bash
NCDRUG_DMODEL=256 NCDRUG_LR=1e-3 NCDRUG_DROPOUT=0.2 python case_study_loo.py
```

## Repository layout

```
config.py      global paths and hyper-parameters
encoders.py    RNA-FM / ChemBERTa feature pre-computation
layers.py      model building blocks (FiLM, interaction, HGT, prediction head)
model.py       the full HISCA model
dataset.py     data assembly and heterogeneous-graph construction
utils.py       training utilities, negative sampling, metrics
main.py        training / evaluation entry point
build_*.py     feature-construction scripts
case_study_loo.py   leave-one-drug-out case study
run_*.sh       experiment driver scripts
```

## License

Released under the MIT License — see [LICENSE](LICENSE).

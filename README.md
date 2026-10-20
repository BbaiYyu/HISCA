# HISCA

**Heterogeneous-graph and Interpretable Substructure Co-Attention** for ncRNA–drug association prediction.

HISCA is a general ncRNA–drug association model that unifies miRNAs, lncRNAs and circRNAs in a single framework and produces an interpretable region–substructure interaction map. It combines:

- an **ncRNA encoder** — a pre-trained RNA language model (RNA-FM) enriched with *k*-mer composition features and adapted to the ncRNA subtype by a type-conditioned FiLM layer;
- a **multi-view drug encoder** — a molecular language model (ChemBERTa), an atom-level graph network, and an ECFP fingerprint;
- a **multi-head bilinear co-attention** module that forms an explicit affinity map between ncRNA sequence regions and drug substructures;
- a **heterogeneous graph transformer (HGT)** branch over the ncRNA–drug association graph.


## Notebook and one-click startup

Open **[HISCA_quickstart.ipynb](HISCA_quickstart.ipynb)** in Jupyter and choose
**Run All Cells** to check deployment files, prepare dependencies, and launch the
local web platform. The notebook embeds the live app and also provides a browser link.

On Windows with Python 3.9–3.12 installed:
- Double-click **one_click_start.bat** to prepare an isolated environment and open the web app.
- Double-click **open_notebook.bat** to prepare JupyterLab and open the notebook.

First installation needs Internet. Later runs reuse the project environment.
Use a complete matching deployment bundle; large model/data files are not in Git.
The first-install CPU path is tested on Windows Python 3.9.23.
See **[Notebook 中文使用说明](docs/NOTEBOOK_QUICKSTART_zh.md)**.
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

## Interactive local web platform

The Gradio platform includes a scientific home page, pair prediction, region ×
substructure explanation maps, and drug-based Top-K ranking. Home-page buttons
open the corresponding analysis tabs.

**[中文本地部署指南 / Local deployment guide](docs/LOCAL_DEPLOYMENT_zh.md)**

~~~bash
# Activate Python 3.10 and install the appropriate Torch wheel first.
python -m pip install -r requirements-web.txt
python check_deployment.py
python app.py
# Open http://127.0.0.1:7860
~~~

Windows: run start_local.bat in an activated environment.
Linux/macOS: run sh start_local.sh.

**Data, precomputed features, and weights are not bundled, and no public deployment
bundle download is currently provided.** Obtain matching files from the maintainer
or prepare them with the training pipeline. The app loads ckpt/hisca_demo.pt;
missing/incompatible weights stop startup rather than silently retraining.

Override paths using NCDRUG_DATA_DIR, NCDRUG_FEAT_DIR, NCDRUG_CKPT_DIR or
NCDRUG_DEMO_CKPT. NCDRUG_PORT changes the local port. Inference runs locally;
explanation maps may download the ChemBERTa tokenizer on first use.
For offline use, prepare NCDRUG_TOKENIZER_DIR as described in the guide.

The cover is an AI-generated conceptual illustration, not experimental data.
Predictions and relative interaction maps support research hypotheses, not
validated biological mechanisms.


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
make_heatmap.py     region x substructure interaction-map figures
app.py         interactive Gradio web demo
run_*.sh       experiment driver scripts
```

## License

Released under the MIT License — see [LICENSE](LICENSE).

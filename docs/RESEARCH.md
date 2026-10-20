# Source, feature preparation and training

The installable platform is intended for inference with the released matching bundle.
This page describes the separate source-code research path.

## Environment

Use Python 3.9–3.12 in a dedicated environment. Install a suitable PyTorch build,
then the research requirements:

```bash
python -m pip install -r requirements.txt
```

`requirements-web.txt` contains the pinned inference stack used by the launcher.
Research feature preparation additionally uses the dependencies in `requirements.txt`
and may download pretrained encoder weights. CPU inference does not require those
encoder downloads because feature tensors are precomputed.

## Data and paths

Place the data tables in `data/`, feature tensors in `features/`, and weights in `ckpt/`.
`NCDRUG_DATA_DIR`, `NCDRUG_FEAT_DIR`, `NCDRUG_CKPT_DIR`, and `NCDRUG_EDGE_CSV`
can point to alternative locations.

| Table | Main columns |
|---|---|
| `ncRNA.csv` | `ncRNA_Name`, `ncRNA_Type`, `Sequence` |
| `drugs.csv` | `Drug_Name`, `SMILES` |
| `merged_clean.csv` | `ncRNA_Name`, `ncRNA_Type`, `Drug_Name`, `rel` |
| `drug_targets.csv` | Drug and protein-target mapping |

Feature preparation entry points:

```bash
python encoders.py
python build_kmer_feats.py
python build_drug_extra.py
python build_drug_graphs.py
python build_target_features.py
```

The released runtime requires `rna_feats.pt`, `drug_feats.pt`, `target_feats.pt`,
`rna_kmer.pt`, `drug_extra.pt`, and `drug_graphs.pt` in a matching node order.
The application checks checkpoint shapes and model loading strictly.

## Training and evaluation

```bash
python main.py --help
python main.py --arch v2 --neg degree --evalneg random --folds 5
```

The latest training entry uses random evaluation negatives by default; specify
`--evalneg` explicitly when reproducing an experiment. Training and Web inference
serve different purposes: the released Web service computes predictions using the
released complete known-association graph and is not an independent test-set evaluation.

## Optional high-order hyperedges

`build_hyperedges_v2.py` is the current optional construction entry. It requires
`data/ncrna_targets.json`, the matched base features, RDKit for scaffold grouping,
and a ChEMBL target cache. Set `NCDRUG_CHEMBL_CACHE` to that cache JSON; otherwise
it uses `data/chembl_target_cache.json`. `NCDRUG_HYPER_FIGURE` selects the output figure
path; by default it is `fig/hyperedge_stats.png` next to the data directory.

The resulting `data/hyperedges_v2.pt` can be enabled in matched research runs with
`NCDRUG_V2_HYPER=1`. It is disabled in the released Web profile. Do not change
architecture or interaction-normalization settings while assuming a different
checkpoint has matching inference semantics.

## Building distributions

```powershell
python -m pip install --upgrade setuptools wheel
python setup.py build --build-base .runtime/build-lite bdist_wheel --dist-dir dist/lite
$env:HISCA_BUNDLE_SOURCE = "完整部署数据目录"
python setup.py build --build-base .runtime/build-full bdist_wheel --dist-dir dist/full
```

Use separate lite/full build directories. The full build validates all required bundle
files. Model data do not enter the Git working tree or the lightweight source distribution.

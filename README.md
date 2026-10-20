# HISCA

Heterogeneous-graph and Interpretable Substructure Co-Attention for ncRNA–drug association prediction.

HISCA provides a local research platform with a home page, ncRNA–drug pair prediction,
region–substructure interaction heatmaps, and drug-based candidate ranking.
The model source and the installable platform are maintained in this repository.

## 安装并启动 / Install and start

安装 **64 位 Python 3.9–3.12** 后，打开终端执行：

```powershell
python -m pip install "https://github.com/BbaiYyu/HISCA/releases/download/v0.3.1/hisca_web-0.3.1%2Bfull.20261020-py3-none-any.whl"
hisca
```

首次运行会自动创建独立 CPU 环境、安装运行依赖、加载模型，并在准备完成后打开浏览器。
以后只需运行 `hisca`；命令找不到时，使用 `python -m hisca`。
不需要 Jupyter、Conda 或 GPU。首次安装依赖需要联网；当前已在 Windows Python 3.9.23 CPU 环境实测。

- [下载完整安装包（wheel，包含模型和数据）](https://github.com/BbaiYyu/HISCA/releases/download/v0.3.1/hisca_web-0.3.1+full.20261020-py3-none-any.whl)
- [中文安装与使用说明](docs/INSTALL_zh.md)
- [所有发布附件与校验值](https://github.com/BbaiYyu/HISCA/releases/tag/v0.3.1)

完整 `.whl` 就是安装包。GitHub 自动生成的 **Source code (zip)** 只含源码，不含预测模型和特征。
本项目尚未发布到 PyPI；请使用上面的安装地址，不要直接运行 `pip install hisca`。

## Web platform

The browser interface includes:

- Home page and model framework illustration.
- Prediction for an existing ncRNA–drug pair, with an association score and a relative interaction heatmap.
- Top-K ncRNA ranking for a selected drug, with ncRNA type filtering.

Keep the startup terminal open. Ctrl+C stops its local service. The default address
is on `127.0.0.1`; an available local port is selected automatically. Inference uses
the released trained weights and precomputed features. No retraining is performed at startup.

## Source and model

The model consumes RNA-FM and ChemBERTa precomputed representations, RNA k-mer features,
drug fingerprints and molecular graphs, together with heterogeneous graph information.
The latest local model implementation is included. Model-code comments and docstrings
have been removed; code behavior is checked separately from this formatting change.

The packaged Web profile uses `ckpt/hisca_demo.pt`, the matching six feature files and
local tokenizer. It fixes `NCDRUG_V2_HYPER=0`, `NCDRUG_INT_CENTER=0`, and
`NCDRUG_INT_NORM=0` to preserve the released checkpoint's inference settings.
The optional high-order hypergraph construction code is available for research;
it is not enabled by this deployment checkpoint.

| Path | Purpose |
|---|---|
| `app.py`, `home.html`, `assets/` | Gradio interface and page assets |
| `src/hisca/` | Installable launcher and model-bundle import |
| `config.py`, `dataset.py` | Model configuration and data assembly |
| `model.py`, `layers.py` | Model and neural-network components |
| `main.py`, `utils.py` | Training, evaluation and sampling |
| `encoders.py`, `build_*.py` | Feature and optional hyperedge preparation |
| `requirements-web.txt` | Tested inference dependency versions |
| `requirements.txt` | Research / feature-preparation dependencies |
| `docs/` | Installation, research and release notes |
| `tests/` | Launcher regression tests |

See [research setup and training](docs/RESEARCH.md) for source use. Superseded Notebook
launchers and machine-specific experiment wrappers have been removed from the current tree.
Earlier source remains in Git history.

The cover is a conceptual illustration. Predicted scores and relative interaction
maps are research outputs and should be interpreted with the model's evaluation setting.

## License

Source code is released under the [MIT License](LICENSE).

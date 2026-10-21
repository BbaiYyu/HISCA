# HISCA 源码开发说明

本页供修改平台代码、构建特征、训练模型或维护安装包时使用。普通使用者请阅读 [安装与使用说明](INSTALL_zh.md)，直接安装 **v0.3.1 稳定完整包**。

## 1. 目录与入口

以下命令均从仓库根目录执行。

| 路径 | 作用 |
|---|---|
| [models/main.py](../models/main.py) | 训练与评估入口 |
| `models/config.py` | 路径和模型配置 |
| `models/dataset.py`、`models/utils.py` | 数据组装、图构建、采样与评估工具 |
| `models/model.py`、`models/layers.py` | 模型及神经网络组件 |
| [web/app.py](../web/app.py) | 源码方式启动 Gradio 平台 |
| [web/check_deployment.py](../web/check_deployment.py) | 不加载模型的部署文件检查 |
| `web/home.html`、`web/assets/` | 首页与图片 |
| `preprocessing/` | 编码特征、分子图与可选超边构建 |
| `src/hisca/` | 安装后的命令行启动器、环境准备和部署包导入 |
| `requirements/web.txt` | 已验证的网页推理依赖 |
| `requirements/research.txt` | 特征构建与研究依赖 |
| `tests/` | 启动器回归检查 |

## 2. 获取源码与准备环境

~~~powershell
git clone https://github.com/BbaiYyu/HISCA.git
cd HISCA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

使用 64 位 Python 3.9–3.12。Linux/macOS 的环境激活方式为 `source .venv/bin/activate`；当前完整推理流程的实测平台为 Windows、Python 3.9.23、CPU。

运行源码网页时，先安装 CPU 版 PyTorch，再安装推理依赖：

~~~powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/web.txt
~~~

需要重新构建特征或开展训练时，另安装研究依赖：

~~~powershell
python -m pip install -r requirements/research.txt
~~~

特征构建可能下载 RNA-FM、ChemBERTa 或蛋白编码器权重；使用已有完整部署特征进行网页推理时不需要下载这些编码器。研究实验应记录实际依赖版本和运行参数。

## 3. 部署数据与路径

Git 仓库不存放大模型、数据表和预计算特征。源码运行需要准备与模型匹配的部署文件。默认目录位于仓库根目录：

~~~text
HISCA/
├── data/
│   ├── drugs.csv
│   ├── merged_clean.csv
│   ├── ncRNA.csv
│   └── drug_targets.csv
├── features/
│   ├── rna_feats.pt
│   ├── drug_feats.pt
│   ├── target_feats.pt
│   ├── rna_kmer.pt
│   ├── drug_extra.pt
│   └── drug_graphs.pt
├── ckpt/
│   └── hisca_demo.pt
└── tokenizer/
~~~

网页使用 `drugs.csv`、关联表、匹配的特征和权重；序列与靶点表也用于研究特征构建。六份特征必须使用匹配的名称映射和节点顺序。

| 数据表 | 主要字段 |
|---|---|
| `ncRNA.csv` | `ncRNA_Name`、`ncRNA_Type`、`Sequence` |
| `drugs.csv` | `Drug_Name`、`SMILES` |
| `merged_clean.csv` | `ncRNA_Name`、`ncRNA_Type`、`Drug_Name`、`rel` |
| `drug_targets.csv` | 药物与蛋白靶点映射 |

文件在其他目录时，可以使用环境变量。PowerShell 示例：

~~~powershell
$env:NCDRUG_DATA_DIR = "D:\HISCA部署数据\data"
$env:NCDRUG_FEAT_DIR = "D:\HISCA部署数据\features"
$env:NCDRUG_CKPT_DIR = "D:\HISCA部署数据\ckpt"
$env:NCDRUG_TOKENIZER_DIR = "D:\HISCA部署数据\tokenizer"
$env:HF_HUB_OFFLINE = "1"
~~~

`NCDRUG_EDGE_CSV` 可单独指定关联表，`NCDRUG_DEMO_CKPT` 可指定网页加载的权重文件。其他系统使用相应的环境变量设置方式。

## 4. 运行源码网页

准备依赖和部署文件后，先显式指定本地 tokenizer。即使完整 tokenizer 已放在仓库根目录的 `tokenizer/` 中，源码入口也不会自动选择它。

在仓库根目录中执行以下 PowerShell 命令。离线运行还需要设置 `HF_HUB_OFFLINE`：

~~~powershell
$env:NCDRUG_TOKENIZER_DIR = (Resolve-Path .\tokenizer).Path
$env:HF_HUB_OFFLINE = "1"
~~~

随后检查文件并启动网页：

~~~powershell
python web/check_deployment.py
python web/app.py
~~~

检查入口先确认部署文件是否存在；权重与模型结构是否匹配，由实际模型加载进一步验证。浏览器访问启动终端给出的本地地址，按 `Ctrl+C` 停止。

源代码启动读取当前工作副本。修改网页或模型后，使用这里的入口验证自己的改动。

## 5. 特征构建与训练

预计算入口：

~~~powershell
python preprocessing/encoders.py
python preprocessing/build_kmer_feats.py
python preprocessing/build_drug_extra.py
python preprocessing/build_drug_graphs.py
python preprocessing/build_target_features.py
~~~

这些脚本依赖相应的数据表、预训练编码器或外部资料；应按各脚本实际输入准备文件。`preprocessing/build_ncrna_targets.py` 用于准备 ncRNA 靶点资料。

训练与评估：

~~~powershell
python models/main.py --help
python models/main.py --arch v2 --neg degree --folds 5
~~~

使用 RF 可靠负样本时，可以明确选择评估负样本：

~~~powershell
python models/main.py --arch v2 --neg rf --evalneg degree --folds 5
~~~

当前实现中，`--evalneg` 仅在 `--neg rf` 时单独决定评估负样本；其他模式与训练负样本模式一致。训练评估与网页部署是两个流程：网页加载已有的 `hisca_demo.pt`，不会在启动时训练或自动生成部署包。

网页使用完整已知关联图来提供预测服务，不能把这个部署过程当作独立测试集上的评估。

## 6. 可选高阶超边

~~~powershell
python preprocessing/build_hyperedges_v2.py
~~~

该入口需要匹配的基础特征、`data/ncrna_targets.json`、用于骨架分组的 RDKit，以及 ChEMBL 靶点缓存。通过 `NCDRUG_CHEMBL_CACHE` 指定缓存 JSON；默认读取 `data/chembl_target_cache.json`。`NCDRUG_HYPER_FIGURE` 可指定输出统计图路径。

构建结果为 `data/hyperedges_v2.pt`。在匹配的研究运行中，通过 `NCDRUG_V2_HYPER=1` 启用。

稳定网页权重的设置保持为 `NCDRUG_V2_HYPER=0`、`NCDRUG_INT_CENTER=0`、`NCDRUG_INT_NORM=0`。开关不会完整记录在权重文件中；即使张量形状能加载，也应保持训练与推理设置一致。

## 7. 安装源码启动器

在仓库根目录安装：

~~~powershell
python -m pip install .
hisca --bundle "完整部署目录或ZIP"
~~~

这会安装轻量启动器，模型文件另备。`--bundle` 接受包含 `data/`、`features/`、`ckpt/`、`tokenizer/` 的目录，或包含同一套文件的 ZIP；首次选择后会记录路径。

已安装的 `hisca` 使用安装时打包的源码；工作副本后续发生修改时，需要重新安装以更新该副本。调试当前工作副本时优先使用 `python web/app.py`。

完整 v0.3.1 wheel 已带模型，普通用户直接按 [安装说明](INSTALL_zh.md) 使用即可。

## 8. 维护者构建安装包

安装构建工具，并为轻量包与完整包使用不同构建目录：

~~~powershell
python -m pip install --upgrade setuptools wheel

$env:HISCA_BUNDLE_SOURCE = ""
python setup.py build --build-base .runtime/build-lite bdist_wheel --dist-dir dist/lite

$env:HISCA_BUNDLE_SOURCE = "D:\HISCA完整部署目录"
python setup.py build --build-base .runtime/build-full bdist_wheel --dist-dir dist/full
$env:HISCA_BUNDLE_SOURCE = ""
~~~

`HISCA_BUNDLE_SOURCE` 指向完整部署目录，其中应包含前述 `data/`、`features/`、`ckpt/`、`tokenizer/`。完整构建会检查所需部署文件。

源码按 `models/`、`web/`、`preprocessing/` 分类；安装包内部将推理代码整理到平铺的 `_runtime`，以维持启动兼容性。运行环境、日志和缓存写入可写工作目录，不写入模型资源目录。

返回 [仓库首页](../README.md)。

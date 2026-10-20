# HISCA 本地部署指南

本指南部署带封面首页、配对预测、解释热图和药物候选排序的本地 Gradio 平台。默认地址为 http://127.0.0.1:7860，仅供运行程序的本机访问。关闭程序后服务停止。

## 1. 部署前必须知道

**GitHub 仅包含代码、页面和展示图片，不包含数据、预计算特征及训练权重。仅克隆仓库不能进行真实预测。**

需要向维护者取得匹配的部署文件，或依照 README 自行构建特征和训练。当前仓库没有公开部署包下载地址。不要把其他模型的权重随意改名，也不要将占位特征用于正式预测。

~~~text
HISCA/
├── app.py
├── home.html
├── interpretation.py
├── check_deployment.py
├── requirements-web.txt
├── start_local.bat
├── start_local.sh
├── assets/
│   ├── cover_rna_drug.png
│   └── architecture_latest.png
├── data/
│   ├── drugs.csv
│   ├── merged_clean.csv
│   └── ncRNA.csv
├── features/
│   ├── rna_feats.pt
│   ├── drug_feats.pt
│   ├── target_feats.pt
│   ├── rna_kmer.pt
│   ├── drug_extra.pt
│   └── drug_graphs.pt
└── ckpt/
    └── hisca_demo.pt
~~~

- 启动必需：drugs.csv、merged_clean.csv、前三项特征和训练权重。
- ncRNA.csv 用于从原始序列构建特征；已有部署特征时，网页不直接读取它。
- 后三项特征在代码中可选，但必须与训练权重匹配。例如包含原子图分支的权重需要 drug_graphs.pt；缺少 k-mer 特征也可能改变结果。部署包应包含训练所用的全部特征。
- 数据名称、特征索引及顺序必须对应。网页使用 v2 架构，不能直接加载不同架构的权重。
- 缺少文件或权重不兼容时程序会停止，不会自动训练。

| 文件 | 列 |
|---|---|
| drugs.csv | Drug_Name, SMILES |
| merged_clean.csv | ncRNA_Name, ncRNA_Type, Drug_Name, rel |
| ncRNA.csv（构建特征） | ncRNA_Name, ncRNA_Type, Sequence |

关系标签为 resistant、sensitive；冲突关系按 config.py 处理。

## 2. 获取代码和创建环境

使用 Python 3.10。依赖版本来自本次验证环境，不表示最新版本。原 CUDA 环境实际 Python 为 3.9.23（此前记录为 3.10，已更正）；新增 Notebook 入口已通过 Windows 纯 CPU 干净环境验证；Linux/macOS 未在此次实机验证。

~~~bash
git clone https://github.com/BbaiYyu/HISCA.git
cd HISCA
conda create -n hisca python=3.10 -y
conda activate hisca
python -m pip install --upgrade pip
~~~

没有 Conda 时，可使用 Python 3.10 执行 python -m venv .venv。Windows PowerShell 激活命令为 .venv\Scripts\Activate.ps1；Linux/macOS 为 source .venv/bin/activate。

## 3. 安装依赖

先安装适合硬件的 PyTorch，再安装网页依赖。

Windows/Linux CPU 示例（无需 NVIDIA GPU，速度可能较慢）：

~~~bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-web.txt
~~~

本次验证的 GPU 组合是 PyTorch 2.6.0+cu124：

~~~bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-web.txt
~~~

CUDA wheel 需要兼容的 NVIDIA 驱动。其他硬件及 macOS 请参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 和 [历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。macOS 不使用 CUDA wheel。

~~~bash
python -c "import torch, gradio; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Gradio:', gradio.__version__)"
~~~

网页接口以 Gradio 4.44.1 验证，请先按 requirements-web.txt 安装。使用现成特征不需要 RNA-FM；构建特征和重新训练还需要原 requirements.txt 中的依赖。

## 4. 检查并启动

放置部署文件后，在激活的环境中执行：

~~~bash
python check_deployment.py
python app.py
~~~

出现 Running on local URL 后打开 http://127.0.0.1:7860。

检查脚本仅验证文件存在；权重和特征形状在模型加载时验证。首次启动还需加载特征、构图和缓存图表示，请等待地址出现。

也可在激活的环境中运行：

- Windows：.\start_local.bat
- Linux/macOS：sh start_local.sh

Windows 双击脚本时，系统默认 python 必须已经安装依赖；建议从激活的 Conda 环境终端启动。脚本没有绑定个人 Anaconda 路径。

程序默认使用可用 CUDA，否则使用 CPU。排序每批最多处理 128 对；预测和排序共用单个任务队列。停止服务：在启动终端按 Ctrl+C。

## 5. 网页操作

1. Home：首页封面 Start prediction、导航 Prediction 和 Open predictor 进入配对预测；Candidate ranking 进入排序。
2. Predict & explain：选择已有 ncRNA 和药物，点击 Predict，查看分数与交互热图。
3. Drug → top ncRNAs：选择药物、Top-K（5–30）、RNA 类型，点击 Rank。
4. Back to Home 返回首页；点击流程图查看高清图。

当前不接受任意新序列或自定义 SMILES。排序不会自动排除已知关联。分数与热图用于研究探索，不等同于实验验证的机制。

## 6. 数据在其他目录

无需复制大文件，可使用绝对路径。

Windows PowerShell：

~~~powershell
$env:NCDRUG_DATA_DIR = 'E:\HISCA-data\data'
$env:NCDRUG_FEAT_DIR = 'E:\HISCA-data\features'
$env:NCDRUG_CKPT_DIR = 'E:\HISCA-data\ckpt'
python check_deployment.py
python app.py
~~~

Linux/macOS：

~~~bash
export NCDRUG_DATA_DIR=/path/to/hisca-data/data
export NCDRUG_FEAT_DIR=/path/to/hisca-data/features
export NCDRUG_CKPT_DIR=/path/to/hisca-data/ckpt
python app.py
~~~

| 环境变量 | 用途 / 默认值 |
|---|---|
| NCDRUG_DATA_DIR | 数据；仓库下 data/ |
| NCDRUG_FEAT_DIR | 特征；仓库下 features/ |
| NCDRUG_CKPT_DIR | 权重；仓库下 ckpt/ |
| NCDRUG_DEMO_CKPT | 网页权重文件；默认 CKPT_DIR/hisca_demo.pt |
| NCDRUG_EDGE_CSV | 关联表；默认 DATA_DIR/merged_clean.csv |
| NCDRUG_PORT | 本地端口；默认 7860 |
| NCDRUG_TOKENIZER_DIR | 本地 tokenizer；默认 DeepChem/ChemBERTa-77M-MLM |
| GRADIO_TEMP_DIR | 临时文件；默认仓库下 .runtime/gradio/ |

NCDRUG_DMODEL、NCDRUG_HGT_LAYER 等模型参数应与训练一致，不要为了绕过加载错误随意修改。

## 7. 联网与离线

推理读取本地模型，不调用远程预测 API。安装依赖需要联网；首次解释热图可能下载 ChemBERTa tokenizer。封面和流程图已保存在仓库中。

离线前先安装依赖，并缓存 tokenizer，或准备本地目录：

~~~bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM').save_pretrained('tokenizer')"
~~~

Windows PowerShell：

~~~powershell
$env:NCDRUG_TOKENIZER_DIR = (Resolve-Path .\tokenizer).Path
$env:HF_HUB_OFFLINE = '1'
python app.py
~~~

Linux/macOS：

~~~bash
export NCDRUG_TOKENIZER_DIR="$(pwd)/tokenizer"
export HF_HUB_OFFLINE=1
python app.py
~~~

已有特征推理不需要完整 RNA-FM/ChemBERTa 编码模型，只需热图用的 tokenizer。药物分组是 token 的启发式分类，不是原子级归因。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| 缺少 CSV 或 .pt | 运行检查脚本，获取匹配部署文件或设置正确路径 |
| Missing key / Unexpected key / size mismatch | 检查权重架构、启用分支、隐藏维度与节点顺序 |
| ModuleNotFoundError | 确认激活了正确环境，并安装 requirements-web.txt |
| 端口占用 | PowerShell：$env:NCDRUG_PORT='7861'；Linux/macOS：export NCDRUG_PORT=7861 |
| 显存不足 | 关闭其他 GPU 程序，或设置 CUDA_VISIBLE_DEVICES=-1 后使用 CPU |
| 有分数但无热图 | 查看终端 heatmap err，检查 tokenizer 下载和本地目录 |
| PyG 可选扩展警告 | 本次验证出现过警告但推理成功；若有实际运算错误，检查 Torch/PyG 二进制兼容性 |
| 其他电脑无法访问 | 默认仅监听 127.0.0.1，属于本地部署的预期行为 |
| 双击脚本无法运行 | 从激活环境的终端运行，读取具体错误 |
| 离线仍尝试下载 | 设置 NCDRUG_TOKENIZER_DIR 与 HF_HUB_OFFLINE |

本指南不包含公网部署、访问控制或生产运维。

## 9. 验证范围与参考

本次使用真实部署文件验证了 Windows + Python 3.9.23 + PyTorch 2.6.0 CUDA + Gradio 4.44.1 的模型加载、配对预测、热图、Top-K 排序和 miRNA 过滤。也检查了缺少部署文件时的提示。

参考：[Gradio 4.44.1 Blocks](https://www.gradio.app/4.44.1/docs/gradio/blocks)、[PyTorch 安装说明](https://pytorch.org/get-started/locally/)。

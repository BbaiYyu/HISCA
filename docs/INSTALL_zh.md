# HISCA 安装与使用说明（v0.3.1）

## 1. 准备 Python

安装 64 位 Python 3.9–3.12。Windows 安装时勾选 Add Python to PATH。
在终端运行 `python --version` 确认 Python 可用。当前验证平台为 Windows、Python 3.9.23、CPU。

## 2. 一行安装

```powershell
python -m pip install "https://github.com/BbaiYyu/HISCA/releases/download/v0.3.1/hisca_web-0.3.1%2Bfull.20261020-py3-none-any.whl"
```

该地址来自本仓库 v0.3.1 Release。完整 wheel 约 720 MiB，包含平台代码、首页图片、
训练权重、六个预计算特征、部署数据和本地 ChemBERTa tokenizer。
无需先安装 Git 或下载整个源码仓库。

也可以从 [Release 页面](https://github.com/BbaiYyu/HISCA/releases/tag/v0.3.1)
下载 `hisca_web-0.3.1+full.20261020-py3-none-any.whl`，进入文件所在目录后安装：

```powershell
python -m pip install "./hisca_web-0.3.1+full.20261020-py3-none-any.whl"
```

下载时选择 `.whl` 安装包。Source code (zip) 是源码，不能代替完整模型安装包。

## 3. 一行启动

```powershell
hisca
```

如果终端提示找不到 hisca，使用：

```powershell
python -m hisca
```

首次启动自动建立独立 venv 环境，安装 CPU 推理依赖并加载模型。需要联网下载依赖，
耗时取决于网络；完成前不要关闭终端。准备好后自动打开浏览器。
以后仍运行同一个命令，已安装环境会被复用。无需 Jupyter、Conda、CUDA，也不要求显卡。

希望双击运行时，可以下载 Release 附件中的 `install_and_start.bat`。脚本会安装对应的
完整 wheel 并启动；`start_hisca.bat` 用于后续启动。首个脚本优先使用同目录已有的 wheel，
否则从本 Release 下载。

## 4. 使用平台

首页展示平台说明与模型框架。进入 Predict & explain，选择已有 ncRNA 和药物后开始预测，
可查看关联分数与相对交互热图。Drug → top ncRNAs 可按药物获取 Top-K 候选，支持 ncRNA 类型筛选。
当前版本使用内置列表，不接受任意新序列或新分子作为预测输入。

保留启动终端，按 Ctrl+C 停止本次服务。默认仅本机访问，端口被占用时自动选择另一个端口。

## 5. 运行目录与排错

Windows 默认运行目录为 `%LOCALAPPDATA%\HISCA`；其中保存独立环境、缓存、日志和部署配置。
可以通过 `--home` 或 `HISCA_HOME` 改为指定目录：

```powershell
hisca --home "D:\HISCA运行环境"
hisca --setup-only
hisca --no-browser
hisca --port 7870
```

- 安装下载失败：检查是否能访问 GitHub、PyPI 和 PyTorch CPU 下载站，恢复网络后重复运行即可。
- 模型启动失败：查看终端提示的 `logs/server-*.log`；请使用同一安装包内匹配的权重与特征。
- 同一目录提示初始化正在进行：等待另一个初始化完成后重试。
- 需要核验下载：Release 附有 `SHA256SUMS.txt`，Windows 可运行 `Get-FileHash 文件名 -Algorithm SHA256` 对照。

首启依赖安装需要网络。环境准备完成后，预测使用本地数据、模型和 tokenizer，不会自动重新训练。
完整包并非完全离线的 Python 运行环境，也不包含 Python 安装程序。

## 6. 从源码安装

```powershell
python -m pip install "git+https://github.com/BbaiYyu/HISCA.git"
hisca --bundle "完整模型目录或ZIP"
```

源码安装只提供轻量启动器；另需匹配的模型、特征、数据和 tokenizer。
完整 wheel 自带这些文件，普通使用者优先采用前面的直接安装方式。
本项目目前未发布 PyPI，`pip install hisca` 不是本版本的安装命令。

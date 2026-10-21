# HISCA

**ncRNA–药物关联预测平台**

HISCA 在本地电脑运行，提供 ncRNA 与药物的配对预测、区域–子结构交互热图，以及按药物筛选 ncRNA 候选的排序功能。

## 安装并启动

先安装 **64 位 Python 3.9–3.12**，然后在终端运行：

~~~powershell
python -m pip install "https://github.com/BbaiYyu/HISCA/releases/download/v0.3.1/hisca_web-0.3.1%2Bfull.20261020-py3-none-any.whl"
hisca
~~~

首次启动会自动准备独立 CPU 环境、安装依赖并加载模型，完成后打开浏览器。以后运行 `hisca` 即可；如果提示找不到命令，使用 `python -m hisca`。

完整安装包已包含模型、匹配数据、预计算特征和 tokenizer，无需自己训练，也不要求显卡、Conda 或 Jupyter。首次下载和安装依赖需要联网。已验证环境为 Windows、Python 3.9.23、CPU。

- **[安装与使用说明](docs/INSTALL_zh.md)**：逐步安装、网页使用、双击启动和常见问题。
- **[v0.3.1 下载页](https://github.com/BbaiYyu/HISCA/releases/tag/v0.3.1)**：完整安装包、启动脚本和文件校验值。
- **[直接下载完整安装包](https://github.com/BbaiYyu/HISCA/releases/download/v0.3.1/hisca_web-0.3.1%2Bfull.20261020-py3-none-any.whl)**：约 720 MiB。

下载时选择 `.whl` 完整安装包。GitHub 的 **Source code (zip)** 仅含源码，不包含模型数据。本项目尚未发布到 PyPI，请使用上述安装地址。

## 平台功能

| 功能 | 操作与结果 |
|---|---|
| 首页 | 查看平台介绍、模型框架和功能入口 |
| 配对预测与解释 | 选择已有 ncRNA 和药物，查看关联分数及相对交互热图 |
| 候选排序 | 选择药物、候选数量和 ncRNA 类型，获取 Top-K 排名 |

服务启动后保留终端窗口，按 `Ctrl+C` 停止。默认只供本机浏览器访问，程序会选择空闲端口。当前输入来自内置列表；分数与热图用于研究探索。

## 源码与开发

普通使用者按上面的安装方法即可。需要修改网页、构建特征或训练模型时，请阅读 **[源码开发说明](docs/DEVELOPMENT_zh.md)**。

| 目录 | 内容 |
|---|---|
| `models/` | 模型结构、数据组装、训练和评估 |
| `web/` | Gradio 网页、首页模板与图片 |
| `preprocessing/` | 特征计算和可选高阶超边构建 |
| `src/hisca/` | 安装后的 `hisca` 命令及环境初始化 |
| `requirements/` | 网页推理与研究依赖 |
| `tests/` | 启动器回归检查 |
| `docs/` | 安装说明与源码开发说明 |

稳定安装包为 **v0.3.1**；模型、数据与预计算特征通过 Release 完整安装包分发。

## 许可

源码采用 [MIT License](LICENSE)。

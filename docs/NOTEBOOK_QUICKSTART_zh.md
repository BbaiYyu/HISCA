# HISCA Notebook 与一键启动

## 最简单的使用方式

完整本地使用包解压后，进入 HISCA 文件夹。需要先安装 Python，支持 3.9–3.12，本次实际验证使用 3.9.23。

- 想直接使用网页：双击 one_click_start.bat。
- 想像 Jupyter Notebook 那样逐步运行：双击 open_notebook.bat，打开后选择 Run → Run All Cells。

第一次运行会创建项目专用 .venv-hisca 环境，并从官方软件仓库安装依赖，耗时取决于网络；之后会复用环境。不要关闭运行窗口。网页加载好后会给出本机访问地址，默认从 7860 开始自动查找空闲端口。

这不是无需 Python 的独立 exe，也不是公网托管。无需安装 Java 或 Apache Maven。

## Notebook 的四个单元

1. 检查项目位置以及数据、六类预计算特征、模型权重。
2. 自动检查和安装推理依赖；使用当前 Notebook 内核对应的 Python。
3. 启动本地 HISCA 网页，等待真实模型加载完成。
4. 显示可点击的网页链接和内嵌页面。

完整包中的 DATA_ROOT 保持空字符串。文件在其他目录时，将 DATA_ROOT 改成包含 data、features、ckpt 的上级目录。更改目录后重新运行启动单元会停止本 Notebook 启动的旧进程，再以新目录启动。

保持内核运行。停止时可新建单元运行 hisca.stop()，或关闭该内核；不会按端口关闭别人的服务。

## 已有 Jupyter 环境

将 HISCA_quickstart.ipynb 放在项目根目录，并从根目录启动 Jupyter 打开它。

~~~bash
python -m pip install jupyterlab
python -m jupyterlab HISCA_quickstart.ipynb
~~~

随后运行全部单元。已有兼容 CUDA Torch 会保留；缺少 Torch 时自动安装 CPU 版本。使用已安装依赖的正确内核，避免系统 Python 与 Notebook 内核不一致。

## Linux/macOS

~~~bash
sh one_click_start.sh
sh one_click_start.sh --mode notebook
~~~

CPU 安装器以 Windows/Linux 的 PyTorch CPU wheel 为默认路径；macOS 用户请按 PyTorch 官方方式先安装匹配的 torch==2.6.0，再在已有环境打开 Notebook。此次未实机验证 Linux/macOS。

## 完整包和 GitHub 代码的区别

完整本地包包含同一套数据、全部六项特征、hisca_demo.pt 以及用于热图的 ChemBERTa tokenizer；不包含 Python 安装程序或所有 pip 依赖。第一次安装仍需要联网。

GitHub 仅发布代码、Notebook、启动脚本和说明，数据/特征/权重/tokenizer 不推送。单独下载 GitHub 代码后，需要先取得相同部署文件，按原本地部署指南放置；不提供不存在的在线下载地址。

如果已安装依赖且使用完整包中的 tokenizer，模型预测和解释不需要调用远程预测服务。项目环境、Notebook 配置和运行日志保存在项目目录内，便于管理。

## 排错

- 缺少文件：Notebook 第一个单元会明确列出；完整包请完整解压，不要只复制 ipynb。
- 没有 Python：先从 https://www.python.org/downloads/ 安装兼容版本，或使用已有 Conda 环境。
- 安装很慢/中断：检查网络后重新运行；已完成的安装会被复用。
- 模型加载失败：查看 .runtime/notebook_server.log；权重必须与特征、架构配置匹配。
- 模型启动等待超时：内核中运行 print(hisca.tail()) 查看日志。可改为 hisca.start(timeout=600)。
- 浏览器内嵌区域空白：点击 Notebook 生成的本地链接。
- Notebook 重新启动内核：原内核启动的模型会停止，再运行全部单元。
- 端口被占用：自动尝试 7860–7880，不终止其他服务。
- 首次运行提示文件版本/导入问题：重启 Notebook 内核后再运行。
- 只有代码没有部署数据：请向维护者取得完整本地包。

## 老师提到的概念

Apache Maven 主要用于 Java 项目的构建和依赖管理。Jupyter Notebook 是将说明文字、可执行 Python 单元和运行结果放在同一页面的交互文档。

HISCA 的实现是：Notebook/双击脚本负责检查环境、安装依赖和管理启动；原 Gradio 网页负责交互；PyTorch 模型负责推理。没有改写成 Java，也没有改变模型预测公式。

参考：
- https://maven.apache.org/
- https://jupyter.org/install
- https://docs.jupyter.org/en/latest/running.html
## 本次实际验证

2026-10-20：通过真实 Jupyter 内核执行全部启动单元；通过全新隔离 CPU 环境的依赖安装、模型加载、/info、配对预测、解释热图、Top-5 miRNA 排序；确认重复启动复用同一进程、停止只针对本启动器创建的进程。
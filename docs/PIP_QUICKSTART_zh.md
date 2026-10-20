# HISCA：像 Jupyter 一样安装和启动

HISCA 已提供 `hisca` 命令。安装一次后，运行该命令会创建独立环境、安装 CPU 推理依赖、加载模型，并在准备完成后自动打开浏览器。无需先打开 Notebook，也不要求 CUDA 或显卡。

## 推荐：完整安装包

先安装 **64 位 Python 3.9–3.12** 并勾选 Add Python to PATH。当前实际验证环境是 Windows、Python 3.9.23、CPU；其他系统尚未完成实测。

从维护者获得完整 `.whl` 文件，在它所在的文件夹打开终端，运行：

```powershell
python -m pip install ".\hisca_web-0.3.0+full.20261020-py3-none-any.whl"
hisca
```

第一行安装 HISCA；第二行自动初始化并打开平台。以后只运行 `hisca` 即可。如果终端找不到命令，可使用等价命令 `python -m hisca`。已有轻量版时，完整版本的 `+full.20261020` 版本号可避免被 pip 当成相同版本跳过。

完整 wheel 包含本平台匹配的数据、六份预计算特征、训练权重、ChemBERTa tokenizer、首页与图片。首启仍需联网安装 PyTorch 等依赖；下载耗时取决于网速。后续启动复用已安装环境和本地模型。正常使用无需重新训练模型或下载编码器权重。

浏览器里可以看到相同的首页、配对预测、解释热图和候选排序。平台使用内置已有 ncRNA 和药物列表。保留启动终端；按 Ctrl+C 可以停止本次启动的服务。默认仅本机访问，自动选择空闲端口。

## GitHub 轻量安装

有 Git 的用户可以从代码仓库安装启动器：

```powershell
python -m pip install "git+https://github.com/BbaiYyu/HISCA.git"
hisca --bundle "完整模型包.zip"
```

`--bundle` 也可以指向同时包含 `data/`、`features/`、`ckpt/`、`tokenizer/` 的完整目录。首次指定后保存位置，以后运行 `hisca` 即可。旧版 `HISCA_完整本地使用包.zip` 可以直接导入，ZIP 只会导入选定的模型、特征和数据文件。

**仓库未上传大模型文件，也没有配置公开模型下载地址；仅安装 GitHub 代码不能凭空获得预测模型。** 不要把 `pip install hisca` 当成目前可用的 PyPI 命令：本项目尚未发布到 PyPI。维护者可以分享完整 wheel 供用户直接安装。

## 工作目录与常见问题

默认运行目录是 Windows 的 `%LOCALAPPDATA%\HISCA`，其他系统是用户数据目录。模型安装资源只读，独立环境、日志和缓存放在工作目录，不写入 site-packages。可自定义：

```powershell
hisca --home "D:\HISCA运行环境"
```

设置 `HISCA_HOME` 环境变量也可固定工作目录。当前作者电脑的快捷启动脚本已将目录设在 `D:\chatGPT`。

- 首启卡在下载：查看终端的 pip 输出，确认可访问 PyPI 和 PyTorch CPU wheel 服务器；失败后可重复运行，已经装好的依赖会保留。
- 模型启动错误：按终端给出的路径查看 `logs/server-*.log`。不要混用不同训练版本的权重和特征。
- 想先准备好环境：`hisca --setup-only`。
- 服务器没有图形桌面：`hisca --no-browser`；按终端输出的本地地址访问。
- 指定端口：`hisca --port 7870`；占用时会提示，不会停止其他进程。

## 维护者构建

```powershell
python -m pip install --upgrade setuptools wheel
python setup.py build --build-base .runtime/build-lite bdist_wheel --dist-dir dist/lite
# 完整包：HISCA_BUNDLE_SOURCE 指向完整部署数据目录
$env:HISCA_BUNDLE_SOURCE = "完整部署目录"
python setup.py build --build-base .runtime/build-full bdist_wheel --dist-dir dist/full
```

轻量和完整构建目录必须分开；源码分发包不包含模型数据。完整 wheel 面向直接文件分享，体积较大。它未包含所有 Python 依赖，不能视为完全离线安装包。对外发布模型文件前应确认研究数据与模型的分发范围。

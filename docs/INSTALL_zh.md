# HISCA 安装与使用说明

本说明适用于 **v0.3.1 稳定完整安装包**。只使用预测平台时，按本文操作即可，无需下载或修改源码。

## 1. 准备 Python

安装 **64 位 Python 3.9–3.12**。Windows 安装时勾选 **Add Python to PATH**，安装完成后重新打开终端，检查：

~~~powershell
python --version
~~~

已验证环境为 Windows、Python 3.9.23、CPU。安装包不包含 Python 本身。

## 2. 安装并启动

在终端依次运行：

~~~powershell
python -m pip install "https://github.com/BbaiYyu/HISCA/releases/download/v0.3.1/hisca_web-0.3.1%2Bfull.20261020-py3-none-any.whl"
hisca
~~~

第一条命令安装 HISCA。第二条命令自动创建独立运行环境、安装 CPU 推理依赖、加载模型，并打开浏览器。

首次启动需要联网下载依赖，请保持终端运行，等待出现 `HISCA is ready` 和本地网址。以后运行 `hisca` 即可，已经准备好的环境会被复用。

如果系统找不到 `hisca` 命令，可用：

~~~powershell
python -m hisca
~~~

完整安装包约 **720 MiB**，包含网页、图片、训练权重、六份预计算特征、部署数据和本地 tokenizer。无需额外训练，也不要求显卡、CUDA、Conda 或 Jupyter。环境准备完成后，模型预测在本地完成。

## 3. 手动下载或双击启动

打开 **[v0.3.1 下载页](https://github.com/BbaiYyu/HISCA/releases/tag/v0.3.1)**，按需要选择附件：

| 附件 | 用途 |
|---|---|
| `hisca_web-0.3.1+full.20261020-py3-none-any.whl` | 包含模型与数据的完整安装包 |
| `install_and_start.bat` | Windows 双击安装并启动 |
| `start_hisca.bat` | 安装完成后双击启动 |
| `HISCA_INSTALL_zh.md` | 随包安装说明 |
| `SHA256SUMS.txt` | 下载文件的 SHA-256 校验值 |

选择完整 `.whl`，而不是页面自动生成的 **Source code (zip)**。源码压缩包不包含预测所需模型文件。

手动下载 wheel 后，在其所在文件夹打开终端：

~~~powershell
python -m pip install "./hisca_web-0.3.1+full.20261020-py3-none-any.whl"
hisca
~~~

使用双击脚本时，先安装好 Python。`install_and_start.bat` 优先安装同目录的完整 wheel；未找到时从该 Release 下载。后续可使用 `start_hisca.bat`。

需要核验文件时，在 PowerShell 中执行，并与下载页的 `SHA256SUMS.txt` 对照：

~~~powershell
Get-FileHash "./hisca_web-0.3.1+full.20261020-py3-none-any.whl" -Algorithm SHA256
~~~

## 4. 使用网页

1. **Home**：查看平台介绍和模型框架，进入预测或排序功能。
2. **Predict & explain**：选择已有 ncRNA 和药物，点击 Predict，查看关联分数及区域–子结构交互热图。
3. **Drug → top ncRNAs**：选择药物、候选数量与 ncRNA 类型，点击 Rank，查看候选排序。

当前版本使用内置 ncRNA 和药物列表，不接受任意新序列或新分子。分数与相对交互热图用于研究探索。

网页服务运行在本机，默认使用 `127.0.0.1`，程序会自动选择空闲端口。保留启动终端，按 `Ctrl+C` 停止本次服务；关闭服务后网页不能继续预测。

## 5. 运行目录与选项

Windows 默认运行目录为 `%LOCALAPPDATA%\HISCA`，用于保存独立环境、缓存、日志和部署配置。完整安装包中的模型与网页资源按只读方式使用。

可指定其他有写权限的目录，例如：

~~~powershell
hisca --home "D:\chatGPT\HISCA运行环境"
~~~

后续启动继续带上同一个 `--home`，也可以设置 `HISCA_HOME` 环境变量。其他选项：

| 命令 | 用途 |
|---|---|
| `hisca --setup-only` | 只准备环境，暂不启动网页 |
| `hisca --no-browser` | 启动后只打印地址，手动打开浏览器 |
| `hisca --port 7870` | 使用指定端口 |
| `hisca --help` | 查看完整命令帮助 |

这些命令可与同一个 `--home` 参数组合使用。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| 找不到 Python | 检查安装及 PATH，重新打开终端后运行 `python --version` |
| 找不到 hisca | 使用安装时相同的 Python 运行 `python -m hisca` |
| 下载或依赖安装失败 | 确认可访问 GitHub、PyPI 和 PyTorch CPU 下载站，恢复网络后用相同命令重试 |
| 首次启动等待较久 | 查看终端中的安装进度，环境与模型准备完成后才会打开网页 |
| 提示另一个初始化正在运行 | 等待同一运行目录中的初始化完成后重试 |
| 模型启动失败 | 按终端提示查看运行目录内 `logs/server-*.log`，使用同一完整包中的匹配模型与特征 |
| 指定端口已占用 | 更换端口或不指定端口，让程序自动选择 |

完整包不是完全离线的 Python 运行环境：首次安装依赖需要网络；之后预测使用本地数据、模型和 tokenizer，启动时不会重新训练。

需要修改代码或开展研究时，请转到 [源码开发说明](DEVELOPMENT_zh.md)。返回 [仓库首页](../README.md)。

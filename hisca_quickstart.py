"""Portable Notebook/one-click launcher; no model imports in this module."""
from pathlib import Path
import atexit
import importlib.metadata as metadata
import json, os, socket, subprocess, sys, time, urllib.request, webbrowser

ROOT = Path(__file__).resolve().parent

def deployment_env(data_root=None):
    base = Path(data_root).expanduser().resolve() if data_root else ROOT
    env = os.environ.copy()
    for key, value in {"NCDRUG_DATA_DIR": base / "data",
                       "NCDRUG_FEAT_DIR": base / "features",
                       "NCDRUG_CKPT_DIR": base / "ckpt"}.items():
        if data_root or key not in env:
            env[key] = str(value)
    if data_root:
        env["NCDRUG_DEMO_CKPT"] = str(base / "ckpt" / "hisca_demo.pt")
        env["NCDRUG_EDGE_CSV"] = str(base / "data" / "merged_clean.csv")
    tokenizer = base / "tokenizer"
    if (tokenizer / "tokenizer_config.json").is_file():
        env.setdefault("NCDRUG_TOKENIZER_DIR", str(tokenizer))
        env.setdefault("HF_HUB_OFFLINE", "1")
    env.update(GRADIO_TEMP_DIR=str(ROOT / ".runtime" / "gradio"),
               GRADIO_ANALYTICS_ENABLED="False", PYTHONUTF8="1")
    return env

def check_files(data_root=None):
    from check_deployment import validate_runtime
    env = deployment_env(data_root)
    paths = validate_runtime(
        data_dir=env["NCDRUG_DATA_DIR"], feat_dir=env["NCDRUG_FEAT_DIR"],
        checkpoint=env.get("NCDRUG_DEMO_CKPT",
                 str(Path(env["NCDRUG_CKPT_DIR"]) / "hisca_demo.pt")),
        edge_csv=env.get("NCDRUG_EDGE_CSV",
                 str(Path(env["NCDRUG_DATA_DIR"]) / "merged_clean.csv")))
    missing = [str(Path(env["NCDRUG_FEAT_DIR"]) / name) for name in
               ("rna_kmer.pt", "drug_extra.pt", "drug_graphs.pt")
               if not (Path(env["NCDRUG_FEAT_DIR"]) / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete standard deployment bundle:\n  " +
                                "\n  ".join(missing))
    return paths

def missing_dependencies():
    missing = []
    for line in (ROOT / "requirements-web.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = metadata.version(name).split("+", 1)[0]
        except metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            missing.append(line)
    return missing

def ensure_dependencies():
    missing = missing_dependencies()
    if not missing:
        print("Inference dependencies are ready.")
        return
    print("Installing inference dependencies; first installation needs Internet.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    if "torch==2.6.0" in missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.6.0+cpu",
                        "--extra-index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(ROOT / "requirements-web.txt")], check=True)
    if missing_dependencies():
        raise RuntimeError("Versions still differ; restart the kernel and check again.")

class Platform:
    def __init__(self, data_root=None):
        self.data_root, self.process, self.url = data_root, None, None
        self.log_path = ROOT / ".runtime" / "notebook_server.log"
        self._log_handle = None

    def start(self, timeout=240, open_browser=False):
        if self.process is not None and self.process.poll() is None:
            print("Already running:", self.url)
            return self.url
        check_files(self.data_root)
        if missing_dependencies():
            raise RuntimeError("Run ensure_dependencies() before starting.")
        env = deployment_env(self.data_root)
        port = None
        for candidate in range(7860, 7881):
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", candidate))
                except OSError:
                    continue
            port = candidate
            break
        if port is None:
            raise RuntimeError("No free local port in 7860-7880.")
        env["NCDRUG_PORT"] = str(port)
        self.url = f"http://127.0.0.1:{port}"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_handle:
            self._log_handle.close()
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "app.py")], cwd=str(ROOT),
            env=env, stdout=self._log_handle, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        atexit.register(self.stop)
        deadline = time.monotonic() + timeout
        print("Loading model. Log:", self.log_path, flush=True)
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Server exited:\n" + self.tail())
                try:
                    with urllib.request.urlopen(self.url + "/config", timeout=2) as r:
                        config = json.load(r)
                    if "HISCA" in config.get("title", ""):
                        print("HISCA is ready:", self.url, flush=True)
                        if open_browser:
                            webbrowser.open(self.url)
                        return self.url
                except (OSError, ValueError):
                    pass
                time.sleep(0.5)
            raise TimeoutError("Startup timed out:\n" + self.tail())
        except BaseException:
            self.stop()
            raise

    def tail(self, lines=25):
        if not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(
            encoding="utf-8", errors="replace").splitlines()[-lines:])

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
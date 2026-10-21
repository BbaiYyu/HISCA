
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
import zipfile
from . import __version__
from .bundle import validate_bundle, import_zip

PACKAGE = Path(__file__).resolve().parent

def runtime_root():
    packaged = PACKAGE / "_runtime"
    return packaged if packaged.is_dir() else PACKAGE.parents[1]

def runtime_file(name):
    base = runtime_root()
    path = base / name
    if path.is_file():
        return path
    locations = {"app.py": "web/app.py", "requirements-web.txt": "requirements/web.txt"}
    return base / locations.get(name, name)

def default_home():
    if os.environ.get("HISCA_HOME"):
        return Path(os.environ["HISCA_HOME"]).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "HISCA"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "hisca"

@contextmanager
def setup_lock(home):
    home.mkdir(parents=True, exist_ok=True)
    handle = (home / "setup.lock").open("a+b")
    try:

        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another HISCA initialization is running in " + str(home)) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()

def choose_bundle(home, source=None):
    config = home / "deployment.json"
    if source:
        source = Path(source).expanduser().resolve()
        base = import_zip(source, home) if source.is_file() else validate_bundle(source)
    elif config.exists():
        try:
            saved = json.loads(config.read_text(encoding="utf-8"))
            selected = saved["data_root"]
            if not isinstance(selected, str) or not selected:
                raise ValueError("data_root must be a path")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("Invalid deployment.json. Run hisca --bundle PATH to select the model again.") from exc
        base = validate_bundle(selected)
    else:
        base = None
        for candidate in (PACKAGE / "_bundle", Path.cwd(), runtime_root()):
            try:
                base = validate_bundle(candidate)
                break
            except FileNotFoundError:
                pass
        if base is None:
            if sys.stdin.isatty():
                print("First start: enter the full HISCA model ZIP or its extracted folder.")
                selected = input("Model bundle path: ").strip().strip('"')
                if selected:
                    return choose_bundle(home, selected)
            raise FileNotFoundError("Model data are absent from the lightweight package. "
                "Install the full wheel, or run: hisca --bundle PATH_TO_ZIP_OR_FOLDER. "
                "Get the full wheel at https://github.com/BbaiYyu/HISCA/releases/latest .")
    temp = config.with_suffix(".tmp")
    temp.write_text(json.dumps({"data_root": str(base)}, indent=2), encoding="utf-8")
    temp.replace(config)
    return base

def process_env(home, base=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith("NCDRUG_")}
    for name, folder in {"PIP_CACHE_DIR": "pip-cache", "MPLCONFIGDIR": "matplotlib",
                         "HF_HOME": "huggingface", "GRADIO_TEMP_DIR": "gradio"}.items():
        path = home / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               GRADIO_ANALYTICS_ENABLED="False", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NCDRUG_V2_HYPER="0",
               NCDRUG_INT_HEADS="1", NCDRUG_INT_CENTER="0", NCDRUG_INT_NORM="0")

    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if base:
        env.update(NCDRUG_DATA_DIR=str(base / "data"), NCDRUG_FEAT_DIR=str(base / "features"),
                   NCDRUG_CKPT_DIR=str(base / "ckpt"),
                   NCDRUG_DEMO_CKPT=str(base / "ckpt/hisca_demo.pt"),
                   NCDRUG_EDGE_CSV=str(base / "data/merged_clean.csv"),
                   NCDRUG_TOKENIZER_DIR=str(base / "tokenizer"), HF_HUB_OFFLINE="1")
    return env

def environment_python(home):
    if not ((3, 9) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError("Use Python 3.9-3.12 (64-bit); Windows Python 3.9 CPU is tested.")
    if platform.architecture()[0] != "64bit":
        raise RuntimeError("HISCA requires 64-bit Python.")
    folder = home / "envs" / ("cpu-py" + str(sys.version_info[0]) + str(sys.version_info[1]))
    python = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("[1/3] Creating the isolated HISCA environment ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(folder)
    return python

def requirements_ready(python, requirements, env):
    code = """import importlib.metadata as m,json,sys
missing=[]
for name,expected in json.loads(sys.argv[1]):
 try: actual=m.version(name).split('+',1)[0]
 except m.PackageNotFoundError: actual=None
 if actual != expected: missing.append(name+'=='+expected)
print(json.dumps(missing))
"""
    pins = [line.strip().split("==", 1) for line in requirements.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    result = subprocess.run([str(python), "-c", code, json.dumps(pins)], env=env,
                            text=True, encoding="utf-8", capture_output=True, check=True)
    return json.loads(result.stdout)

def ensure_environment(home):
    python = environment_python(home)
    requirements = runtime_file("requirements-web.txt")
    env = process_env(home)
    missing = requirements_ready(python, requirements, env)
    if missing:
        print("[2/3] Installing CPU inference dependencies (first start needs Internet).", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], env=env, check=True)
        if "torch==2.6.0" in missing:
            subprocess.run([str(python), "-m", "pip", "install", "torch==2.6.0+cpu",
                            "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
                           env=env, check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)],
                       env=env, check=True)
        if requirements_ready(python, requirements, env):
            raise RuntimeError("Dependency installation is incomplete; run hisca again to retry.")
    else:
        print("[1/3] Isolated environment ready. [2/3] Dependencies ready.", flush=True)
    return python

def free_port(requested=None):
    for port in ([requested] if requested else range(7860, 7900)):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise RuntimeError("Local port is occupied; use --port with another available port.")

def start_server(python, home, base, *, port=None, open_browser=True, timeout=300):
    port = free_port(port)
    env = process_env(home, base)
    env["NCDRUG_PORT"] = str(port)
    logs = home / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / ("server-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()) + ".log")
    output = log.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen([str(python), "-u", str(runtime_file("app.py"))],
                                cwd=str(home), env=env, stdout=output, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    finally:
        output.close()
    url = "http://127.0.0.1:" + str(port)
    print("[3/3] Loading model ... Log: " + str(log), flush=True)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("Platform failed to start. Log:\n" + log.read_text(encoding="utf-8", errors="replace")[-5000:])
            try:
                with urllib.request.urlopen(url + "/config", timeout=2) as response:
                    config = json.load(response)
                if "HISCA" in config.get("title", ""):
                    print("HISCA is ready: " + url + "\nKeep this window open. Ctrl+C stops HISCA.", flush=True)
                    if open_browser:
                        webbrowser.open(url)
                    return proc, url, log
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        raise TimeoutError("Model startup timed out. See " + str(log))
    except BaseException:
        stop_server(proc)
        raise

def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

def launch(bundle=None, home=None, port=None, open_browser=True, setup_only=False):
    home = Path(home).expanduser().resolve() if home else default_home()
    with setup_lock(home):
        base = choose_bundle(home, bundle)
        python = ensure_environment(home)
    if setup_only:
        print("Initialization complete. Run hisca to open the platform.")
        return 0
    proc, _, _ = start_server(python, home, base, port=port, open_browser=open_browser)
    try:
        proc.wait()
        if proc.returncode:
            raise RuntimeError("HISCA server exited with code " + str(proc.returncode))
    except KeyboardInterrupt:
        print("\nStopping HISCA.")
    finally:
        stop_server(proc)
    return 0

def main(argv=None):
    parser = argparse.ArgumentParser(description="Install once, run hisca, open the local prediction platform.")
    parser.add_argument("--version", action="version", version="HISCA " + __version__)
    parser.add_argument("--bundle", help="Full model ZIP or extracted data/features/ckpt/tokenizer folder (first start).")
    parser.add_argument("--home", help="Writable environment/cache/log directory; also HISCA_HOME.")
    parser.add_argument("--port", type=int, help="Local port (default: first free port from 7860).")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser.")
    parser.add_argument("--setup-only", action="store_true", help="Prepare the environment without starting the web server.")
    args = parser.parse_args(argv)
    if args.port is not None and not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    try:
        return launch(bundle=args.bundle, home=args.home, port=args.port,
                      open_browser=not args.no_browser, setup_only=args.setup_only)
    except KeyboardInterrupt:
        print("\nInitialization interrupted. Run hisca again to retry.")
        return 130
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        print("HISCA: " + str(exc), file=sys.stderr)
        return 1

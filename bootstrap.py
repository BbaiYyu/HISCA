"""Create a project environment and open HISCA or JupyterLab."""
from pathlib import Path
import argparse, importlib.metadata as metadata, os, subprocess, sys, venv

ROOT = Path(__file__).resolve().parent
for key, folder in {
    "PIP_CACHE_DIR": "pip-cache",
    "JUPYTER_CONFIG_DIR": "jupyter-config",
    "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
    "IPYTHONDIR": "ipython",
}.items():
    location = ROOT / ".runtime" / folder
    location.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(location))

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["web", "notebook"], default="web")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--current-env", action="store_true")
    args = parser.parse_args()
    if not (3, 9) <= sys.version_info[:2] <= (3, 12):
        raise SystemExit("Use Python 3.9-3.12; Python 3.9.23 was tested.")
    if args.mode == "web":
        from hisca_quickstart import check_files
        check_files(args.data_root)
    if not args.current_env:
        target = ROOT / ".venv-hisca"
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            print("Creating isolated project environment:", target, flush=True)
            venv.EnvBuilder(with_pip=True).create(target)
        command = [str(python), str(Path(__file__).resolve()),
                   "--mode", args.mode, "--current-env"]
        if args.data_root:
            command += ["--data-root", args.data_root]
        raise SystemExit(subprocess.call(command))
    from hisca_quickstart import Platform, check_files, ensure_dependencies, deployment_env
    if args.mode == "web":
        check_files(args.data_root)
        ensure_dependencies()
        platform = Platform(args.data_root)
        try:
            platform.start(open_browser=True)
            print("Keep this window open. Press Ctrl+C to stop.", flush=True)
            platform.process.wait()
        except KeyboardInterrupt:
            pass
        finally:
            platform.stop()
    else:
        try:
            metadata.version("jupyterlab")
            metadata.version("ipykernel")
        except metadata.PackageNotFoundError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "jupyterlab>=4,<5", "ipykernel>=6,<7"], check=True)
        command = [sys.executable, "-m", "jupyterlab",
                   str(ROOT / "HISCA_quickstart.ipynb"),
                   "--ServerApp.ip=127.0.0.1", "--ServerApp.root_dir=" + str(ROOT)]
        raise SystemExit(subprocess.call(command, cwd=str(ROOT), env=deployment_env(args.data_root)))

if __name__ == "__main__":
    run()
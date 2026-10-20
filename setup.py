"""Build a small launcher wheel, or a full local distribution with HISCA_BUNDLE_SOURCE."""
import os
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = ["app.py", "config.py", "utils.py", "dataset.py", "main.py", "model.py",
                 "layers.py", "interpretation.py", "check_deployment.py", "home.html",
                 "requirements-web.txt", "assets/cover_rna_drug.png", "assets/architecture_latest.png"]

class BuildRuntime(build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / "hisca" / "_runtime"
        for name in RUNTIME_FILES:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(ROOT / name), str(dest))
        source = os.environ.get("HISCA_BUNDLE_SOURCE")
        if not source and (Path(self.build_lib) / "hisca" / "_bundle").exists():
            raise RuntimeError("Full bundle found in build output. Use a separate clean build directory for lite.")
        if source:
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location("hisca_bundle_build", ROOT / "src/hisca/bundle.py")
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            base = Path(source).resolve()
            mod.validate_bundle(base)
            for name in mod.BUNDLE_FILES:
                dest = Path(self.build_lib) / "hisca" / "_bundle" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                self.copy_file(str(base / name), str(dest))

full = bool(os.environ.get("HISCA_BUNDLE_SOURCE"))
setup(name="hisca-web", version="0.3.0+full.20261020" if full else "0.3.0",
      description="HISCA ncRNA-drug prediction platform launcher",
      long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
      long_description_content_type="text/markdown", url="https://github.com/BbaiYyu/HISCA",
      license="MIT", python_requires=">=3.9,<3.13",
      package_dir={"": "src"}, packages=find_packages("src"),
      cmdclass={"build_py": BuildRuntime}, include_package_data=False,
      entry_points={"console_scripts": ["hisca=hisca.cli:main"]})

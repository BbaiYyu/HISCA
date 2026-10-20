"""Focused stdlib regression tests for the lightweight launcher; no model imports."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

from hisca import cli
from hisca.bundle import BUNDLE_FILES, import_zip, validate_bundle

PROJECT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        # Keep all temporary working files with this task, not on the C drive.
        self.temporary = tempfile.TemporaryDirectory(prefix="launcher-test-", dir=str(PROJECT / "tests"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_bundle(self, base, omit=None, empty=None):
        for name in BUNDLE_FILES:
            if name == omit:
                continue
            target = base / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"" if name == empty else b"test")
        return base

    def make_zip(self, path, prefix="", extra=(), omit=None):
        with zipfile.ZipFile(path, "w") as archive:
            for name in BUNDLE_FILES:
                if name != omit:
                    archive.writestr(prefix + name, b"test")
            for name, content in extra:
                archive.writestr(name, content)
        return path

    def test_complete_bundle_and_missing_or_empty_feature(self):
        valid = self.make_bundle(self.root / "valid")
        self.assertEqual(validate_bundle(valid), valid.resolve())
        for missing in ("features/rna_kmer.pt", "features/drug_graphs.pt"):
            with self.subTest(missing=missing):
                bad = self.make_bundle(self.root / missing.replace("/", "_"), omit=missing)
                with self.assertRaisesRegex(FileNotFoundError, Path(missing).name):
                    validate_bundle(bad)
        bad = self.make_bundle(self.root / "empty", empty="features/drug_extra.pt")
        with self.assertRaisesRegex(FileNotFoundError, "drug_extra.pt"):
            validate_bundle(bad)

    def test_import_nested_bundle_is_selective_and_repeatable(self):
        archive = self.make_zip(self.root / "valid.zip", prefix="HISCA/",
                                extra=[("HISCA/ignored.py", b"not imported")])
        home = self.root / "home"
        first = import_zip(archive, home)
        self.assertEqual(import_zip(archive, home), first)
        self.assertEqual({p.relative_to(first).as_posix() for p in first.rglob("*") if p.is_file()},
                         set(BUNDLE_FILES))
        self.assertEqual(list((home / "bundles").glob("import-*")), [])

    def test_zip_path_escapes_rejected_before_extraction(self):
        hostile = ("../escaped.txt", "folder/../../escaped.txt", "/escaped.txt",
                   "C:/escaped.txt", "..\\escaped.txt", "folder/C:/escaped.txt")
        for index, name in enumerate(hostile):
            with self.subTest(name=name):
                archive = self.make_zip(self.root / ("unsafe-%s.zip" % index),
                                        extra=[(name, b"unsafe")])
                home = self.root / ("home-%s" % index)
                with self.assertRaisesRegex(ValueError, "Unsafe path"):
                    import_zip(archive, home)
                self.assertFalse((self.root / "escaped.txt").exists())
                self.assertEqual(list((home / "bundles").iterdir()), [])

    def test_zip_missing_feature_leaves_no_partial_bundle(self):
        archive = self.make_zip(self.root / "missing.zip", omit="features/rna_kmer.pt")
        home = self.root / "home"
        with self.assertRaisesRegex(FileNotFoundError, "rna_kmer.pt"):
            import_zip(archive, home)
        self.assertEqual(list((home / "bundles").iterdir()), [])

    def test_duplicate_and_multiple_bundles_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            duplicate = self.make_zip(self.root / "duplicate.zip",
                                      extra=[("features/rna_feats.pt", b"duplicate")])
        with self.assertRaisesRegex(ValueError, "Duplicate path"):
            import_zip(duplicate, self.root / "duplicate-home")
        two = self.make_zip(self.root / "two.zip", prefix="one/",
                            extra=[("two/ckpt/hisca_demo.pt", b"second")])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            import_zip(two, self.root / "two-home")

    def test_research_environment_is_cleaned_and_paths_are_explicit(self):
        home, base = self.root / "home", self.root / "model"
        inherited = {"NCDRUG_DMODEL": "64", "NCDRUG_NO_HGT": "1",
                     "NCDRUG_DATA_DIR": "old-data", "NCDRUG_TOKENIZER_DIR": "old-tokenizer",
                     "PYTHONPATH": "foreign-modules", "PYTHONHOME": "foreign-python",
                     "CUDA_VISIBLE_DEVICES": "3", "HISCA_TEST_KEEP": "preserved"}
        with mock.patch.dict(os.environ, inherited):
            env = cli.process_env(home, base)
            self.assertEqual(os.environ["NCDRUG_DMODEL"], "64")
        self.assertNotIn("NCDRUG_DMODEL", env)
        self.assertNotIn("NCDRUG_NO_HGT", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["NCDRUG_DATA_DIR"], str(base / "data"))
        self.assertEqual(env["NCDRUG_TOKENIZER_DIR"], str(base / "tokenizer"))
        self.assertEqual(env["NCDRUG_EDGE_CSV"], str(base / "data/merged_clean.csv"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["HISCA_TEST_KEEP"], "preserved")
        for name in ("PIP_CACHE_DIR", "MPLCONFIGDIR", "HF_HOME", "GRADIO_TEMP_DIR"):
            self.assertEqual(Path(env[name]).parent, home)
            self.assertTrue(Path(env[name]).is_dir())

    def test_setup_lock_blocks_competing_initializer_and_is_released(self):
        home = self.root / "home"
        with cli.setup_lock(home):
            with self.assertRaisesRegex(RuntimeError, "Another HISCA initialization"):
                with cli.setup_lock(home):
                    self.fail("Second initializer acquired the lock")
        with cli.setup_lock(home):
            pass

    def test_explicit_bundle_is_remembered_without_environment_install(self):
        home = self.root / "home"
        home.mkdir()
        base = self.make_bundle(self.root / "models")
        self.assertEqual(cli.choose_bundle(home, str(base)), base.resolve())
        self.assertEqual(cli.choose_bundle(home), base.resolve())
        self.assertEqual(json.loads((home / "deployment.json").read_text())["data_root"],
                         str(base.resolve()))

    def test_help_and_version_work_without_site_packages(self):
        env = os.environ.copy()
        env.update(PYTHONPATH=str(PROJECT / "src"), PYTHONDONTWRITEBYTECODE="1",
                   HISCA_HOME=str(self.root / "unused-home"))
        env.pop("PYTHONHOME", None)
        # -S disables site-packages: Torch/NumPy/Gradio are unavailable here.
        for argument in ("--help", "--version"):
            with self.subTest(argument=argument):
                result = subprocess.run([sys.executable, "-S", "-m", "hisca", argument],
                                        cwd=str(self.root), env=env, capture_output=True,
                                        text=True, encoding="utf-8", timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("HISCA" if argument == "--version" else "--bundle", result.stdout)
        self.assertFalse((self.root / "unused-home").exists())

    def test_invalid_port_is_rejected_before_initialization(self):
        with mock.patch.object(cli, "launch") as launch:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(["--port", "0"])
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_startup_timeout_stops_only_the_created_child(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        home, base = self.root / "home", self.root / "model"
        home.mkdir()
        with mock.patch.object(cli, "free_port", return_value=7860), \
             mock.patch.object(cli.subprocess, "Popen", return_value=proc) as start, \
             mock.patch.object(cli, "stop_server") as stop, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):
                cli.start_server(Path(sys.executable), home, base, timeout=0,
                                 open_browser=False)
        start.assert_called_once()
        stop.assert_called_once_with(proc)

    def test_cli_reports_invalid_zip_without_traceback(self):
        invalid = self.root / "not-a-model.zip"
        invalid.write_text("This is not a ZIP", encoding="utf-8")
        stderr = io.StringIO()
        with mock.patch.object(cli, "ensure_environment") as install, \
             contextlib.redirect_stderr(stderr):
            result = cli.main(["--bundle", str(invalid), "--home",
                               str(self.root / "home"), "--setup-only"])
        self.assertEqual(result, 1)
        self.assertIn("HISCA:", stderr.getvalue())
        install.assert_not_called()

    def test_cli_reports_malformed_deployment_state_without_traceback(self):
        home = self.root / "home"
        home.mkdir()
        (home / "deployment.json").write_text("{}", encoding="utf-8")
        stderr = io.StringIO()
        with mock.patch.object(cli, "ensure_environment") as install, \
             contextlib.redirect_stderr(stderr):
            result = cli.main(["--home", str(home), "--setup-only"])
        self.assertEqual(result, 1)
        self.assertIn("HISCA:", stderr.getvalue())
        install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
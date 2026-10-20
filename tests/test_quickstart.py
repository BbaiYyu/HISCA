import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hisca_quickstart as q

class QuickstartTests(unittest.TestCase):
    def test_external_data_overrides_old_paths(self):
        with patch.dict(os.environ, {"NCDRUG_DATA_DIR": "old-data",
                                   "NCDRUG_DEMO_CKPT": "old-model"}):
            base = (q.ROOT / ".runtime" / "external-data").resolve()
            env = q.deployment_env(str(base))
            self.assertEqual(env["NCDRUG_DATA_DIR"], str(base / "data"))
            self.assertEqual(env["NCDRUG_DEMO_CKPT"], str(base / "ckpt/hisca_demo.pt"))
            self.assertEqual(env["GRADIO_ANALYTICS_ENABLED"], "False")

    def test_missing_kmer_is_rejected(self):
        temp_parent = q.ROOT / ".runtime"
        temp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as td:
            base = Path(td)
            names = ["data/drugs.csv", "data/merged_clean.csv",
                     "ckpt/hisca_demo.pt", "features/rna_feats.pt",
                     "features/drug_feats.pt", "features/target_feats.pt",
                     "features/drug_extra.pt", "features/drug_graphs.pt"]
            for name in names:
                file = base / name
                file.parent.mkdir(exist_ok=True)
                file.touch()
            with self.assertRaisesRegex(FileNotFoundError, "rna_kmer"):
                q.check_files(str(base))
            (base/"features/rna_kmer.pt").touch()
            paths = q.check_files(str(base))
            self.assertEqual(paths["checkpoint"], str(base/"ckpt/hisca_demo.pt"))

    def test_live_owned_process_is_reused(self):
        from unittest.mock import Mock
        platform = q.Platform()
        platform.process = Mock()
        platform.process.poll.return_value = None
        platform.url = "http://127.0.0.1:7879"
        with patch.object(q.subprocess, "Popen") as popen:
            self.assertEqual(platform.start(), platform.url)
            popen.assert_not_called()
        platform.process = None

if __name__ == "__main__":
    unittest.main()
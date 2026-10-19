"""Check deployment inputs without importing Torch or starting the model."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def validate_runtime(data_dir=None, feat_dir=None, checkpoint=None, edge_csv=None):
    data = Path(data_dir or os.environ.get("NCDRUG_DATA_DIR", ROOT / "data"))
    feat = Path(feat_dir or os.environ.get("NCDRUG_FEAT_DIR", ROOT / "features"))
    ckpt = Path(checkpoint or os.environ.get("NCDRUG_DEMO_CKPT",
                Path(os.environ.get("NCDRUG_CKPT_DIR", ROOT / "ckpt")) / "hisca_demo.pt"))
    edge = Path(edge_csv or os.environ.get("NCDRUG_EDGE_CSV", data / "merged_clean.csv"))
    required = [edge, data / "drugs.csv", ckpt, ROOT / "home.html",
                ROOT / "assets" / "cover_rna_drug.png",
                ROOT / "assets" / "architecture_latest.png"]
    required += [feat / name for name in ("rna_feats.pt", "drug_feats.pt", "target_feats.pt")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing deployment files:\n  " + "\n  ".join(missing)
                                + "\nSee docs/LOCAL_DEPLOYMENT_zh.md. "
                                "Dataset/features/checkpoints are not bundled with this repository.")
    return {"data_dir": str(data), "feat_dir": str(feat), "checkpoint": str(ckpt)}

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        paths = validate_runtime()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print("Required deployment files found (existence check only).")
    for key, value in paths.items():
        print(f"{key}: {value}")
    print("The checkpoint and feature shapes are validated when the model loads.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

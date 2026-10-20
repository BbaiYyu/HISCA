
from pathlib import Path, PurePosixPath
import hashlib
import shutil
import tempfile
import zipfile

BUNDLE_FILES = (
    "data/drugs.csv", "data/merged_clean.csv", "data/ncRNA.csv", "data/drug_targets.csv",
    "features/rna_feats.pt", "features/drug_feats.pt", "features/target_feats.pt",
    "features/rna_kmer.pt", "features/drug_extra.pt", "features/drug_graphs.pt",
    "ckpt/hisca_demo.pt", "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json",
    "tokenizer/special_tokens_map.json", "tokenizer/added_tokens.json",
    "tokenizer/vocab.json", "tokenizer/merges.txt")

def validate_bundle(base):
    base = Path(base)
    missing = [name for name in BUNDLE_FILES if not (base / name).is_file()
               or (base / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Incomplete HISCA model bundle:\n  " + "\n  ".join(missing))
    return base.resolve()

def import_zip(source, home):
    source, home = Path(source).resolve(), Path(home)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    dest = home / "bundles" / digest.hexdigest()[:20]
    if dest.exists():
        return validate_bundle(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="import-", dir=str(dest.parent)))
    try:
        with zipfile.ZipFile(source) as archive:
            names = {}
            for info in archive.infolist():
                name = info.filename.replace("\\", "/")
                parts = PurePosixPath(name).parts
                if name.startswith("/") or ".." in parts or any(":" in p for p in parts):
                    raise ValueError("Unsafe path in ZIP: " + info.filename)
                if name in names:
                    raise ValueError("Duplicate path in ZIP: " + name)
                names[name] = info
            anchors = [n[:-len("ckpt/hisca_demo.pt")] for n in names
                       if n == "ckpt/hisca_demo.pt" or n.endswith("/ckpt/hisca_demo.pt")]
            if len(anchors) != 1:
                raise ValueError("ZIP must contain exactly one HISCA model bundle.")
            prefix = anchors[0]
            for relative in BUNDLE_FILES:
                if prefix + relative not in names:
                    raise FileNotFoundError("Missing bundle file in ZIP: " + relative)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(names[prefix + relative]) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        validate_bundle(stage)
        stage.rename(dest)
        return dest.resolve()
    finally:
        if stage.exists():

            stage.resolve().relative_to((home / "bundles").resolve())
            if not stage.name.startswith("import-"):
                raise RuntimeError("Unexpected staging directory")
            shutil.rmtree(stage)

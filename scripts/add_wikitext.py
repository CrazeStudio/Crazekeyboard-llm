#!/usr/bin/env python3
"""Add WikiText-103 to an existing CrazeLM corpus without duplicating lines.

Source: Salesforce/wikitext, wikitext-103-raw-v1.
The source card lists CC-BY-SA-3.0 and GFDL metadata; its prose also refers
to CC BY-SA 4.0. Review the upstream card and original Wikipedia terms before
redistribution. This script only prepares text; it does not grant a new licence.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
SHARDS = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 100_000:
        with target.open("rb") as f:
            if f.read(4) == b"PAR1":
                print(f"Using cached {target.name}")
                return
        target.unlink()
    if shutil.which("curl"):
        subprocess.run([
            "curl", "-L", "--fail", "--show-error", "--retry", "4",
            "--retry-all-errors", "--connect-timeout", "30",
            "--max-time", "900", "-o", str(target), url
        ], check=True)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "CrazeLM/1.0"})
        with urllib.request.urlopen(req, timeout=900) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    with target.open("rb") as f:
        if f.read(4) != b"PAR1":
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Download is not a valid Parquet file: {target}")


def clean(text):
    text = " ".join(str(text).replace("@-@", "-").replace("@,@", ",").replace("@.@", ".").split())
    if len(text) < 25 or not any(c.isalpha() for c in text):
        return ""
    if text.startswith(("=", "<unk>", "http://", "https://")):
        return ""
    if text.count("|") > 8:
        return ""
    return text


def main():
    p = argparse.ArgumentParser(description="Add licensed WikiText-103 English text to CrazeLM splits.")
    p.add_argument("--corpus_dir", default="/kaggle/working/crazelm_data")
    p.add_argument("--cache_dir", default="/kaggle/working/wikitext_downloads")
    p.add_argument("--min_added_train_chars", type=int, default=100_000_000)
    args = p.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("PyArrow is required. In Kaggle it is usually preinstalled; install pyarrow only if missing.") from exc

    root = Path(args.corpus_dir)
    cache = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "DATASET_MANIFEST.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    # SQLite gives exact, disk-backed de-duplication without retaining millions
    # of strings in RAM. It is shared across WikiText shards within each split.
    totals = {}
    for split, names in SHARDS.items():
        out_dir = root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "wikitext103.txt"
        db_path = cache / f"dedup_{split}.sqlite"
        cache.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(db_path)
        db.execute("CREATE TABLE IF NOT EXISTS seen (k TEXT PRIMARY KEY)")
        added_lines = added_chars = skipped = 0
        with output.open("w", encoding="utf-8") as dst:
            for name in names:
                parquet_path = cache / name
                download(f"{BASE}/{name}?download=true", parquet_path)
                parquet = pq.ParquetFile(parquet_path)
                if "text" not in parquet.schema.names:
                    raise RuntimeError(f"Expected text column in {name}; got {parquet.schema.names}")
                for batch in parquet.iter_batches(batch_size=8192, columns=["text"]):
                    for value in batch.column(0).to_pylist():
                        line = clean(value or "")
                        if not line:
                            skipped += 1
                            continue
                        key = line.casefold()
                        try:
                            db.execute("INSERT INTO seen(k) VALUES (?)", (key,))
                        except sqlite3.IntegrityError:
                            skipped += 1
                            continue
                        dst.write(line + "\n")
                        added_lines += 1
                        added_chars += len(line)
                    db.commit()
        db.close()
        totals[split] = {
            "unique_lines_added": added_lines,
            "characters_added": added_chars,
            "filtered_or_duplicate_rows": skipped,
            "file": str(output),
        }
        print(f"{split}: added {added_lines:,} unique lines, {added_chars:,} characters; filtered/duplicates {skipped:,}")

    if totals["train"]["characters_added"] < args.min_added_train_chars:
        raise SystemExit(
            f"WikiText train added only {totals['train']['characters_added']:,} characters; "
            f"minimum requested is {args.min_added_train_chars:,}. Inspect the dataset extraction."
        )

    manifest.setdefault("sources", [])
    manifest["sources"].append({
        "dataset": "Salesforce/wikitext",
        "configuration": "wikitext-103-raw-v1",
        "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
        "license_metadata": ["CC-BY-SA-3.0", "GFDL"],
        "license_note": "Dataset card prose also references CC BY-SA 4.0; verify upstream dataset and Wikipedia article terms. Preserve attribution and share-alike obligations. This manifest is not legal advice.",
        "splits": totals,
        "cleaning": "Whitespace normalization; removed short/nonalphabetic/markup-like lines; exact case-insensitive deduplication within each WikiText split.",
    })
    manifest["language"] = "English"
    manifest["note"] = "Source files remain separate within each split so provenance is traceable. Training scripts recursively read .txt files."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))
    print(f"Updated manifest: {manifest_path}")
    print("Next: run the corpus audit and tokenizer training only after verifying split counts and licences.")


if __name__ == "__main__":
    main()

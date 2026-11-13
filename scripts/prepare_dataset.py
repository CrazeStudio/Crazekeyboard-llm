#!/usr/bin/env python3
"""Prepare English-only WikiText-2 data from Hugging Face-hosted Parquet files.

This avoids the retired/redirecting S3 archive URL and does not import the
Hugging Face datasets package. Kaggle's existing pandas + pyarrow can read the
small Parquet split files.
"""
import argparse
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

DATASET_REPO = "EleutherAI/wikitext_document_level"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/wikitext-2-raw-v1"
SPLITS = {
    "train": "wikitext-2-raw-v1-train.parquet",
    "validation": "wikitext-2-raw-v1-validation.parquet",
}


def clean_line(value: str) -> str:
    value = value.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    value = re.sub(r"\s+", " ", value).strip()
    if not value or (value.startswith("=") and value.endswith("=")):
        return ""
    if not re.search(r"[A-Za-z]", value):
        return ""
    return value


def download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        with destination.open("rb") as stream:
            if stream.read(4) == b"PAR1":
                print(f"Using cached {destination.name}")
                return
        destination.unlink()

    if shutil.which("curl"):
        command = [
            "curl", "--location", "--fail", "--show-error",
            "--retry", "5", "--retry-all-errors", "--retry-delay", "2",
            "--connect-timeout", "30", "--max-time", "300",
            "--output", str(destination), url,
        ]
        print(f"Downloading {destination.name} from Hugging Face...")
        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {destination.name}. Check Kaggle Internet access.") from exc
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CrazeLM/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as target:
                shutil.copyfileobj(response, target)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    if not destination.exists() or destination.stat().st_size < 1024:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Download of {destination.name} was empty or too small.")
    with destination.open("rb") as stream:
        magic = stream.read(4)
    if magic != b"PAR1":
        preview = destination.read_bytes()[:200].decode("utf-8", errors="replace")
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"{destination.name} is not a Parquet file (HTTP proxy may have returned an error page). "
            f"Response preview: {preview!r}"
        )


def load_wikitext(cache_dir: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas/pyarrow is required to read the small Parquet files. "
            "Kaggle normally includes both; do not upgrade pyarrow blindly."
        ) from exc

    result = {}
    for split, filename in SPLITS.items():
        path = cache_dir / filename
        download_file(f"{BASE_URL}/{filename}?download=true", path)
        frame = pd.read_parquet(path, columns=["text"])
        result[split] = [
            cleaned for value in frame["text"].dropna().astype(str)
            if (cleaned := clean_line(value))
        ]
        if not result[split]:
            raise RuntimeError(f"No usable English text found in {split} split.")
        print(f"{split}: {len(result[split]):,} cleaned lines")
    return result


def collect_local(input_dir: Path):
    files = sorted(input_dir.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {input_dir}")
    rows = []
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            rows.extend(filter(None, (clean_line(line) for line in stream)))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare an English-only CrazeLM corpus.")
    parser.add_argument("--output_dir", default="/kaggle/working/crazelm_data")
    parser.add_argument("--input_dir", help="Optional folder of appropriately licensed English .txt files.")
    parser.add_argument("--max_train_lines", type=int, default=0)
    parser.add_argument("--max_validation_lines", type=int, default=0)
    args = parser.parse_args()
    if args.max_train_lines < 0 or args.max_validation_lines < 0:
        parser.error("line limits cannot be negative")

    output = Path(args.output_dir)
    train_dir, val_dir = output / "train", output / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        source = Path(args.input_dir)
        if not source.is_dir():
            parser.error(f"--input_dir does not exist or is not a directory: {source}")
        rows = collect_local(source)
        if len(rows) < 20:
            raise SystemExit(f"Only {len(rows)} usable English lines found; provide a larger corpus.")
        validation_ids = set(range(0, len(rows), 50))
        train = [line for i, line in enumerate(rows) if i not in validation_ids]
        validation = [line for i, line in enumerate(rows) if i in validation_ids]
        source_name = f"local:{source.resolve()}"
        source_note = "User-supplied text. Verify rights and license before training or distributing a model."
        license_name = "User supplied; not verified by this script"
    else:
        splits = load_wikitext(output / "downloads")
        train, validation = splits["train"], splits["validation"]
        source_name = DATASET_REPO
        source_note = (
            "WikiText-2 is a small Wikipedia-derived benchmark corpus, not a natural keyboard-typing corpus. "
            "Dataset card identifies CC BY-SA 3.0; retain attribution and review share-alike obligations "
            "before distributing a derived model."
        )
        license_name = "CC BY-SA 3.0 (verify current dataset card)"

    if args.max_train_lines:
        train = train[:args.max_train_lines]
    if args.max_validation_lines:
        validation = validation[:args.max_validation_lines]
    if len(train) < 20 or len(validation) < 2:
        raise SystemExit(f"Corpus too small: train={len(train)}, validation={len(validation)}")

    for folder, lines in ((train_dir, train), (val_dir, validation)):
        with (folder / "english.txt").open("w", encoding="utf-8") as target:
            target.write("\n".join(lines) + "\n")

    manifest = {
        "dataset": source_name,
        "language": "English",
        "license": license_name,
        "train_lines": len(train),
        "validation_lines": len(validation),
        "source_note": source_note,
        "format": "UTF-8 plain text, one cleaned line per example",
    }
    (output / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Prepared corpus at {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare an English-only WikiText-2 corpus for CrazeLM.

Uses curl with redirect/retry support when available, then Python urllib as a
fallback. Does not require Hugging Face datasets or pyarrow.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

WIKITEXT_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
WIKITEXT_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"


def clean_line(value: str) -> str:
    value = value.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    value = re.sub(r"\\s+", " ", value).strip()
    if not value or (value.startswith("=") and value.endswith("=")):
        return ""
    if not re.search(r"[A-Za-z]", value):
        return ""
    return value


def download_archive(archive: Path):
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest == WIKITEXT_SHA256:
            print("Using verified cached WikiText-2 archive.")
            return
        print("Cached archive checksum mismatch; downloading a fresh copy.")
        archive.unlink()

    if shutil.which("curl"):
        command = [
            "curl", "--location", "--fail", "--show-error",
            "--retry", "5", "--retry-all-errors", "--retry-delay", "2",
            "--connect-timeout", "30", "--max-time", "300",
            "--output", str(archive), WIKITEXT_URL,
        ]
        print("Downloading WikiText-2 with curl (redirects and retries enabled)...")
        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, OSError):
            archive.unlink(missing_ok=True)
            raise RuntimeError("curl could not download WikiText-2. Check Kaggle Internet access and retry.")
    else:
        print("Downloading WikiText-2 with Python urllib...")
        request = urllib.request.Request(
            WIKITEXT_URL,
            headers={"User-Agent": "Mozilla/5.0 CrazeLM-dataset-preparer/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
        except Exception:
            archive.unlink(missing_ok=True)
            raise

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != WIKITEXT_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded WikiText-2 archive failed SHA-256 verification. "
            f"Expected {WIKITEXT_SHA256}, got {digest}. The download may have been an HTML error page."
        )


def download_wikitext(cache_dir: Path):
    archive = cache_dir / "wikitext-2-raw-v1.zip"
    download_archive(archive)
    with zipfile.ZipFile(archive) as zf:
        members = {Path(name).name: name for name in zf.namelist() if name.endswith(".txt")}
        required = {"wiki.train.raw", "wiki.valid.raw"}
        if not required.issubset(members):
            raise RuntimeError(f"Unexpected WikiText archive contents: {sorted(members)}")
        result = {}
        for split, filename in (("train", "wiki.train.raw"), ("validation", "wiki.valid.raw")):
            with zf.open(members[filename]) as source:
                lines = (line.decode("utf-8", errors="ignore") for line in source)
                result[split] = list(filter(None, (clean_line(line) for line in lines)))
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
    else:
        splits = download_wikitext(output / "downloads")
        train, validation = splits["train"], splits["validation"]
        source_name = "Salesforce/WikiText-2-raw-v1"
        source_note = "WikiText-2 is a small Wikipedia-derived benchmark corpus. Dataset metadata lists CC BY-SA licensing; preserve attribution and review share-alike obligations before distributing a derived model."

    if args.max_train_lines:
        train = train[:args.max_train_lines]
    if args.max_validation_lines:
        validation = validation[:args.max_validation_lines]
    if len(train) < 20 or len(validation) < 2:
        raise SystemExit(f"Corpus too small: train={len(train)}, validation={len(validation)}")

    for folder, lines in ((train_dir, train), (val_dir, validation)):
        with (folder / "english.txt").open("w", encoding="utf-8") as target:
            target.write("\\n".join(lines) + "\\n")

    manifest = {
        "dataset": source_name,
        "language": "English",
        "train_lines": len(train),
        "validation_lines": len(validation),
        "source_note": source_note,
        "archive_sha256": WIKITEXT_SHA256 if not args.input_dir else None,
        "format": "UTF-8 plain text, one cleaned line per example",
    }
    (output / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Prepared corpus at {output}")


if __name__ == "__main__":
    main()

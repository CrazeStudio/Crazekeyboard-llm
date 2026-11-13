#!/usr/bin/env python3
"""Prepare an English-only text corpus for CrazeLM.

Default source: the official WikiText-2 raw archive. You can also pass a local
folder containing .txt files with --input_dir. No Hugging Face datasets import
is required, avoiding the large pyarrow dependency stack in Kaggle.
"""
import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path

WIKITEXT_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"


def clean_line(value: str) -> str:
    value = value.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    value = re.sub(r"\s+", " ", value).strip()
    if not value or value.startswith("=") and value.endswith("="):
        return ""
    if not re.search(r"[A-Za-z]", value):
        return ""
    return value


def collect_local(input_dir: Path):
    files = sorted(input_dir.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {input_dir}")
    rows = []
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            rows.extend(filter(None, (clean_line(line) for line in stream)))
    return rows


def download_wikitext(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "wikitext-2-raw-v1.zip"
    if not archive.exists():
        print(f"Downloading English WikiText-2 from {WIKITEXT_URL}")
        request = urllib.request.Request(WIKITEXT_URL, headers={"User-Agent": "CrazeLM-dataset-preparer/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
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


def main():
    parser = argparse.ArgumentParser(description="Prepare an English-only CrazeLM corpus.")
    parser.add_argument("--output_dir", default="/kaggle/working/crazelm_data")
    parser.add_argument("--input_dir", help="Optional folder of your own appropriately licensed English .txt files.")
    parser.add_argument("--max_train_lines", type=int, default=0, help="Optional cap; 0 keeps all lines.")
    parser.add_argument("--max_validation_lines", type=int, default=0, help="Optional cap; 0 keeps all lines.")
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
        # Deterministic 98/2 split, preserving source order.
        validation = rows[::50]
        validation_ids = set(range(0, len(rows), 50))
        train = [line for i, line in enumerate(rows) if i not in validation_ids]
        source_name = f"local:{source.resolve()}"
        source_note = "User-supplied text. Verify that you have rights to use it for model training."
    else:
        splits = download_wikitext(output / "downloads")
        train, validation = splits["train"], splits["validation"]
        source_name = "Salesforce/wikitext-2-raw-v1 (official raw archive)"
        source_note = "WikiText-2 is a small Wikipedia-derived language-model benchmark, not a representative keyboard-typing corpus. Check the source dataset terms and attribution obligations before commercial use."

    if args.max_train_lines:
        train = train[:args.max_train_lines]
    if args.max_validation_lines:
        validation = validation[:args.max_validation_lines]
    if len(train) < 20 or len(validation) < 2:
        raise SystemExit(f"Corpus too small after preparation: train={len(train)}, validation={len(validation)}")

    for folder, filename, lines in (
        (train_dir, "english.txt", train),
        (val_dir, "english.txt", validation),
    ):
        with (folder / filename).open("w", encoding="utf-8") as target:
            target.write("\n".join(lines) + "\n")

    manifest = {
        "dataset": source_name,
        "language": "English",
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

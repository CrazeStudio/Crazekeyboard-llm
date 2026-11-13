#!/usr/bin/env python3
"""Build a clean English conversational corpus from DailyDialog.

DailyDialog is an Apache-2.0-labelled Hugging Face mirror of the human-written
DailyDialog corpus. We use its existing train/validation/test split and do not
mix validation/test examples into training.
"""
import argparse
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

DATASET = "anezatra/dailydialog"
API_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/data?recursive=true"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
SPLITS = ("train", "validation", "test")


def request_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "CrazeLM/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        with path.open("rb") as f:
            if f.read(4) == b"PAR1":
                print(f"Using cached {path.name}")
                return
        path.unlink()
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--fail", "--show-error", "--retry", "4",
               "--retry-all-errors", "--retry-delay", "2", "--connect-timeout", "30",
               "--max-time", "300", "-o", str(path), url]
        subprocess.run(cmd, check=True)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "CrazeLM/1.0"})
        with urllib.request.urlopen(req, timeout=180) as src, path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    with path.open("rb") as f:
        if f.read(4) != b"PAR1":
            preview = path.read_bytes()[:160].decode("utf-8", "replace")
            path.unlink(missing_ok=True)
            raise RuntimeError(f"{path.name} is not Parquet. Response: {preview!r}")


def clean_sentence(value):
    value = str(value).replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    value = re.sub(r"<[^>]{1,40}>", " ", value)
    value = re.sub(r"\\s+", " ", value).strip()
    if not value or not re.search(r"[A-Za-z]", value):
        return ""
    if len(value) < 2 or len(value) > 1000:
        return ""
    # Drop obvious markup/metadata and malformed fragments.
    if value.startswith(("http://", "https://", "{", "[")):
        return ""
    if re.fullmatch(r"[\\W_\\d]+", value):
        return ""
    return value


def split_dialog(value):
    if value is None:
        return []
    raw = str(value).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if isinstance(x, (str, int, float))]
        if isinstance(parsed, dict):
            for key in ("messages", "dialog", "conversation", "utterances", "text"):
                if key in parsed:
                    return split_dialog(parsed[key])
    except (json.JSONDecodeError, TypeError):
        pass
    # ShareGPT-like text and newline-delimited dialogue formats.
    parts = re.split(r"\\n+|(?=(?:Human|Assistant|User|Bot|Speaker [0-9]+)\\s*:)", raw)
    return [p for p in parts if p.strip()]


def read_split(files, split, cache):
    import pandas as pd
    paths = [x for x in files if re.search(rf"/{split}[^/]*\\.parquet$", "/" + x)]
    if not paths:
        raise RuntimeError(f"No Parquet file found for {split} in {DATASET}")
    lines = []
    for relative in paths:
        local = cache / Path(relative).name
        download(f"{BASE_URL}/{relative}?download=true", local)
        frame = pd.read_parquet(local)
        text_columns = [c for c in ("dialog", "text", "conversation", "messages") if c in frame.columns]
        if not text_columns:
            raise RuntimeError(f"Unexpected columns in {local.name}: {list(frame.columns)}")
        for value in frame[text_columns[0]].dropna():
            for item in split_dialog(value):
                cleaned = clean_sentence(item)
                if cleaned:
                    lines.append(cleaned)
    # Stable de-duplication keeps the corpus compact and reduces repeated targets.
    seen, unique = set(), []
    for line in lines:
        key = line.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(line)
    return unique


def write_lines(folder, lines):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "english.txt").open("w", encoding="utf-8") as out:
        out.write("\\n".join(lines) + "\\n")


def main():
    p = argparse.ArgumentParser(description="Prepare English DailyDialog data for CrazeLM.")
    p.add_argument("--output_dir", default="/kaggle/working/crazelm_data")
    p.add_argument("--min_train_lines", type=int, default=50000)
    p.add_argument("--max_train_lines", type=int, default=0)
    p.add_argument("--max_validation_lines", type=int, default=0)
    p.add_argument("--max_test_lines", type=int, default=0)
    args = p.parse_args()
    if min(args.min_train_lines, args.max_train_lines, args.max_validation_lines, args.max_test_lines) < 0:
        p.error("line limits cannot be negative")
    out = Path(args.output_dir)
    cache = out / "downloads"
    listing = request_json(API_URL)
    files = [item["path"] for item in listing if item.get("type") == "file" and item.get("path", "").endswith(".parquet")]
    splits = {name: read_split(files, name, cache) for name in SPLITS}
    for name, limit in (("train", args.max_train_lines), ("validation", args.max_validation_lines), ("test", args.max_test_lines)):
        if limit:
            splits[name] = splits[name][:limit]
    if len(splits["train"]) < args.min_train_lines:
        raise SystemExit(
            f"Only {len(splits['train']):,} unique training utterances were extracted; "
            f"minimum requested is {args.min_train_lines:,}. Check the downloaded schema or lower --min_train_lines."
        )
    if len(splits["validation"]) < 100 or len(splits["test"]) < 100:
        raise SystemExit("Validation/test split unexpectedly small; refusing to create a misleading evaluation set.")
    for name, lines in splits.items():
        write_lines(out / name, lines)
        print(f"{name}: {len(lines):,} unique English utterances")
    manifest = {
        "dataset": DATASET,
        "dataset_description": "DailyDialog: human-written multi-turn everyday conversations",
        "license_label_on_huggingface_mirror": "Apache-2.0; verify upstream rights and attribution before redistribution",
        "language": "English",
        "train_lines": len(splits["train"]),
        "validation_lines": len(splits["validation"]),
        "test_lines": len(splits["test"]),
        "deduplication": "case-insensitive exact line deduplication within each split",
        "note": "The dataset is conversational but relatively small; use additional properly licensed data for production keyboard coverage."
    }
    (out / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Prepared corpus: {out}")


if __name__ == "__main__":
    main()

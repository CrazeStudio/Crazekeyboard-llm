import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "ai4bharat/samanantar"
LANGUAGES = ("bn", "hi")
LICENSE = "CC BY-NC 4.0"

def main():
    parser = argparse.ArgumentParser(description="Download a balanced English/Bengali/Hindi CrazeLM corpus.")
    parser.add_argument("--output_dir", default="/kaggle/working/crazelm_data")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--max_pairs_per_language", type=int, default=100000)
    parser.add_argument("--val_every", type=int, default=50, help="Every Nth source row goes to validation (default 2%).")
    parser.add_argument("--accept_cc_by_nc", action="store_true", help="Acknowledge the source dataset's non-commercial license.")
    args = parser.parse_args()

    if not args.accept_cc_by_nc:
        parser.error("Samanantar is CC BY-NC 4.0. Pass --accept_cc_by_nc only if non-commercial use is acceptable for your project.")

    root = Path(args.output_dir)
    train_dir, val_dir = root / "train", root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    handles = {}
    counts = {split: {lang: 0 for lang in ["en", *args.languages]} for split in ("train", "validation")}
    try:
        for split_dir, split_name in ((train_dir, "train"), (val_dir, "validation")):
            for lang in ["en", *args.languages]:
                handles[(split_name, lang)] = (split_dir / f"{lang}.txt").open("w", encoding="utf-8")

        for language in args.languages:
            print(f"Streaming {DATASET_ID} [{language}]...")
            dataset = load_dataset(DATASET_ID, language, split="train", streaming=True)
            seen = 0
            for row in tqdm(dataset, total=args.max_pairs_per_language, desc=language):
                if seen >= args.max_pairs_per_language:
                    break
                seen += 1
                src = " ".join(str(row.get("src", "")).split())
                tgt = " ".join(str(row.get("tgt", "")).split())
                if len(src) < 2 or len(tgt) < 2:
                    continue
                split_name = "validation" if int(row.get("idx", seen)) % args.val_every == 0 else "train"
                handles[(split_name, "en")].write(src + "\n")
                handles[(split_name, language)].write(tgt + "\n")
                counts[split_name]["en"] += 1
                counts[split_name][language] += 1

            if seen == 0:
                raise RuntimeError(f"No rows received for language {language}. Check dataset access/configuration.")
            print(f"{language}: processed {seen:,} pairs")

    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "dataset": DATASET_ID,
        "license": LICENSE,
        "languages": ["en", *args.languages],
        "max_pairs_per_language": args.max_pairs_per_language,
        "validation_rule": f"row idx modulo {args.val_every} equals zero",
        "counts": counts,
        "attribution": "Ramesh et al., Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages, TACL 2022. https://doi.org/10.1162/tacl_a_00452",
        "license_notice": "CC BY-NC 4.0. Non-commercial use only unless separate permission is obtained. Review license obligations before training, distributing, or commercializing a model.",
    }
    (root / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nTraining text: {train_dir}\nValidation text: {val_dir}")

if __name__ == "__main__":
    main()

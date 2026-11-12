#!/usr/bin/env python3
"""
Step 1.2: Full Corpus Acquisition, Benchmark Decontamination,
RAM-Safe Exact + Paragraph/Shingle Deduplication, and Train/Val/Tokenizer Split Builder.

Designed for Local Laptop (WSL):
- Stores heavy files on /mnt/d/odia_llm_data (symlinked to data/processed)
- Uses local Parquet reading via pyarrow for maximum speed & zero HTTP thread crashes
- Uses 64-bit xxhash integer sets so 10M+ documents consume < 400 MB RAM.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import xxhash
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.step_03_eda_secondary_and_eval import load_dotenv_wsl_safe
from src.hygiene import clean_and_filter_document

ARTIFACT_PATH = Path("artifacts/step_12_corpus_build_report.json")
RE_WS = re.compile(r"\s+")
RE_NON_ALNUM_ODIA = re.compile(r"[^\w\u0b00-\u0b7f]+")


def normalize_for_hash(text: str) -> str:
    """Lowercases and strips punctuation/whitespace for canonical hashing."""
    return RE_NON_ALNUM_ODIA.sub("", text.lower())


def fast_hash64(s: str) -> int:
    """UTF-8 encodes string and returns 64-bit unsigned integer xxhash."""
    return xxhash.xxh64_intdigest(s.encode("utf-8"))


def extract_13word_ngrams(text: str) -> list:
    """Extracts 64-bit xxhash digests of normalized 13-word windows."""
    words = RE_WS.sub(" ", text.strip().lower()).split()
    if len(words) < 13:
        if len(words) >= 7:
            # For short benchmark questions (7-12 words), hash the full normalized question
            return [fast_hash64(" ".join(words))]
        return []
    hashes = []
    # Step by 1 for eval set, or step by 2 for fast document scanning
    for i in range(len(words) - 12):
        window = " ".join(words[i : i + 13])
        hashes.append(fast_hash64(window))
    return hashes


def build_eval_decontamination_set(token: str) -> set:
    """
    Downloads saidutta69/odia-eval-benchmark (1 parquet file, 125K rows)
    and builds a 64-bit integer hash set of all 13-word windows across
    prompt_odia, reference_odia, prompt, and reference.
    """
    dataset_id = "saidutta69/odia-eval-benchmark"
    print(f"\n[Stage 1/5] Building Decontamination Index from {dataset_id} ...")
    parquet_path = hf_hub_download(
        repo_id=dataset_id,
        filename="data/odia-00000-of-00001.parquet",
        repo_type="dataset",
        token=token,
    )
    table = pq.read_table(
        parquet_path,
        columns=["prompt_odia", "reference_odia", "prompt", "reference"],
    )
    eval_hashes = set()
    num_rows = table.num_rows

    for col_name in ["prompt_odia", "reference_odia", "prompt", "reference"]:
        col_values = table.column(col_name).to_pylist()
        for val in col_values:
            if val and isinstance(val, str) and len(val) >= 25:
                nfc_val = unicodedata.normalize("NFC", val)
                # Exact normalized string hash
                norm_str = normalize_for_hash(nfc_val)
                if len(norm_str) >= 20:
                    eval_hashes.add(fast_hash64(norm_str))
                # 13-word contiguous n-gram hashes
                for h in extract_13word_ngrams(nfc_val):
                    eval_hashes.add(h)

    print(f"            Indexed {len(eval_hashes):,} unique evaluation n-gram/passage hashes from {num_rows:,} eval rows.")
    return eval_hashes


def is_contaminated_or_duplicate(
    cleaned_text: str,
    eval_hashes: set,
    seen_doc_hashes: set,
    seen_shingle_hashes: set,
) -> tuple:
    """
    Checks:
    1. Exact document duplicate (64-bit xxhash of normalized text)
    2. Near-duplicate fuzzy collision (64-bit xxhash of first 150-char & middle 150-char normalized shingles)
    3. Benchmark contamination (13-word window match against eval_hashes)
    Returns (is_rejected: bool, reason: str)
    """
    norm_full = normalize_for_hash(cleaned_text)
    if len(norm_full) < 20:
        return True, "dedup_too_short_norm"

    # 1. Exact document hash
    doc_h = fast_hash64(norm_full)
    if doc_h in seen_doc_hashes:
        return True, "duplicate_exact_doc"
    if doc_h in eval_hashes:
        return True, "contaminated_eval_exact"

    # 2. Fuzzy shingle signature (head + mid + tail 120-char normalized windows)
    # Catches syndicated wire articles that only differ by a dateline or footer
    shingle_keys = []
    if len(norm_full) >= 180:
        head_s = fast_hash64("H:" + norm_full[:120])
        mid_idx = len(norm_full) // 2
        mid_s = fast_hash64("M:" + norm_full[mid_idx - 60 : mid_idx + 60])
        # If BOTH head and middle 120-char normalized blocks match an existing doc, it is a near-duplicate
        combo_s = fast_hash64(f"{head_s}_{mid_s}")
        if combo_s in seen_shingle_hashes:
            return True, "duplicate_fuzzy_shingle"
        shingle_keys.append(combo_s)

    # 3. Benchmark 13-word window scan (stride=2 for speed, covers every 14-word overlap)
    words = RE_WS.sub(" ", cleaned_text.strip().lower()).split()
    n_words = len(words)
    if n_words >= 13:
        for i in range(0, n_words - 12, 2):
            w_hash = fast_hash64(" ".join(words[i : i + 13]))
            if w_hash in eval_hashes:
                return True, "contaminated_eval_13gram"
    elif n_words >= 7:
        w_hash = fast_hash64(" ".join(words))
        if w_hash in eval_hashes:
            return True, "contaminated_eval_short"

    # Register document in seen sets
    seen_doc_hashes.add(doc_h)
    for sk in shingle_keys:
        seen_shingle_hashes.add(sk)

    return False, "clean"


def main():
    parser = argparse.ArgumentParser(description="Step 1.2 Full Corpus Builder")
    parser.add_argument("--storage-dir", type=str, default="/mnt/d/odia_llm_data", help="Disk path for large outputs")
    parser.add_argument("--target-total-words", type=int, default=1_550_000_000, help="Target clean words (~2.5B tokens)")
    parser.add_argument("--sangraha-max-docs", type=int, default=2_200_000, help="Max docs to pull from Sangraha ori")
    parser.add_argument("--english-target-words", type=int, default=75_000_000, help="Target clean words for 5% English anchor")
    parser.add_argument("--tokenizer-sample-mb", type=int, default=1600, help="Size of tokenizer_train.txt in MB")
    parser.add_argument("--val-ratio", type=float, default=0.005, help="Validation split ratio (0.5%)")
    args = parser.parse_args()

    t0 = time.time()
    load_dotenv_wsl_safe(".env")
    token = os.environ.get("HF_TOKEN")

    # Set up output storage on /mnt/d/odia_llm_data/processed and symlink to data/processed
    storage_root = Path(args.storage_dir) / "processed"
    storage_root.mkdir(parents=True, exist_ok=True)

    local_processed = Path("data/processed")
    if local_processed.exists() and not local_processed.is_symlink():
        try:
            local_processed.rmdir()
        except OSError:
            pass
    if not local_processed.exists():
        local_processed.symlink_to(storage_root, target_is_directory=True)

    train_path = storage_root / "train_docs.jsonl"
    val_path = storage_root / "val_docs.jsonl"
    tok_txt_path = storage_root / "tokenizer_train.txt"

    print(f"Output storage directory : {storage_root}")
    print(f"Symlinked local path     : {local_processed.resolve()}")

    # Stage 1: Build Eval Decontamination Set
    eval_hashes = build_eval_decontamination_set(token=token)
    seen_doc_hashes = set()
    seen_shingle_hashes = set()

    stats = Counter()
    source_word_counts = Counter()
    source_doc_counts = Counter()

    rng = random.Random(42)
    tok_bytes_written = 0
    tok_target_bytes = args.tokenizer_sample_mb * 1024 * 1024

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val, \
         open(tok_txt_path, "w", encoding="utf-8") as f_tok:

        def process_and_write_record(raw_text: str, source_tag: str, is_en: bool = False):
            nonlocal tok_bytes_written
            stats["raw_docs_seen"] += 1

            cleaned, reason = clean_and_filter_document(
                raw_text, source_name=source_tag, is_english_anchor=is_en
            )
            if cleaned is None:
                stats[reason] += 1
                return 0

            is_bad, dedup_reason = is_contaminated_or_duplicate(
                cleaned, eval_hashes, seen_doc_hashes, seen_shingle_hashes
            )
            if is_bad:
                stats[dedup_reason] += 1
                return 0

            w_count = len(cleaned.split())
            c_count = len(cleaned)
            stats["clean_docs_kept"] += 1
            stats["clean_words_kept"] += w_count
            stats["clean_chars_kept"] += c_count
            source_word_counts[source_tag] += w_count
            source_doc_counts[source_tag] += 1

            record_json = json.dumps(
                {"text": cleaned, "source": source_tag, "words": w_count},
                ensure_ascii=False,
            ) + "\n"

            # Deterministic 0.5% validation split vs 99.5% train split
            if rng.random() < args.val_ratio:
                f_val.write(record_json)
                stats["val_docs"] += 1
                stats["val_words"] += w_count
            else:
                f_train.write(record_json)
                stats["train_docs"] += 1
                stats["train_words"] += w_count

                # Subsample ~20% of clean training docs into tokenizer_train.txt until target MB reached
                if tok_bytes_written < tok_target_bytes and rng.random() < 0.22:
                    line_bytes = (cleaned + "\n\n").encode("utf-8")
                    f_tok.write(cleaned + "\n\n")
                    tok_bytes_written += len(line_bytes)
                    stats["tokenizer_sample_docs"] += 1

            return w_count

        # ------------------------------------------------------------------
        # Stage 2: Process Primary Corpus (saidutta69/odia_pretrain_dataset_v2)
        # Download 32 parquet files locally for 20x faster disk streaming
        # ------------------------------------------------------------------
        primary_id = "saidutta69/odia_pretrain_dataset_v2"
        print(f"\n[Stage 2/5] Downloading & processing all 32 Parquet shards of {primary_id} ...")
        primary_dir = snapshot_download(
            repo_id=primary_id,
            repo_type="dataset",
            allow_patterns=["*.parquet"],
            token=token,
        )
        parquet_files = sorted(Path(primary_dir).rglob("*.parquet"))
        print(f"            Found {len(parquet_files)} local Parquet shards.")

        for pf_idx, pf in enumerate(parquet_files, 1):
            pf_file = pq.ParquetFile(pf)
            for batch in pf_file.iter_batches(batch_size=10_000, columns=["text", "source"]):
                texts = batch.column("text").to_pylist()
                sources = batch.column("source").to_pylist()
                for txt, src in zip(texts, sources):
                    process_and_write_record(txt, f"v2:{src or 'unknown'}", is_en=False)
            print(
                f"            Shard {pf_idx:02d}/{len(parquet_files):02d} done | "
                f"Kept Docs: {stats['clean_docs_kept']:,} | "
                f"Clean Words: {stats['clean_words_kept']:,} | "
                f"Eval Contam Caught: {stats['contaminated_eval_13gram'] + stats['contaminated_eval_exact']:,}"
            )

        # ------------------------------------------------------------------
        # Stage 3: Process Secondary Odia Corpus (ai4bharat/sangraha verified/ori)
        # Download parquet files in verified/ori locally for zero-crash reading
        # ------------------------------------------------------------------
        sangraha_id = "ai4bharat/sangraha"
        print(f"\n[Stage 3/5] Downloading & processing {sangraha_id} (verified/ori) ...")
        sangraha_dir = snapshot_download(
            repo_id=sangraha_id,
            repo_type="dataset",
            allow_patterns=["verified/ori/*.parquet"],
            token=token,
        )
        sangraha_files = sorted(Path(sangraha_dir).rglob("*.parquet"))
        print(f"            Found {len(sangraha_files)} Sangraha (ori) Parquet shards.")

        sangraha_seen = 0
        for sf_idx, sf in enumerate(sangraha_files, 1):
            if sangraha_seen >= args.sangraha_max_docs or stats["clean_words_kept"] >= (args.target_total_words - args.english_target_words):
                break
            sf_file = pq.ParquetFile(sf)
            for batch in sf_file.iter_batches(batch_size=10_000, columns=["text"]):
                texts = batch.column("text").to_pylist()
                for txt in texts:
                    sangraha_seen += 1
                    process_and_write_record(txt, "sangraha:verified_ori", is_en=False)
                    if sangraha_seen >= args.sangraha_max_docs or stats["clean_words_kept"] >= (args.target_total_words - args.english_target_words):
                        break
                if sangraha_seen >= args.sangraha_max_docs or stats["clean_words_kept"] >= (args.target_total_words - args.english_target_words):
                    break
            print(
                f"            Sangraha Shard {sf_idx:02d}/{len(sangraha_files):02d} | "
                f"Total Clean Words: {stats['clean_words_kept']:,} | "
                f"Cross-Dups Caught: {stats['duplicate_exact_doc'] + stats['duplicate_fuzzy_shingle']:,}"
            )

        # ------------------------------------------------------------------
        # Stage 4: Process 5% English/Numeric Anchor (FineWeb-Edu 1 shard)
        # Download just 1 single parquet shard (~2 GB) instead of streaming
        # ------------------------------------------------------------------
        fw_id = "HuggingFaceFW/fineweb-edu"
        print(f"\n[Stage 4/5] Downloading 1 Parquet shard of {fw_id} for the 5% English Anchor ...")
        fw_Shard_path = hf_hub_download(
            repo_id=fw_id,
            filename="sample/10BT/000_00000.parquet",
            repo_type="dataset",
            token=token,
        )
        fw_file = pq.ParquetFile(fw_Shard_path)
        en_words_collected = 0
        for batch in fw_file.iter_batches(batch_size=5_000, columns=["text"]):
            texts = batch.column("text").to_pylist()
            for txt in texts:
                w_added = process_and_write_record(txt, "anchor:fineweb_edu_en", is_en=True)
                en_words_collected += w_added
                if en_words_collected >= args.english_target_words:
                    break
            if en_words_collected >= args.english_target_words:
                break
        print(f"            English Anchor complete: {en_words_collected:,} clean words added.")

    elapsed = round(time.time() - t0, 2)
    tok_mb = round(tok_bytes_written / (1024 * 1024), 2)

    report = {
        "storage_root": str(storage_root),
        "train_jsonl_path": str(train_path),
        "val_jsonl_path": str(val_path),
        "tokenizer_train_txt_path": str(tok_txt_path),
        "tokenizer_train_size_mb": tok_mb,
        "counters": dict(stats),
        "source_doc_counts": dict(source_doc_counts),
        "source_word_counts": dict(source_word_counts),
        "estimated_total_tokens_at_1_6_fertility": int(stats["clean_words_kept"] * 1.6),
        "elapsed_seconds": elapsed,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== STEP 1.2 VERIFICATION SUMMARY ===")
    print(f"Raw Docs Processed        : {stats['raw_docs_seen']:,}")
    print(f"Clean Docs Retained       : {stats['clean_docs_kept']:,} (Train: {stats['train_docs']:,} | Val: {stats['val_docs']:,})")
    print(f"Total Clean Words         : {stats['clean_words_kept']:,} (~{int(stats['clean_words_kept'] * 1.6):,} estimated tokens)")
    print(f"Exact + Fuzzy Dups Removed: {stats['duplicate_exact_doc']:,} exact | {stats['duplicate_fuzzy_shingle']:,} fuzzy")
    print(f"Eval Benchmark Decontam   : {stats['contaminated_eval_13gram'] + stats['contaminated_eval_exact'] + stats['contaminated_eval_short']:,} contaminated docs removed")
    print(f"Source Word Breakdown     : {dict(source_word_counts.most_common(6))}")
    print(f"Tokenizer Sample Written  : {tok_txt_path} ({tok_mb} MB, {stats['tokenizer_sample_docs']:,} docs)")
    print(f"Output Files              : {train_path} & {val_path}")
    print(f"Total Runtime             : {elapsed}s ({round(elapsed/60, 1)} mins)")
    print("=====================================")


if __name__ == "__main__":
    main()

# python3 scripts/step_12_build_corpus.py
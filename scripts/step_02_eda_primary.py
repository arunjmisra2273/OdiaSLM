#!/usr/bin/env python3
"""
Step 0.2: Streaming EDA & Orthographic Audit for saidutta69/odia_pretrain_dataset_v2
Runs cleanly on a local laptop with near-zero disk footprint using HF Streaming.
"""

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

DATASET_ID = "saidutta69/odia_pretrain_dataset_v2"
ARTIFACT_PATH = Path("artifacts/step_02_eda_primary.json")

# Regex patterns for artifact & boilerplate detection
RE_INST = re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|### Instruction:|### Response:", re.IGNORECASE)
RE_HTML = re.compile(r"<(html|body|div|script|style|table|p|a|span)\b[^>]*>", re.IGNORECASE)
RE_MULTI_SPACE = re.compile(r"\s+")


def analyze_odia_orthography(text: str) -> dict:
    """Computes Odia script density and special Unicode character counts."""
    odia_chars = 0
    latin_chars = 0
    digit_chars = 0
    total_non_ws = 0

    zws_count = text.count("\u200b")   # Zero-Width Space (noise)
    zwnj_count = text.count("\u200c")  # Zero-Width Non-Joiner (morphological)
    zwj_count = text.count("\u200d")   # Zero-Width Joiner (morphological)
    halant_count = text.count("\u0b4d")  # Odia Virama / Halant
    nukta_count = text.count("\u0b3c") + text.count("\u0b5c") + text.count("\u0b5d")

    for ch in text:
        if ch.isspace():
            continue
        total_non_ws += 1
        cp = ord(ch)
        # Odia block (0B00-0B7F) + Danda/Double Danda (0964-0965)
        if (0x0B00 <= cp <= 0x0B7F) or cp in (0x0964, 0x0965):
            odia_chars += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            latin_chars += 1
        elif ch.isdigit():
            digit_chars += 1

    odia_ratio = (odia_chars / total_non_ws) if total_non_ws > 0 else 0.0
    latin_ratio = (latin_chars / total_non_ws) if total_non_ws > 0 else 0.0

    return {
        "total_non_ws": total_non_ws,
        "odia_chars": odia_chars,
        "odia_ratio": odia_ratio,
        "latin_ratio": latin_ratio,
        "digit_chars": digit_chars,
        "zws": zws_count,
        "zwnj": zwnj_count,
        "zwj": zwj_count,
        "halant": halant_count,
        "nukta": nukta_count,
    }


def fetch_repo_metadata(dataset_id: str) -> dict:
    """Queries Hugging Face Hub API for file count, parquet sizes, and license."""
    api = HfApi()
    try:
        info = api.dataset_info(dataset_id, files_metadata=True)
        siblings = info.siblings or []
        parquet_files = [f for f in siblings if f.rfilename.endswith(".parquet")]
        total_bytes = sum((f.size or 0) for f in siblings)
        parquet_bytes = sum((f.size or 0) for f in parquet_files)
        return {
            "id": info.id,
            "license": getattr(info, "card_data", {}).get("license", "unknown") if info.card_data else "unknown",
            "total_files": len(siblings),
            "parquet_files": len(parquet_files),
            "total_size_gb": round(total_bytes / (1024 ** 3), 3),
            "parquet_size_gb": round(parquet_bytes / (1024 ** 3), 3),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Step 0.2 Streaming EDA")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Number of streamed rows to profile")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[1/3] Fetching Hub metadata for {DATASET_ID} ...")
    repo_meta = fetch_repo_metadata(DATASET_ID)
    print(f"      Repo Metadata: {json.dumps(repo_meta)}")

    print(f"[2/3] Initializing streaming dataset ({args.sample_size:,} rows) ...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    # Metrics accumulators
    char_lengths = []
    word_lengths = []
    odia_ratios = []
    latin_ratios = []
    short_line_ratios = []

    seen_hashes = set()
    duplicate_count = 0
    null_or_empty_count = 0
    under_150_chars = 0
    under_25_words = 0
    low_odia_ratio_count = 0  # < 0.65
    high_short_lines_count = 0  # > 0.35 of lines have < 4 words
    inst_artifact_count = 0
    html_artifact_count = 0
    nfc_diff_count = 0
    nfkc_destructive_diff_count = 0

    unicode_totals = Counter()
    source_counter = Counter()
    column_names = None
    text_col = None
    sample_records = []

    for idx, row in enumerate(tqdm(ds, total=args.sample_size, desc="Streaming EDA")):
        if idx >= args.sample_size:
            break

        if column_names is None:
            column_names = list(row.keys())
            # Automatically identify primary text column
            for candidate in ["text", "content", "document", "odia", "sentence"]:
                if candidate in row:
                    text_col = candidate
                    break
            if text_col is None:
                text_col = column_names[0]
            print(f"      Detected columns: {column_names} | Using text column: '{text_col}'")

        # Track source/domain metadata if present
        for src_col in ["source", "domain", "subset", "dataset", "origin"]:
            if src_col in row and row[src_col] is not None:
                source_counter[str(row[src_col])] += 1
                break

        raw_text = row.get(text_col)
        if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
            null_or_empty_count += 1
            continue

        # Check NFC vs NFKC behavior on raw Odia text
        nfc_text = unicodedata.normalize("NFC", raw_text)
        nfkc_text = unicodedata.normalize("NFKC", nfc_text)
        if raw_text != nfc_text:
            nfc_diff_count += 1
        if nfc_text != nfkc_text:
            nfkc_destructive_diff_count += 1

        # Exact hash deduplication check (normalized whitespace/case)
        norm_for_hash = RE_MULTI_SPACE.sub(" ", nfc_text.strip().lower())
        h = hashlib.sha256(norm_for_hash.encode("utf-8")).digest()
        if h in seen_hashes:
            duplicate_count += 1
        else:
            seen_hashes.add(h)

        c_len = len(nfc_text)
        words = nfc_text.split()
        w_len = len(words)

        char_lengths.append(c_len)
        word_lengths.append(w_len)

        if c_len < 150:
            under_150_chars += 1
        if w_len < 25:
            under_25_words += 1

        # Line structure analysis
        lines = [ln.strip() for ln in nfc_text.splitlines() if ln.strip()]
        if lines:
            short_lines = sum(1 for ln in lines if len(ln.split()) < 4)
            sl_ratio = short_lines / len(lines)
        else:
            sl_ratio = 1.0
        short_line_ratios.append(sl_ratio)
        if sl_ratio > 0.35 and len(lines) >= 3:
            high_short_lines_count += 1

        # Artifact detection
        if RE_INST.search(nfc_text):
            inst_artifact_count += 1
        if RE_HTML.search(nfc_text):
            html_artifact_count += 1

        # Odia script and orthographic profile
        ortho = analyze_odia_orthography(nfc_text)
        odia_ratios.append(ortho["odia_ratio"])
        latin_ratios.append(ortho["latin_ratio"])
        if ortho["odia_ratio"] < 0.65:
            low_odia_ratio_count += 1

        unicode_totals["zws_docs"] += int(ortho["zws"] > 0)
        unicode_totals["zwnj_docs"] += int(ortho["zwnj"] > 0)
        unicode_totals["zwj_docs"] += int(ortho["zwj"] > 0)
        unicode_totals["halant_docs"] += int(ortho["halant"] > 0)
        unicode_totals["nukta_docs"] += int(ortho["nukta"] > 0)

        # Save 3 tiny snippet previews for sanity inspection
        if len(sample_records) < 3 and c_len > 200:
            sample_records.append({
                "idx": idx,
                "char_len": c_len,
                "word_len": w_len,
                "odia_ratio": round(ortho["odia_ratio"], 4),
                "preview": nfc_text[:180].replace("\n", " "),
            })

    # Compute percentiles
    def get_percentiles(arr):
        if not arr:
            return {}
        a = np.array(arr, dtype=np.float64)
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        vals = np.percentile(a, pcts)
        res = {f"p{p}": round(float(v), 2) for p, v in zip(pcts, vals)}
        res["mean"] = round(float(np.mean(a)), 2)
        res["max"] = round(float(np.max(a)), 2)
        return res

    valid_docs = len(char_lengths)
    elapsed = round(time.time() - t0, 2)

    # Estimate surviving documents if all proposed Step 1.2 filters are applied
    report = {
        "dataset_id": DATASET_ID,
        "repo_metadata": repo_meta,
        "sample_size_streamed": idx if "idx" in locals() else 0,
        "valid_non_empty_docs": valid_docs,
        "null_or_empty_docs": null_or_empty_count,
        "columns": column_names,
        "text_column": text_col,
        "top_sources": dict(source_counter.most_common(15)),
        "char_length_stats": get_percentiles(char_lengths),
        "word_length_stats": get_percentiles(word_lengths),
        "odia_script_ratio_stats": get_percentiles(odia_ratios),
        "latin_script_ratio_stats": get_percentiles(latin_ratios),
        "hygiene_and_filter_flags": {
            "exact_duplicates_in_sample": duplicate_count,
            "exact_duplicate_pct": round(100.0 * duplicate_count / max(1, valid_docs), 2),
            "under_150_chars_count": under_150_chars,
            "under_150_chars_pct": round(100.0 * under_150_chars / max(1, valid_docs), 2),
            "under_25_words_count": under_25_words,
            "under_25_words_pct": round(100.0 * under_25_words / max(1, valid_docs), 2),
            "low_odia_ratio_under_0_65_count": low_odia_ratio_count,
            "low_odia_ratio_under_0_65_pct": round(100.0 * low_odia_ratio_count / max(1, valid_docs), 2),
            "high_short_line_boilerplate_count": high_short_lines_count,
            "high_short_line_boilerplate_pct": round(100.0 * high_short_lines_count / max(1, valid_docs), 2),
            "inst_chat_artifact_count": inst_artifact_count,
            "html_tag_artifact_count": html_artifact_count,
        },
        "orthography_and_unicode_audit": {
            "raw_not_nfc_count": nfc_diff_count,
            "nfkc_differs_from_nfc_count": nfkc_destructive_diff_count,
            "docs_with_zws_u200b": unicode_totals["zws_docs"],
            "docs_with_zwnj_u200c": unicode_totals["zwnj_docs"],
            "docs_with_zwj_u200d": unicode_totals["zwj_docs"],
            "docs_with_halant_u0b4d": unicode_totals["halant_docs"],
            "docs_with_nukta": unicode_totals["nukta_docs"],
        },
        "sample_previews": sample_records,
        "elapsed_seconds": elapsed,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== STEP 0.2 VERIFICATION SUMMARY ===")
    print(f"Dataset                 : {DATASET_ID}")
    print(f"Hub Parquet Size (GB)   : {repo_meta.get('parquet_size_gb')} GB across {repo_meta.get('parquet_files')} files (License: {repo_meta.get('license')})")
    print(f"Columns Detected        : {column_names} (Text col: '{text_col}')")
    print(f"Streamed Sample Size    : {valid_docs:,} valid docs (Null/Empty: {null_or_empty_count})")
    print(f"Top Sources (if any)    : {dict(source_counter.most_common(5))}")
    print(f"Char Length (p10/p50/p90): {report['char_length_stats'].get('p10')} / {report['char_length_stats'].get('p50')} / {report['char_length_stats'].get('p90')} (mean: {report['char_length_stats'].get('mean')})")
    print(f"Word Length (p10/p50/p90): {report['word_length_stats'].get('p10')} / {report['word_length_stats'].get('p50')} / {report['word_length_stats'].get('p90')} (mean: {report['word_length_stats'].get('mean')})")
    print(f"Odia Ratio  (p05/p50/p95): {report['odia_script_ratio_stats'].get('p5')} / {report['odia_script_ratio_stats'].get('p50')} / {report['odia_script_ratio_stats'].get('p95')}")
    print(f"Exact Duplicates in 100K: {duplicate_count:,} ({report['hygiene_and_filter_flags']['exact_duplicate_pct']}%)")
    print(f"Under 150 chars / 25 wds: {under_150_chars:,} ({report['hygiene_and_filter_flags']['under_150_chars_pct']}%) / {under_25_words:,} ({report['hygiene_and_filter_flags']['under_25_words_pct']}%)")
    print(f"Odia Ratio < 0.65       : {low_odia_ratio_count:,} ({report['hygiene_and_filter_flags']['low_odia_ratio_under_0_65_pct']}%)")
    print(f"Short-Line Boilerplate  : {high_short_lines_count:,} ({report['hygiene_and_filter_flags']['high_short_line_boilerplate_pct']}%)")
    print(f"[INST] / HTML Artifacts : {inst_artifact_count:,} / {html_artifact_count:,}")
    print(f"NFKC Alters NFC Docs    : {nfkc_destructive_diff_count:,} | ZWS(u200b): {unicode_totals['zws_docs']:,} | ZWNJ(u200c): {unicode_totals['zwnj_docs']:,} | Halant: {unicode_totals['halant_docs']:,}")
    print(f"Artifact Saved To       : {ARTIFACT_PATH} (in {elapsed}s)")
    print("=====================================")


if __name__ == "__main__":
    main()

# python3 scripts/step_02_eda_primary.py --sample-size 100000
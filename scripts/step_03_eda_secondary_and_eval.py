#!/usr/bin/env python3
"""
Step 0.3: Streaming EDA for Secondary Corpus (ai4bharat/sangraha),
English Anchor (HuggingFaceFW/fineweb-edu), and Evaluation Benchmark
(saidutta69/odia-eval-benchmark).
Automatically loads .env and normalizes Windows D:/ paths for WSL.
"""

import hashlib
import json
import os
import platform
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import get_dataset_split_names, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

ARTIFACT_PATH = Path("artifacts/step_03_eda_secondary_and_eval.json")
RE_MULTI_SPACE = re.compile(r"\s+")


def load_dotenv_wsl_safe(dotenv_path: str = ".env") -> dict:
    """Loads .env into os.environ and translates Windows drive paths (e.g. D:/) to WSL /mnt/d/."""
    loaded = {}
    p = Path(dotenv_path)
    if p.exists():
        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            # Convert Windows drive letter path (e.g., D:/huggingface_cache) when running on Linux/WSL
            if platform.system() == "Linux" and len(v) >= 3 and v[1] == ":" and v[2] in ("/", "\\"):
                drive = v[0].lower()
                rest = v[3:].replace("\\", "/")
                v = f"/mnt/{drive}/{rest}"
            os.environ[k] = v
            loaded[k] = "***REDACTED***" if "TOKEN" in k or "KEY" in k else v
    return loaded


def get_percentiles(arr: list) -> dict:
    if not arr:
        return {}
    a = np.array(arr, dtype=np.float64)
    pcts = [5, 10, 25, 50, 75, 90, 95]
    vals = np.percentile(a, pcts)
    res = {f"p{p}": round(float(v), 2) for p, v in zip(pcts, vals)}
    res["mean"] = round(float(np.mean(a)), 2)
    res["max"] = round(float(np.max(a)), 2)
    return res


def compute_odia_ratio(text: str) -> float:
    odia = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cp = ord(ch)
        if (0x0B00 <= cp <= 0x0B7F) or cp in (0x0964, 0x0965):
            odia += 1
    return (odia / total) if total > 0 else 0.0


def profile_sangraha(token: str, sample_size: int = 25_000) -> dict:
    dataset_id = "ai4bharat/sangraha"
    print(f"\n[1/3] Profiling {dataset_id} (data_dir='verified/ori', {sample_size:,} rows) ...")
    ds = load_dataset(dataset_id, data_dir="verified/ori", split="train", streaming=True, token=token)

    char_lens, word_lens, odia_ratios = [], [], []
    seen_hashes = set()
    dup_count = 0
    zwnj_docs = 0
    nfkc_diff_docs = 0
    doc_types = Counter()
    columns = None
    text_col = "text"

    for idx, row in enumerate(tqdm(ds, total=sample_size, desc="Sangraha (ori)")):
        if idx >= sample_size:
            break
        if columns is None:
            columns = list(row.keys())
            for c in ["text", "content", "document"]:
                if c in row:
                    text_col = c
                    break
        if "type" in row and row["type"]:
            doc_types[str(row["type"])] += 1

        raw = row.get(text_col, "")
        if not raw or not isinstance(raw, str) or not raw.strip():
            continue

        nfc = unicodedata.normalize("NFC", raw)
        if nfc != unicodedata.normalize("NFKC", nfc):
            nfkc_diff_docs += 1
        if "\u200c" in nfc:
            zwnj_docs += 1

        h = hashlib.sha256(RE_MULTI_SPACE.sub(" ", nfc.strip().lower()).encode("utf-8")).digest()
        if h in seen_hashes:
            dup_count += 1
        else:
            seen_hashes.add(h)

        char_lens.append(len(nfc))
        word_lens.append(len(nfc.split()))
        odia_ratios.append(compute_odia_ratio(nfc))

    return {
        "dataset_id": dataset_id,
        "data_dir": "verified/ori",
        "columns": columns,
        "text_column": text_col,
        "streamed_rows": len(char_lens),
        "doc_types": dict(doc_types),
        "char_length_stats": get_percentiles(char_lens),
        "word_length_stats": get_percentiles(word_lens),
        "odia_ratio_stats": get_percentiles(odia_ratios),
        "exact_duplicates_in_sample": dup_count,
        "zwnj_docs": zwnj_docs,
        "nfkc_diff_docs": nfkc_diff_docs,
    }


def profile_fineweb_anchor(token: str, sample_size: int = 10_000) -> dict:
    dataset_id = "HuggingFaceFW/fineweb-edu"
    subset = "sample-10BT"
    print(f"\n[2/3] Profiling {dataset_id} ({subset}, {sample_size:,} rows) ...")
    ds = load_dataset(dataset_id, name=subset, split="train", streaming=True, token=token)

    char_lens, word_lens = [], []
    columns = None

    for idx, row in enumerate(tqdm(ds, total=sample_size, desc="FineWeb-Edu")):
        if idx >= sample_size:
            break
        if columns is None:
            columns = list(row.keys())
        txt = row.get("text", "")
        if txt and isinstance(txt, str):
            char_lens.append(len(txt))
            word_lens.append(len(txt.split()))

    return {
        "dataset_id": dataset_id,
        "subset": subset,
        "columns": columns,
        "streamed_rows": len(char_lens),
        "char_length_stats": get_percentiles(char_lens),
        "word_length_stats": get_percentiles(word_lens),
    }


def profile_eval_benchmark(token: str) -> dict:
    dataset_id = "saidutta69/odia-eval-benchmark"
    print(f"\n[3/3] Profiling evaluation benchmark: {dataset_id} ...")
    api = HfApi(token=token)
    info = api.dataset_info(dataset_id, files_metadata=True)
    siblings = [f.rfilename for f in (info.siblings or [])]

    try:
        splits = get_dataset_split_names(dataset_id, token=token)
    except Exception:
        splits = ["train"]

    split_summaries = {}
    for sp in splits:
        ds = load_dataset(dataset_id, split=sp, streaming=True, token=token)
        cols = None
        row_count = 0
        task_counter = Counter()
        sample_rows = []
        char_lens = []

        for idx, row in enumerate(tqdm(ds, desc=f"Eval split '{sp}'")):
            if cols is None:
                cols = list(row.keys())
            row_count += 1
            for tcol in ["task", "category", "subset", "source", "type", "benchmark"]:
                if tcol in row and row[tcol] is not None:
                    task_counter[str(row[tcol])] += 1
                    break
            # Measure combined text length across string fields
            combined_str = " ".join(str(v) for k, v in row.items() if isinstance(v, str))
            char_lens.append(len(combined_str))
            if len(sample_rows) < 3:
                sample_rows.append({
                    k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                    for k, v in row.items()
                })
            if idx >= 150_000:  # Safety cap
                break

        split_summaries[sp] = {
            "total_rows_counted": row_count,
            "columns": cols,
            "task_distribution": dict(task_counter.most_common(20)),
            "combined_char_stats": get_percentiles(char_lens),
            "sample_rows": sample_rows,
        }

    return {
        "dataset_id": dataset_id,
        "repo_files": siblings,
        "splits": splits,
        "split_details": split_summaries,
    }


def main():
    t0 = time.time()
    env_loaded = load_dotenv_wsl_safe(".env")
    token = os.environ.get("HF_TOKEN")
    print(f"Loaded .env settings: {env_loaded} | HF Auth Active: {bool(token)}")

    sangraha_rep = profile_sangraha(token=token, sample_size=25_000)
    fineweb_rep = profile_fineweb_anchor(token=token, sample_size=10_000)
    eval_rep = profile_eval_benchmark(token=token)

    elapsed = round(time.time() - t0, 2)
    full_report = {
        "env_settings": env_loaded,
        "sangraha_ori": sangraha_rep,
        "fineweb_edu_anchor": fineweb_rep,
        "odia_eval_benchmark": eval_rep,
        "elapsed_seconds": elapsed,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print("\n=== STEP 0.3 VERIFICATION SUMMARY ===")
    print(f"1. Sangraha (verified/ori) : {sangraha_rep['streamed_rows']:,} rows | Cols: {sangraha_rep['columns']}")
    print(f"   Words (p10 / p50 / p90) : {sangraha_rep['word_length_stats'].get('p10')} / {sangraha_rep['word_length_stats'].get('p50')} / {sangraha_rep['word_length_stats'].get('p90')} (mean: {sangraha_rep['word_length_stats'].get('mean')})")
    print(f"   Odia Ratio (p05 / p50)  : {sangraha_rep['odia_ratio_stats'].get('p5')} / {sangraha_rep['odia_ratio_stats'].get('p50')} | ZWNJ docs: {sangraha_rep['zwnj_docs']:,} | NFKC diffs: {sangraha_rep['nfkc_diff_docs']:,}")
    print(f"2. FineWeb-Edu (10BT)      : {fineweb_rep['streamed_rows']:,} rows | Words (p10/p50/p90): {fineweb_rep['word_length_stats'].get('p10')} / {fineweb_rep['word_length_stats'].get('p50')} / {fineweb_rep['word_length_stats'].get('p90')} (mean: {fineweb_rep['word_length_stats'].get('mean')})")
    print(f"3. Odia Eval Benchmark     : Splits={eval_rep['splits']} | Files={eval_rep['repo_files']}")
    for sp, d in eval_rep["split_details"].items():
        print(f"   Split '{sp}'            : {d['total_rows_counted']:,} rows | Cols: {d['columns']} | Tasks: {d['task_distribution']}")
    print(f"Artifact Saved To          : {ARTIFACT_PATH} (in {elapsed}s)")
    print("=====================================")


if __name__ == "__main__":
    main()

# python3 scripts/step_03_eda_secondary_and_eval.py
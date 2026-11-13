#!/usr/bin/env python3
"""
Phase 1 (Step 11.0): Upstream Odia SFT & Preference Dataset Discovery & Schema Audit.
Runs 100% on local CPU. Self-contained with zero sys.path issues.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["HF_HUB_DISABLE_XET"] = "1"


def load_env_file(env_path: str = ".env") -> None:
    p = Path(env_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip().replace("\r", "")
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k.strip() and v:
            os.environ[k.strip()] = v


load_env_file(".env")
token = os.environ.get("HF_TOKEN")

from huggingface_hub import HfApi
from datasets import load_dataset, get_dataset_config_names

api = HfApi(token=token)


def main():
    print("\n=== STEP 11.0: ODIA SFT DATASET DISCOVERY & SCHEMA AUDIT ===")

    # 1. Search Hugging Face Hub for top public Odia instruction / SFT datasets
    print("\n[1/3] Searching HF Hub for Odia Instruction / Chat / QA datasets...")
    queries = [
        "odia instruct",
        "odia qa",
        "indic instruct",
        "odia alpaca",
        "odia sft",
        "OdiaGenAI",
    ]
    seen = set()
    for q in queries:
        try:
            for ds in api.list_datasets(search=q, limit=10):
                if ds.id not in seen:
                    seen.add(ds.id)
                    print(f"  - Found HF Dataset: {ds.id} (downloads: {getattr(ds, 'downloads', 0)})")
        except Exception as e:
            print(f"  - Search '{q}' warning: {e}")

    # 2. Probe primary candidate instruction / QA datasets in streaming mode
    candidates = [
        ("ai4bharat/indic-instruct-data-v0.1", None),
        ("OdiaGenAI/Alpaca_Odia_52k", None),
        ("OdiaGenAI/odia_domain_context_qa_v1", None),
        ("CohereForAI/aya_dataset", None),
        ("saidutta69/odia-eval-benchmark", None),
    ]

    # Also dynamically add any OdiaGenAI or instruct datasets found in step 1
    for ds_id in list(seen)[:12]:
        if any(w in ds_id.lower() for w in ["instruct", "alpaca", "qa", "sft", "chat"]):
            if (ds_id, None) not in candidates:
                candidates.append((ds_id, None))

    print("\n[2/3] Probing candidate schemas & sampling first row...")
    for repo_id, cfg in candidates:
        try:
            configs = get_dataset_config_names(repo_id, token=token)
            print(f"\n  -> {repo_id} | Configs ({len(configs)}): {configs[:8]}")
            target_cfg = cfg if cfg in configs else (configs[0] if configs else None)
            ds = load_dataset(repo_id, target_cfg, streaming=True, token=token)
            splits = list(ds.keys())
            first_split = splits[0]
            sample = next(iter(ds[first_split]))
            print(f"     Active Config : {target_cfg} | Splits: {splits}")
            print(f"     Columns       : {list(sample.keys())}")
            preview = str(sample)[:240].replace("\n", " ")
            print(f"     Sample Preview: {preview}...")
        except Exception as e:
            print(f"  -> {repo_id}: Skipped ({type(e).__name__}: {str(e)[:100]})")

    # 3. Check local clean files & tokenizer readiness
    print("\n[3/3] Checking local clean corpora, tokenizer & base weights...")
    for p in [
        Path("artifacts/tokenizer/tokenizer.model"),
        Path("data/clean/val_docs.jsonl"),
        Path("data/clean/train_docs.jsonl"),
        Path("artifacts/hf_export/odia-slm-100m-base/model.safetensors"),
    ]:
        status = f"READY ({round(p.stat().st_size / (1024*1024), 2)} MB)" if p.exists() else "MISSING"
        print(f"  - {p}: {status}")

    print("\n=== STEP 11.0 DISCOVERY COMPLETE ===")


if __name__ == "__main__":
    main()

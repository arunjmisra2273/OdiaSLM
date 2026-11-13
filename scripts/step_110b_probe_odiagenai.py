#!/usr/bin/env python3
"""
Step 11.0b: Schema & Quality Probe for the 7 High-Value Odia Instruction/QA Datasets.
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

from datasets import load_dataset

TARGETS = [
    "OdiaGenAI/dolly-odia-15k",
    "OdiaGenAI/odia_context_qa_98k",
    "OdiaGenAI/OdiEnCorp_translation_instructions_25k",
    "OdiaGenAI/gpt-teacher-roleplay-odia-3k",
    "OdiaGenAI/odia_domain_knowledge_503",
    "OdiaGenAI/hardcode_odia_qa_105",
    "abhilash88/odia-instruction-dataset",
    "saillab/alpaca-odia_oriya-cleaned",
]


def main():
    print("\n=== STEP 11.0b: PROBING HIGH-VALUE ODIA INSTRUCTION DATASETS ===")
    for repo_id in TARGETS:
        try:
            ds = load_dataset(repo_id, streaming=True, token=token)
            splits = list(ds.keys())
            first_split = splits[0]
            it = iter(ds[first_split])
            row = next(it)
            print(f"\n  [OK] {repo_id} | Splits: {splits}")
            print(f"       Columns: {list(row.keys())}")
            for k, v in row.items():
                val_str = str(v).replace("\n", " ")[:140]
                print(f"         - {k}: {val_str}")
        except Exception as e:
            print(f"\n  [SKIP] {repo_id}: {type(e).__name__}: {str(e)[:120]}")

    print("\n=== STEP 11.0b PROBE COMPLETE ===")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

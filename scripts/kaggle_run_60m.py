#!/usr/bin/env python3
"""
KAGGLE 2xT4 PRODUCTION RUNNER — Odia SLM 60M (62,111,360 Params)
Paste this entire script into a SINGLE code cell in a Kaggle Notebook with:
  - Accelerator : GPU T4 x2
  - Internet    : ON
  - Kaggle Secret: Add `HF_TOKEN` under Add-ons -> Secrets (and toggle it ON)
Then click "Save Version" -> "Save & Run All (Commit)" so it runs in the background
even with your browser and laptop completely closed!
"""

import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

# 1. Load HF_TOKEN from Kaggle Secrets (or environment)
if not os.environ.get("HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("[Kaggle Init] Loaded HF_TOKEN from Kaggle Secrets.")
    except Exception as e:
        raise RuntimeError(
            "HF_TOKEN not found! In Kaggle Notebook menu, click 'Add-ons' -> 'Secrets', "
            "add a secret named HF_TOKEN with your Hugging Face write token, and toggle it ON."
        ) from e

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DATASET_REPO = "arunjmisra2273/odia-slm-pretrain-bin"
CKPT_REPO = "arunjmisra2273/odia-slm-60m-ckpt"
WORK_DIR = Path("/kaggle/working/odia-llm")
BIN_DIR = WORK_DIR / "data/bin"
OUT_DIR = WORK_DIR / "checkpoints/60m"

WORK_DIR.mkdir(parents=True, exist_ok=True)
BIN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[1/3] Downloading pre-tokenized binary bundle & src/ from {DATASET_REPO} ...")
snapshot_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    local_dir=str(WORK_DIR),
    token=os.environ["HF_TOKEN"],
)

# 2. Check if a previous session already uploaded ckpt_latest.pt to CKPT_REPO
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    repo_files = api.list_repo_files(repo_id=CKPT_REPO, repo_type="model")
    if "ckpt_latest.pt" in repo_files:
        print(f"[2/3] Found existing ckpt_latest.pt in {CKPT_REPO}! Downloading to resume session ...")
        for fname in ["ckpt_latest.pt", "training_log.jsonl", "val_history.jsonl"]:
            if fname in repo_files:
                hf_hub_download(
                    repo_id=CKPT_REPO,
                    filename=fname,
                    repo_type="model",
                    local_dir=str(OUT_DIR),
                    token=os.environ["HF_TOKEN"],
                )
    else:
        print(f"[2/3] No existing checkpoint in {CKPT_REPO}. Starting fresh from Step 0!")
except Exception as e:
    print(f"[2/3] Starting fresh session ({e})")

# 3. Launch 2xT4 DistributedDataParallel (DDP) Training via torchrun
cmd = [
    "torchrun",
    "--standalone",
    "--nproc_per_node=2",
    str(WORK_DIR / "src/train_gpu.py"),
    "--variant", "60m",
    "--bin-dir", str(BIN_DIR),
    "--out-dir", str(OUT_DIR),
    "--hf-repo-id", CKPT_REPO,
    "--max-steps", "10000",
    "--warmup-steps", "400",
    "--context-switch-step", "8000",
    "--decay-start-step", "8500",
    "--stage1-seq-len", "512",
    "--stage2-seq-len", "1024",
    "--stage1-micro-bsz", "32",
    "--stage2-micro-bsz", "16",
    "--grad-accum", "4",
    "--max-lr", "6e-4",
    "--min-lr", "6e-5",
    "--dtype", "float16",
    "--max-session-seconds", "40500",
]

print(f"[3/3] Launching 2xT4 DDP Training: {' '.join(cmd)}")
env = os.environ.copy()
env["PYTHONPATH"] = str(WORK_DIR)
proc = subprocess.run(cmd, cwd=str(WORK_DIR), env=env)
if proc.returncode != 0:
    sys.exit(proc.returncode)
print("=== KAGGLE SESSION COMPLETED CLEANLY ===")

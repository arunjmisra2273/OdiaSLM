#!/usr/bin/env python3
"""
Step 6.1 (WSL-Hardened): Cloud Dataset & Code Bundle Uploader + Kaggle 2xT4 Runner Generator.
Disables buggy WSL hf-xet staging (uses standard HTTP LFS multipart upload) and
skips any files already committed to the remote repository.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# CRITICAL FOR WSL2: Disable hf-xet Rust staging bug & use native Linux /tmp for upload cache
# MUST be set BEFORE importing huggingface_hub!
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
Path("/tmp/hf_upload_cache").mkdir(parents=True, exist_ok=True)
os.environ["HF_XET_CACHE"] = "/tmp/hf_upload_cache"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.step_03_eda_secondary_and_eval import load_dotenv_wsl_safe

# Load .env before importing HfApi
load_dotenv_wsl_safe(".env")

from huggingface_hub import HfApi

BIN_DIR = Path("/mnt/d/odia_llm_data/bin")
SRC_DIR = Path("src")
ARTIFACT_PATH = Path("artifacts/step_61_cloud_prep_report.json")
KAGGLE_SCRIPT_PATH = Path("scripts/kaggle_run_60m.py")


def generate_kaggle_runner_script(
    hf_user: str,
    dataset_repo: str,
    ckpt_repo_60m: str,
) -> str:
    """Generates a self-contained single-cell Python script to paste into a Kaggle 2xT4 Notebook."""
    return f'''#!/usr/bin/env python3
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

DATASET_REPO = "{dataset_repo}"
CKPT_REPO = "{ckpt_repo_60m}"
WORK_DIR = Path("/kaggle/working/odia-llm")
BIN_DIR = WORK_DIR / "data/bin"
OUT_DIR = WORK_DIR / "checkpoints/60m"

WORK_DIR.mkdir(parents=True, exist_ok=True)
BIN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[1/3] Downloading pre-tokenized binary bundle & src/ from {{DATASET_REPO}} ...")
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
        print(f"[2/3] Found existing ckpt_latest.pt in {{CKPT_REPO}}! Downloading to resume session ...")
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
        print(f"[2/3] No existing checkpoint in {{CKPT_REPO}}. Starting fresh from Step 0!")
except Exception as e:
    print(f"[2/3] Starting fresh session ({{e}})")

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

print(f"[3/3] Launching 2xT4 DDP Training: {{' '.join(cmd)}}")
env = os.environ.copy()
env["PYTHONPATH"] = str(WORK_DIR)
proc = subprocess.run(cmd, cwd=str(WORK_DIR), env=env)
if proc.returncode != 0:
    sys.exit(proc.returncode)
print("=== KAGGLE SESSION COMPLETED CLEANLY ===")
'''


def upload_with_retry(api: HfApi, local_path: Path, path_in_repo: str, repo_id: str, repo_type: str):
    """Uploads a file with 3 automatic retries in case of transient network drops."""
    for attempt in range(1, 4):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            return
        except Exception as e:
            if attempt == 3:
                raise
            print(f"      [Retry {attempt}/3] Upload error on {path_in_repo}: {e}. Retrying in 5s ...")
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Step 6.1 Cloud Upload & Kaggle Runner Prep")
    parser.add_argument("--skip-train-bin", action="store_true", help="Skip train.bin upload if testing")
    parser.add_argument("--force-reupload-src", action="store_true", help="Re-upload src/ files even if present")
    args = parser.parse_args()

    t0 = time.time()
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN is missing from .env!"

    api = HfApi(token=token)
    user_info = api.whoami()
    hf_user = user_info["name"]
    print(f"[1/4] Authenticated with Hugging Face Hub as: '{hf_user}' (XET Disabled={os.environ.get('HF_HUB_DISABLE_XET')})")

    dataset_repo = f"{hf_user}/odia-slm-pretrain-bin"
    ckpt_repo_60m = f"{hf_user}/odia-slm-60m-ckpt"
    ckpt_repo_100m = f"{hf_user}/odia-slm-100m-ckpt"

    print(f"[2/4] Verifying private Hugging Face repos:")
    print(f"      - Dataset Bundle Repo : {dataset_repo} (private)")
    print(f"      - 60M Checkpoint Repo : {ckpt_repo_60m} (private)")
    print(f"      - 100M Checkpoint Repo: {ckpt_repo_100m} (private)")

    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(repo_id=ckpt_repo_60m, repo_type="model", private=True, exist_ok=True)
    api.create_repo(repo_id=ckpt_repo_100m, repo_type="model", private=True, exist_ok=True)

    existing_remote_files = set(api.list_repo_files(repo_id=dataset_repo, repo_type="dataset"))
    print(f"      Already present in {dataset_repo}: {sorted(existing_remote_files)}")

    # Ensure src/__init__.py exists
    init_py = SRC_DIR / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")

    # Upload src/ files
    print(f"\n[3/4] Uploading verified source code (src/) and binary bundle to {dataset_repo} ...")
    uploaded_files = []
    for src_file in ["__init__.py", "hygiene.py", "model.py", "dataset.py", "train_gpu.py"]:
        repo_path = f"src/{src_file}"
        local_f = SRC_DIR / src_file
        assert local_f.exists(), f"Missing {local_f}"
        if repo_path in existing_remote_files and not args.force_reupload_src:
            print(f"      [Skip] {repo_path} already exists on Hub.")
            uploaded_files.append(repo_path)
            continue
        print(f"      Uploading {repo_path} ...")
        upload_with_retry(api, local_f, repo_path, dataset_repo, "dataset")
        uploaded_files.append(repo_path)

    # Upload data/bin files
    bin_files = ["meta.json", "tokenizer.model", "val.bin"]
    if not args.skip_train_bin:
        bin_files.append("train.bin")

    for bname in bin_files:
        repo_path = f"data/bin/{bname}"
        bf = BIN_DIR / bname
        assert bf.exists(), f"Missing {bf}"
        if repo_path in existing_remote_files:
            print(f"      [Skip] {repo_path} already exists on Hub.")
            uploaded_files.append(repo_path)
            continue
        size_mb = round(bf.stat().st_size / (1024 * 1024), 2)
        print(f"      Uploading {repo_path} ({size_mb:,.2f} MB) -> {dataset_repo} ...")
        upload_with_retry(api, bf, repo_path, dataset_repo, "dataset")
        uploaded_files.append(repo_path)

    # Generate Kaggle runner script
    print(f"\n[4/4] Generating ready-to-paste Kaggle 2xT4 runner at {KAGGLE_SCRIPT_PATH} ...")
    runner_code = generate_kaggle_runner_script(
        hf_user=hf_user,
        dataset_repo=dataset_repo,
        ckpt_repo_60m=ckpt_repo_60m,
    )
    KAGGLE_SCRIPT_PATH.write_text(runner_code, encoding="utf-8")

    elapsed = round(time.time() - t0, 2)
    report = {
        "hf_username": hf_user,
        "dataset_repo": dataset_repo,
        "ckpt_repo_60m": ckpt_repo_60m,
        "ckpt_repo_100m": ckpt_repo_100m,
        "uploaded_files": uploaded_files,
        "kaggle_runner_script": str(KAGGLE_SCRIPT_PATH),
        "elapsed_seconds": elapsed,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== STEP 6.1 VERIFICATION SUMMARY ===")
    print(f"HF Authenticated User   : {hf_user}")
    print(f"Private Dataset Repo    : https://huggingface.co/datasets/{dataset_repo}")
    print(f"Private 60M Ckpt Repo   : https://huggingface.co/{ckpt_repo_60m}")
    print(f"Private 100M Ckpt Repo  : https://huggingface.co/{ckpt_repo_100m}")
    print(f"Files Verified on Hub   : {uploaded_files}")
    print(f"Kaggle 2xT4 Runner Saved: {KAGGLE_SCRIPT_PATH}")
    print(f"Total Runtime           : {elapsed}s ({round(elapsed/60, 1)} mins)")
    print("=====================================")


if __name__ == "__main__":
    main()

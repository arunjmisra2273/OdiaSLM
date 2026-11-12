#!/usr/bin/env python3
"""
Local Remote Monitor: Pulls training_log.jsonl & val_history.jsonl from your private
Hugging Face checkpoint repo (60m or 100m) anytime to display live training progress,
validation BPB/PPL, Top-1/Top-5 accuracy, and remaining ETA.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi, hf_hub_download
from scripts.step_03_eda_secondary_and_eval import load_dotenv_wsl_safe


def main():
    parser = argparse.ArgumentParser(description="Monitor remote Kaggle training progress via HF Hub")
    parser.add_argument("--repo-id", type=str, default="arunjmisra2273/odia-slm-100m-ckpt")
    parser.add_argument("--local-dir", type=str, default="checkpoints/100m_remote_sync")
    args = parser.parse_args()

    load_dotenv_wsl_safe(".env")
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    out_dir = Path(args.local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
    if "training_log.jsonl" not in files:
        print(f"[Monitor] No training_log.jsonl uploaded yet in {args.repo_id} (first upload happens at step 500).")
        print(f"          Current files in repo: {files}")
        return

    for fname in ["training_log.jsonl", "val_history.jsonl"]:
        if fname in files:
            hf_hub_download(
                repo_id=args.repo_id,
                filename=fname,
                repo_type="model",
                local_dir=str(out_dir),
                token=token,
                force_download=True,
            )

    log_lines = [
        json.loads(line)
        for line in (out_dir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_steps = [r for r in log_lines if r.get("type") == "train_step"]
    val_steps = [r for r in log_lines if r.get("type") == "validation"]

    latest = train_steps[-1]
    step = latest["step"]
    max_steps = 12000 if "100m" in args.repo_id else 10000
    max_tokens = max_steps * 131072
    pct = round(100.0 * (step + 1) / max_steps, 1)

    tok_s = latest.get("tok_per_sec", 26000)
    rem_tokens = max(0, (max_steps - 1 - step) * 131072)
    eta_hours = round(rem_tokens / max(1.0, tok_s) / 3600.0, 2)

    print("\n=== LIVE REMOTE TRAINING TELEMETRY ===")
    print(f"Repo                    : {args.repo_id}")
    print(f"Latest Logged Step      : {step:,} / {max_steps:,} ({pct}%) | Seq Len: {latest['seq_len']}")
    print(f"Tokens Seen             : {latest['tokens_seen']:,} / {max_tokens:,}")
    print(f"Train CE / PPL / BPB    : {latest['train_loss_ce']} / {latest['train_ppl']} / {latest['train_bpb']}")
    print(f"Train Top-1 / Top-5 Acc : {latest.get('top1_acc')} / {latest.get('top5_acc')}")
    print(f"Speed & Hardware        : {tok_s:,.0f} tok/s | VRAM: {latest.get('vram_allocated_gb')} GB | Est MFU: {latest.get('est_mfu_pct')}%")
    if val_steps:
        v = val_steps[-1]
        print(f"Latest Val (Step {v['step']:05d}) : CE={v['val_loss_ce']} | PPL={v['val_perplexity']} | BPB={v['val_bits_per_byte']} | Top1={v['val_top1_acc']} | Top5={v['val_top5_acc']}")
    print(f"Estimated Remaining GPU : ~{eta_hours} hours")
    print("======================================")


if __name__ == "__main__":
    main()

# python3 scripts/monitor_remote_training.py
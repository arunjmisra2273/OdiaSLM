#!/usr/bin/env python3
"""
Master End-to-End Post-Training Pipeline for OdiaSLM-100M & Joint 60M-vs-100M Paper Artifacts.
Runs all 5 stages sequentially:
  1. Pull latest logs & checkpoint from arunjmisra2273/odia-slm-100m-ckpt
  2. Plot 6 training telemetry figures for 100M -> artifacts/figures/100m_training/
  3. Export 100M checkpoint to HF LlamaForCausalLM -> artifacts/hf_export/odia-slm-100m-base/
  4. Run 4-metric 27-task + 6-domain evaluation -> artifacts/evaluation_results_100m.json & figures
  5. Generate 60M vs 100M scaling figures, combined LaTeX tables, and publish arunjmisra2273/OdiaSLM-100M-Base.
"""

import argparse
import subprocess
import sys


def run_step(title: str, cmd: list):
    print(f"\n======================================================================")
    print(f"[MASTER PIPELINE] {title}")
    print(f"Command: {' '.join(cmd)}")
    print(f"======================================================================")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Finalize OdiaSLM-100M evaluation, HF release & joint paper charts")
    parser.add_argument("--max-samples-per-task", type=int, default=80)
    parser.add_argument("--max-docs-per-domain", type=int, default=80)
    parser.add_argument("--skip-upload", action="store_true", help="Prepare local release without pushing to HF")
    args = parser.parse_args()

    py = sys.executable

    # Stage 1: Sync remote telemetry logs
    run_step(
        "Stage 1/5: Syncing training_log.jsonl & val_history.jsonl from arunjmisra2273/odia-slm-100m-ckpt",
        [
            py, "scripts/monitor_remote_training.py",
            "--repo-id", "arunjmisra2273/odia-slm-100m-ckpt",
            "--local-dir", "checkpoints/100m_remote_sync",
        ],
    )

    # Stage 2: Plot 100M training telemetry charts
    run_step(
        "Stage 2/5: Generating 6 Training Telemetry Figures for OdiaSLM-100M",
        [
            py, "scripts/plot_training_telemetry.py",
            "--log-dir", "checkpoints/100m_remote_sync",
            "--out-dir", "artifacts/figures/100m_training",
            "--label", "OdiaSLM-100M (100.7M)",
        ],
    )

    # Stage 3: Download ckpt_latest.pt and export to HF LlamaForCausalLM (safetensors)
    run_step(
        "Stage 3/5: Downloading 100M Checkpoint & Exporting to Native HF LlamaForCausalLM",
        [
            py, "scripts/step_81_export_hf_and_api.py",
            "--variant", "100m",
            "--download-from-repo", "arunjmisra2273/odia-slm-100m-ckpt",
            "--ckpt-path", "checkpoints/100m_remote_sync/ckpt_latest.pt",
            "--export-dir", "artifacts/hf_export/odia-slm-100m-base",
        ],
    )

    # Stage 4: Run 27-task 4-metric evaluation & plot 100M evaluation figures
    run_step(
        "Stage 4/5: Running 27-Benchmark 4-Metric Evaluation & Generating 100M Figures",
        [
            py, "scripts/step_82_evaluate_benchmark.py",
            "--variant", "100m",
            "--ckpt-path", "checkpoints/100m_remote_sync/ckpt_latest.pt",
            "--max-samples-per-task", str(args.max_samples_per_task),
            "--max-docs-per-domain", str(args.max_docs_per_domain),
            "--out-json", "artifacts/evaluation_results_100m.json",
        ],
    )
    run_step(
        "Stage 4b/5: Generating 100M Evaluation Figures & LaTeX Tables",
        [
            py, "scripts/plot_evaluation_charts.py",
            "--eval-json", "artifacts/evaluation_results_100m.json",
            "--out-dir", "artifacts/figures/100m_evaluation",
            "--latex-out", "artifacts/paper_tables_100m.tex",
            "--label", "OdiaSLM-100M",
        ],
    )

    # Stage 5: Head-to-Head 60M vs 100M Scaling Figures + Publish 100M Release to HF
    run_step(
        "Stage 5a/5: Generating Head-to-Head 60M vs 100M Scaling Charts & Joint Paper Tables",
        [py, "scripts/plot_head_to_head_scaling.py"],
    )

    pub_cmd = [
        py, "scripts/step_91_publish_hf_release.py",
        "--variant", "100m",
        "--export-dir", "artifacts/hf_export/odia-slm-100m-base",
        "--eval-json", "artifacts/evaluation_results_100m.json",
        "--train-figs-dir", "artifacts/figures/100m_training",
        "--eval-figs-dir", "artifacts/figures/100m_evaluation",
        "--repo-id", "arunjmisra2273/OdiaSLM-100M-Base",
        "--private",
    ]
    if not args.skip_upload:
        pub_cmd.append("--upload")

    run_step("Stage 5b/5: Packaging & Publishing arunjmisra2273/OdiaSLM-100M-Base", pub_cmd)
    print("\n=== ALL 100M & JOINT PAPER ARTIFACTS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()

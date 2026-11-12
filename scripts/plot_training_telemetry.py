#!/usr/bin/env python3
"""
Step 6.3: Publication-Grade Training Telemetry Chart Generator.
Reads training_log.jsonl & val_history.jsonl and outputs 6 high-resolution
(300 DPI PNG + Vector PDF) academic figures into artifacts/figures/.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_logs(log_dir: Path):
    train_log_path = log_dir / "training_log.jsonl"
    assert train_log_path.exists(), f"Missing {train_log_path}. Run monitor_remote_training.py first!"

    train_recs = []
    val_recs = []
    seen_train_steps = set()
    seen_val_steps = set()

    for line in train_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        s = rec.get("step", 0)
        if rec.get("type") == "train_step" and s not in seen_train_steps:
            seen_train_steps.add(s)
            train_recs.append(rec)
        elif rec.get("type") == "validation" and s not in seen_val_steps:
            seen_val_steps.add(s)
            val_recs.append(rec)

    train_recs.sort(key=lambda r: r["step"])
    val_recs.sort(key=lambda r: r["step"])
    return train_recs, val_recs


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def generate_all_training_charts(log_dir: Path, out_dir: Path, model_label: str = "OdiaSLM-60M") -> list:
    train_recs, val_recs = load_logs(log_dir)
    assert len(train_recs) > 0, "No train_step records found in training_log.jsonl"

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    })

    steps = np.array([r["step"] for r in train_recs])
    tokens_m = np.array([r["tokens_seen"] / 1e6 for r in train_recs])
    train_ce = np.array([r["train_loss_ce"] for r in train_recs])
    train_bpb = np.array([r["train_bpb"] for r in train_recs])
    train_top1 = np.array([r.get("top1_acc", np.nan) for r in train_recs])
    lrs = np.array([r["lr"] for r in train_recs])
    grad_norms = np.array([r["grad_norm"] for r in train_recs])
    tok_s = np.array([r.get("tok_per_sec", 0.0) for r in train_recs])
    vram_gb = np.array([r.get("vram_allocated_gb", 0.0) for r in train_recs])

    val_steps = np.array([r["step"] for r in val_recs]) if val_recs else np.array([])
    val_tokens_m = np.array([r["tokens_seen"] / 1e6 for r in val_recs]) if val_recs else np.array([])
    val_ce = np.array([r["val_loss_ce"] for r in val_recs]) if val_recs else np.array([])
    val_bpb = np.array([r["val_bits_per_byte"] for r in val_recs]) if val_recs else np.array([])
    val_ppl = np.array([r["val_perplexity"] for r in val_recs]) if val_recs else np.array([])
    val_top1 = np.array([r.get("val_top1_acc", np.nan) for r in val_recs]) if val_recs else np.array([])
    val_top5 = np.array([r.get("val_top5_acc", np.nan) for r in val_recs]) if val_recs else np.array([])

    saved_files = []

    # -------------------------------------------------------------------------
    # Figure 1: Cross-Entropy Loss & Tokenizer-Agnostic Bits-Per-Byte (BPB)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(tokens_m, train_ce, label="Train CE Loss", color="#1f77b4", alpha=0.65, linewidth=1.2)
    if len(val_recs) > 0:
        ax1.plot(val_tokens_m, val_ce, "o-", label="Validation CE Loss", color="#d62728", linewidth=2.0, markersize=4)
    ax1.set_title(f"{model_label}: Cross-Entropy Loss vs. Tokens")
    ax1.set_xlabel("Tokens Processed (Millions)")
    ax1.set_ylabel("Cross-Entropy Loss (nats/token)")
    ax1.legend()

    ax2.plot(tokens_m, train_bpb, label="Train BPB", color="#2ca02c", alpha=0.65, linewidth=1.2)
    if len(val_recs) > 0:
        ax2.plot(val_tokens_m, val_bpb, "s-", label="Validation BPB", color="#ff7f0e", linewidth=2.0, markersize=4)
    ax2.set_title(f"{model_label}: Bits-Per-Byte (BPB) Compression")
    ax2.set_xlabel("Tokens Processed (Millions)")
    ax2.set_ylabel("Bits-Per-Byte (BPB)")
    ax2.legend()
    fig.tight_layout()
    saved_files.append(save_fig(fig, out_dir, "fig1_loss_and_bpb_curve"))

    # -------------------------------------------------------------------------
    # Figure 2: Validation Perplexity & Next-Token Accuracy (Top-1 & Top-5)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    if len(val_recs) > 0:
        ax1.plot(val_tokens_m, np.minimum(val_ppl, 500.0), "o-", color="#9467bd", linewidth=2.0, markersize=4)
        ax1.set_yscale("log")
    ax1.set_title(f"{model_label}: Validation Perplexity (Log Scale)")
    ax1.set_xlabel("Tokens Processed (Millions)")
    ax1.set_ylabel("Validation Perplexity (PPL)")

    ax2.plot(tokens_m, train_top1 * 100.0, label="Train Top-1 Acc (%)", color="#17becf", alpha=0.55, linewidth=1.2)
    if len(val_recs) > 0:
        ax2.plot(val_tokens_m, val_top1 * 100.0, "o-", label="Val Top-1 Acc (%)", color="#1f77b4", linewidth=2.0, markersize=4)
        ax2.plot(val_tokens_m, val_top5 * 100.0, "^-", label="Val Top-5 Acc (%)", color="#2ca02c", linewidth=2.0, markersize=4)
    ax2.set_title(f"{model_label}: Next-Token Prediction Accuracy")
    ax2.set_xlabel("Tokens Processed (Millions)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    fig.tight_layout()
    saved_files.append(save_fig(fig, out_dir, "fig2_perplexity_and_accuracy"))

    # -------------------------------------------------------------------------
    # Figure 3: WSD Learning Rate Schedule & Global Gradient Norm Stability
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(steps, lrs * 1e4, color="#e377c2", linewidth=2.0)
    ax1.set_title(f"{model_label}: Warmup-Stable-Decay (WSD) Schedule")
    ax1.set_xlabel("Optimizer Step")
    ax1.set_ylabel("Learning Rate ($\\times 10^{-4}$)")

    ax2.plot(steps, np.minimum(grad_norms, 5.0), color="#8c564b", alpha=0.8, linewidth=1.2)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="Clip Threshold (1.0)")
    ax2.set_title(f"{model_label}: Global L2 Gradient Norm")
    ax2.set_xlabel("Optimizer Step")
    ax2.set_ylabel("L2 Gradient Norm (capped at 5.0 for display)")
    ax2.legend()
    fig.tight_layout()
    saved_files.append(save_fig(fig, out_dir, "fig3_wsd_lr_and_grad_norm"))

    # -------------------------------------------------------------------------
    # Figure 4: Layer-Wise Gradient Norm Heatmap Across Training
    # -------------------------------------------------------------------------
    layer_recs = [r for r in train_recs if "layer_grad_norms" in r and r["layer_grad_norms"]]
    if len(layer_recs) >= 2:
        layer_keys = list(layer_recs[0]["layer_grad_norms"].keys())
        mat = np.array([[r["layer_grad_norms"].get(k, 0.0) for r in layer_recs] for k in layer_keys])
        l_steps = [r["step"] for r in layer_recs]

        fig, ax = plt.subplots(figsize=(12, 5.2))
        im = ax.imshow(
            np.log10(np.maximum(mat, 1e-5)),
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_yticks(range(len(layer_keys)))
        ax.set_yticklabels(layer_keys)
        xtick_idx = np.linspace(0, len(l_steps) - 1, min(10, len(l_steps)), dtype=int)
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels([str(l_steps[i]) for i in xtick_idx])
        ax.set_title(f"{model_label}: Layer-Wise Gradient L2 Norm Heatmap ($\\log_{{10}}$ scale)")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Transformer Parameter Group")
        fig.colorbar(im, ax=ax, label="$\\log_{10}(\\|\\nabla W\\|_2)$")
        fig.tight_layout()
        saved_files.append(save_fig(fig, out_dir, "fig4_layer_grad_heatmap"))

    # -------------------------------------------------------------------------
    # Figure 5: Parameter Weight RMS Evolution & Z-Loss Logit Stability
    # -------------------------------------------------------------------------
    if len(layer_recs) >= 2:
        w_steps = [r["step"] for r in layer_recs]
        emb_rms = [r["weight_rms_summary"]["embed"] for r in layer_recs]
        attn_rms = [r["weight_rms_summary"]["attn_q_mean_rms"] for r in layer_recs]
        mlp_rms = [r["weight_rms_summary"]["mlp_up_mean_rms"] for r in layer_recs]
        gamma_mean = [r["weight_rms_summary"]["rmsnorm_gamma_mean"] for r in layer_recs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        ax1.plot(w_steps, emb_rms, "o-", label="Tied Embedding RMS", markersize=3)
        ax1.plot(w_steps, attn_rms, "s-", label="Attention Q-Proj RMS", markersize=3)
        ax1.plot(w_steps, mlp_rms, "^-", label="SwiGLU Up-Proj RMS", markersize=3)
        ax1.set_title(f"{model_label}: Parameter Weight RMS Evolution")
        ax1.set_xlabel("Optimizer Step")
        ax1.set_ylabel("Root-Mean-Square Weight Magnitude")
        ax1.legend()

        logit_max = [r.get("logit_max", np.nan) for r in train_recs]
        logit_std = [r.get("logit_std", np.nan) for r in train_recs]
        ax2.plot(steps, logit_max, label="Max Output Logit", color="#d62728", alpha=0.75)
        ax2.plot(steps, logit_std, label="Output Logit Std Dev", color="#1f77b4", linewidth=1.8)
        ax2.set_title(f"{model_label}: Z-Loss Output Logit Stabilization")
        ax2.set_xlabel("Optimizer Step")
        ax2.set_ylabel("Logit Magnitude")
        ax2.legend()
        fig.tight_layout()
        saved_files.append(save_fig(fig, out_dir, "fig5_weight_rms_and_logit_stability"))

    # -------------------------------------------------------------------------
    # Figure 6: Hardware Systems Telemetry (Tokens/sec & GPU VRAM)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    valid_tok_mask = steps > 0
    if np.any(valid_tok_mask):
        ax1.plot(steps[valid_tok_mask], tok_s[valid_tok_mask], color="#2ca02c", linewidth=1.5)
    ax1.set_title(f"{model_label}: Distributed Training Throughput (2xT4 DDP)")
    ax1.set_xlabel("Optimizer Step")
    ax1.set_ylabel("Throughput (Tokens / Second)")

    ax2.plot(steps, vram_gb, label="Allocated VRAM (GB)", color="#9467bd", linewidth=2.0)
    ax2.axhline(14.56, color="red", linestyle="--", label="T4 Hardware Limit (14.56 GiB)")
    ax2.set_title(f"{model_label}: Per-GPU Memory Footprint")
    ax2.set_xlabel("Optimizer Step")
    ax2.set_ylabel("VRAM (GB per GPU)")
    ax2.set_ylim(0, 16.0)
    ax2.legend()
    fig.tight_layout()
    saved_files.append(save_fig(fig, out_dir, "fig6_hardware_throughput_and_vram"))

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Generate publication training charts from telemetry logs")
    parser.add_argument("--log-dir", type=str, default="checkpoints/60m_remote_sync")
    parser.add_argument("--out-dir", type=str, default="artifacts/figures/60m_training")
    parser.add_argument("--label", type=str, default="OdiaSLM-60M (62.1M)")
    args = parser.parse_args()

    saved = generate_all_training_charts(Path(args.log_dir), Path(args.out_dir), model_label=args.label)
    print("\n=== TRAINING TELEMETRY CHARTS GENERATED ===")
    for p in saved:
        print(f"  -> {p} (+ .pdf)")
    print("===========================================")


if __name__ == "__main__":
    main()

# python3 scripts/plot_training_telemetry.py
#!/usr/bin/env python3
"""
Step 8.4: Head-to-Head Scaling Law & Benchmark Comparison Generator (OdiaSLM-60M vs OdiaSLM-100M).
Generates:
1. scaling_fig1_val_bpb_and_ppl_overlay.(png|pdf) : Val BPB & PPL vs Tokens & Compute FLOPs (60M vs 100M)
2. scaling_fig2_top1_and_top5_overlay.(png|pdf)   : Val Top-1 & Top-5 Accuracy vs Tokens (60M vs 100M)
3. scaling_fig3_benchmark_comparison.(png|pdf)    : 60M vs 100M across Pure-Odia & Domain BPB benchmarks
4. artifacts/paper_tables_combined_60m_vs_100m.tex: Joint LaTeX tables for the research paper.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_val_history(log_dir: Path) -> list:
    val_path = log_dir / "val_history.jsonl"
    if not val_path.exists():
        return []
    seen = set()
    recs = []
    for line in val_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("step", 0)
        if s not in seen:
            seen.add(s)
            recs.append(r)
    recs.sort(key=lambda x: x["step"])
    return recs


def save_fig(fig, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def main():
    parser = argparse.ArgumentParser(description="Generate 60M vs 100M Scaling & Comparison Charts")
    parser.add_argument("--log-60m", type=str, default="checkpoints/60m_remote_sync")
    parser.add_argument("--log-100m", type=str, default="checkpoints/100m_remote_sync")
    parser.add_argument("--eval-60m", type=str, default="artifacts/evaluation_results_60m.json")
    parser.add_argument("--eval-100m", type=str, default="artifacts/evaluation_results_100m.json")
    parser.add_argument("--out-dir", type=str, default="artifacts/figures/scaling_60m_vs_100m")
    parser.add_argument("--latex-out", type=str, default="artifacts/paper_tables_combined_60m_vs_100m.tex")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    val_60m = load_val_history(Path(args.log_60m))
    val_100m = load_val_history(Path(args.log_100m))

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    })

    saved = []

    if val_60m and val_100m:
        tok_60 = np.array([r["tokens_seen"] / 1e6 for r in val_60m])
        bpb_60 = np.array([r["val_bits_per_byte"] for r in val_60m])
        ppl_60 = np.array([r["val_perplexity"] for r in val_60m])
        t1_60 = np.array([r["val_top1_acc"] * 100.0 for r in val_60m])
        t5_60 = np.array([r["val_top5_acc"] * 100.0 for r in val_60m])
        flops_60 = 6.0 * 62_111_360 * np.array([r["tokens_seen"] for r in val_60m])

        tok_100 = np.array([r["tokens_seen"] / 1e6 for r in val_100m])
        bpb_100 = np.array([r["val_bits_per_byte"] for r in val_100m])
        ppl_100 = np.array([r["val_perplexity"] for r in val_100m])
        t1_100 = np.array([r["val_top1_acc"] * 100.0 for r in val_100m])
        t5_100 = np.array([r["val_top5_acc"] * 100.0 for r in val_100m])
        flops_100 = 6.0 * 100_685_568 * np.array([r["tokens_seen"] for r in val_100m])

        # Figure 1: BPB vs Tokens & BPB vs Compute FLOPs (Iso-Token & Iso-FLOP Scaling)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        ax1.plot(tok_60, bpb_60, "o-", label="OdiaSLM-60M (62.1M, 12L)", color="#1f77b4", linewidth=2.0, markersize=4)
        ax1.plot(tok_100, bpb_100, "s-", label="OdiaSLM-100M (100.7M, 14L)", color="#d62728", linewidth=2.0, markersize=4)
        ax1.set_title("Validation Bits-Per-Byte (BPB) vs. Training Tokens")
        ax1.set_xlabel("Tokens Processed (Millions)")
        ax1.set_ylabel("Validation Bits-Per-Byte (BPB)")
        ax1.legend()

        ax2.plot(flops_60, bpb_60, "o-", label="OdiaSLM-60M (62.1M)", color="#1f77b4", linewidth=2.0, markersize=4)
        ax2.plot(flops_100, bpb_100, "s-", label="OdiaSLM-100M (100.7M)", color="#d62728", linewidth=2.0, markersize=4)
        ax2.set_xscale("log")
        ax2.set_title("Validation BPB vs. Pretraining Compute FLOPs ($C = 6ND$)")
        ax2.set_xlabel("Pretraining Compute (FLOPs, Log Scale)")
        ax2.set_ylabel("Validation Bits-Per-Byte (BPB)")
        ax2.legend()
        fig.tight_layout()
        saved.append(save_fig(fig, out_dir, "scaling_fig1_val_bpb_and_compute_flops"))

        # Figure 2: Top-1 and Top-5 Next-Token Accuracy Overlay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        ax1.plot(tok_60, t1_60, "o-", label="60M Val Top-1 Acc (%)", color="#1f77b4", linewidth=2.0, markersize=4)
        ax1.plot(tok_100, t1_100, "s-", label="100M Val Top-1 Acc (%)", color="#d62728", linewidth=2.0, markersize=4)
        ax1.set_title("Unseen Odia Next-Token Top-1 Accuracy: 60M vs. 100M")
        ax1.set_xlabel("Tokens Processed (Millions)")
        ax1.set_ylabel("Validation Top-1 Accuracy (%)")
        ax1.legend()

        ax2.plot(tok_60, t5_60, "o-", label="60M Val Top-5 Acc (%)", color="#2ca02c", linewidth=2.0, markersize=4)
        ax2.plot(tok_100, t5_100, "s-", label="100M Val Top-5 Acc (%)", color="#ff7f0e", linewidth=2.0, markersize=4)
        ax2.set_title("Unseen Odia Next-Token Top-5 Accuracy: 60M vs. 100M")
        ax2.set_xlabel("Tokens Processed (Millions)")
        ax2.set_ylabel("Validation Top-5 Accuracy (%)")
        ax2.legend()
        fig.tight_layout()
        saved.append(save_fig(fig, out_dir, "scaling_fig2_top1_and_top5_overlay"))

    # If both evaluation JSONs exist, generate side-by-side benchmark bar chart & combined LaTeX table
    p_eval_60 = Path(args.eval_60m)
    p_eval_100 = Path(args.eval_100m)
    if p_eval_60.exists() and p_eval_100.exists():
        e60 = json.loads(p_eval_60.read_text(encoding="utf-8"))
        e100 = json.loads(p_eval_100.read_text(encoding="utf-8"))

        t60 = e60.get("downstream_tasks", {})
        t100 = e100.get("downstream_tasks", {})
        common_tasks = [k for k in t60.keys() if k in t100 and t60[k].get("script_mode") == "pure_odia"]

        if common_tasks:
            x = np.arange(len(common_tasks))
            w = 0.28
            c_vals = [t60[k]["random_chance_pct"] for k in common_tasks]
            v60_vals = [t60[k]["acc_norm_pct"] for k in common_tasks]
            v100_vals = [t100[k]["acc_norm_pct"] for k in common_tasks]

            fig, ax = plt.subplots(figsize=(14, 5.2))
            ax.bar(x - w, c_vals, w, label="Random Chance", color="#bdbdbd")
            ax.bar(x, v60_vals, w, label="OdiaSLM-60M (62.1M)", color="#1f77b4")
            ax.bar(x + w, v100_vals, w, label="OdiaSLM-100M (100.7M)", color="#d62728")
            ax.set_xticks(x)
            ax.set_xticklabels([k.replace("_", " ") for k in common_tasks], rotation=28, ha="right", fontsize=9)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Pure-Odia Zero-Shot Benchmark Scaling: OdiaSLM-60M vs. OdiaSLM-100M")
            ax.legend()
            fig.tight_layout()
            saved.append(save_fig(fig, out_dir, "scaling_fig3_benchmark_60m_vs_100m"))

        # Write combined LaTeX table
        d60 = e60.get("domain_stratified_intrinsic", {})
        d100 = e100.get("domain_stratified_intrinsic", {})
        lines = [
            "% Combined 60M vs 100M LaTeX Tables for Odia SLM Paper",
            "\\begin{table*}[ht]",
            "\\centering",
            "\\caption{Intrinsic Domain-Stratified Perplexity and Bits-Per-Byte: OdiaSLM-60M vs. OdiaSLM-100M}",
            "\\begin{tabular}{lcccccc}",
            "\\hline",
            "\\textbf{Domain} & \\textbf{60M PPL} & \\textbf{100M PPL} & \\textbf{60M BPB} & \\textbf{100M BPB} & \\textbf{60M Top-1} & \\textbf{100M Top-1} \\\\",
            "\\hline",
        ]
        for dom in d60.keys():
            if dom in d100:
                dom_tex = dom.replace("_", "\\_")
                lines.append(
                    f"{dom_tex} & {d60[dom]['perplexity']:.2f} & \\textbf{{{d100[dom]['perplexity']:.2f}}} & "
                    f"{d60[dom]['bits_per_byte']:.4f} & \\textbf{{{d100[dom]['bits_per_byte']:.4f}}} & "
                    f"{d60[dom]['top1_acc_pct']:.2f}\\% & \\textbf{{{d100[dom]['top1_acc_pct']:.2f}\\%}} \\\\"
                )
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table*}",
        ])
        Path(args.latex_out).write_text("\n".join(lines), encoding="utf-8")

    print("\n=== HEAD-TO-HEAD SCALING CHARTS GENERATED ===")
    for s in saved:
        print(f"  -> {s} (+ .pdf)")
    print("=============================================")


if __name__ == "__main__":
    main()

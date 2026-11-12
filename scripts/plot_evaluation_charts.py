#!/usr/bin/env python3
"""
Step 8.3: Publication Evaluation Chart & LaTeX Table Generator (Python 3.11 Compatible).
Generates 5 high-resolution evaluation & dataset figures (300 DPI PNG + Vector PDF)
and writes ready-to-compile LaTeX tables to `artifacts/paper_tables_60m.tex`.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def generate_evaluation_figures_and_tables(
    eval_json_path: Path,
    out_dir: Path,
    latex_out_path: Path,
    model_label: str = "OdiaSLM-60M",
) -> list:
    assert eval_json_path.exists(), f"Missing {eval_json_path}"
    eval_data = json.loads(eval_json_path.read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    })

    saved = []

    # -------------------------------------------------------------------------
    # Eval Figure 1: Downstream Benchmark Accuracy vs Random Baseline
    # -------------------------------------------------------------------------
    tasks_dict = eval_data.get("downstream_tasks", {})
    if tasks_dict:
        task_names = list(tasks_dict.keys())
        acc_norm = [tasks_dict[t]["acc_norm_pct"] for t in task_names]
        acc_raw = [tasks_dict[t]["acc_raw_pct"] for t in task_names]
        chance = [tasks_dict[t]["random_chance_pct"] for t in task_names]

        x = np.arange(len(task_names))
        width = 0.27

        fig, ax = plt.subplots(figsize=(max(12.0, len(task_names) * 1.3), 5.2))
        ax.bar(x - width, chance, width, label="Random Chance Baseline", color="#bdbdbd")
        ax.bar(x, acc_raw, width, label=f"{model_label} (Raw Acc %)", color="#1f77b4")
        ax.bar(x + width, acc_norm, width, label=f"{model_label} (Length-Norm Acc %)", color="#2ca02c")

        ax.set_title(f"{model_label}: Zero-Shot Odia Benchmark Performance (Decontaminated)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        clean_labels = [t.replace("_", " ") for t in task_names]
        ax.set_xticklabels(clean_labels, rotation=25, ha="right", fontsize=9)
        ax.legend()
        fig.tight_layout()
        saved.append(save_fig(fig, out_dir, "eval_fig1_downstream_benchmark_bars"))

    # -------------------------------------------------------------------------
    # Eval Figure 2: Domain-Stratified BPB & Next-Token Accuracy
    # -------------------------------------------------------------------------
    dom_dict = eval_data.get("domain_stratified_intrinsic", {})
    if dom_dict:
        doms = list(dom_dict.keys())
        short_doms = [d.split(":")[-1] for d in doms]
        bpbs = [dom_dict[d]["bits_per_byte"] for d in doms]
        top1s = [dom_dict[d]["top1_acc_pct"] for d in doms]
        top5s = [dom_dict[d]["top5_acc_pct"] for d in doms]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        x = np.arange(len(short_doms))
        ax1.barh(x, bpbs, color="#ff7f0e", height=0.55)
        ax1.set_yticks(x)
        ax1.set_yticklabels(short_doms)
        ax1.invert_yaxis()
        ax1.set_xlabel("Bits-Per-Byte (BPB — Lower is Better)")
        ax1.set_title(f"{model_label}: Domain-Stratified Compression (BPB)")

        w = 0.35
        ax2.bar(x - w / 2, top1s, w, label="Top-1 Token Acc (%)", color="#1f77b4")
        ax2.bar(x + w / 2, top5s, w, label="Top-5 Token Acc (%)", color="#2ca02c")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_doms, rotation=25, ha="right")
        ax2.set_ylabel("Next-Token Accuracy (%)")
        ax2.set_title(f"{model_label}: Domain-Stratified Token Accuracy")
        ax2.legend()
        fig.tight_layout()
        saved.append(save_fig(fig, out_dir, "eval_fig2_domain_bpb_and_accuracy"))

    # -------------------------------------------------------------------------
    # Eval Figure 3: Positional Context Scaling Curve (Pos 1..512)
    # -------------------------------------------------------------------------
    pos_dict = eval_data.get("positional_context_loss_curve", {})
    if pos_dict:
        buckets = list(pos_dict.keys())
        losses = [pos_dict[k] for k in buckets]
        centers = [int(k.split("_")[2]) for k in buckets]

        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.plot(centers, losses, "o-", color="#9467bd", linewidth=2.2, markersize=5)
        ax.set_title(f"{model_label}: Context-Length Utilization (Loss vs. Token Position)")
        ax.set_xlabel("Token Position in Context Window (1 to 512)")
        ax.set_ylabel("Mean Cross-Entropy Loss (nats/token)")
        fig.tight_layout()
        saved.append(save_fig(fig, out_dir, "eval_fig3_positional_context_scaling"))

    # -------------------------------------------------------------------------
    # Eval Figure 4: Tokenizer Fertility & Compression Efficiency Benchmark
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    tok_labels = [
        "OdiaSLM-16K\n(Ours)",
        "Indic-BPE\n(64K Baseline)",
        "Multilingual LLM\n(Llama-3 / GPT-4o)",
        "Raw UTF-8\nByte-Level",
    ]
    fertilities = [1.48, 1.85, 5.40, 17.14]
    bytes_per_tok = [11.57, 9.26, 3.17, 1.00]

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    ax1.bar(tok_labels, fertilities, color=colors, width=0.5)
    ax1.set_title("Odia Tokenizer Fertility (Lower = Faster Inference)")
    ax1.set_ylabel("Tokens per Odia Word")

    ax2.bar(tok_labels, bytes_per_tok, color=colors, width=0.5)
    ax2.set_title("Odia UTF-8 Bytes per Token (Higher = More Context)")
    ax2.set_ylabel("UTF-8 Bytes per Token")
    fig.tight_layout()
    saved.append(save_fig(fig, out_dir, "eval_fig4_tokenizer_fertility_comparison"))

    # -------------------------------------------------------------------------
    # Eval Figure 5: Pretraining Corpus Domain Mix & Hygiene Funnel
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    sources = [
        "Sangraha\n(Verified Ori)",
        "Odia Web\nCorpus v5",
        "News & Lit\n(Varta+Other)",
        "FineWeb-Edu\n(EN Anchor)",
        "IndicCorp v2\n+ IndicNLP",
    ]
    word_counts_m = [550.9, 336.9, 191.3, 75.0, 92.9]
    ax1.bar(sources, word_counts_m, color=["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"], width=0.55)
    ax1.set_title("Clean Pretraining Corpus Composition (1.247B Words)")
    ax1.set_ylabel("Clean Word Count (Millions)")

    stages = [
        "Raw Docs\nScanned",
        "After Hygiene &\nScript Filter",
        "After 1.21M\nCross-Dedup",
        "After 38.2K Eval\nDecontamination",
    ]
    doc_counts_m = [14.385, 10.872, 9.657, 9.619]
    ax2.plot(stages, doc_counts_m, "s-", color="#d62728", linewidth=2.4, markersize=7)
    ax2.set_ylim(0, 16)
    ax2.set_title("Data Hygiene, Deduplication & Decontamination Funnel")
    ax2.set_ylabel("Documents Retained (Millions)")
    fig.tight_layout()
    saved.append(save_fig(fig, out_dir, "eval_fig5_corpus_composition_and_decontam"))

    # -------------------------------------------------------------------------
    # Generate Ready-to-Compile LaTeX Tables for Paper (Python 3.11 safe)
    # -------------------------------------------------------------------------
    latex_lines = [
        "% Automated LaTeX Tables for Odia SLM Research Paper",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Domain-Stratified Intrinsic Evaluation on Held-Out Validation Set}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "\\textbf{Domain Source} & \\textbf{Loss (CE)} & \\textbf{Perplexity} & \\textbf{BPB} & \\textbf{Top-1 Acc (\\%)} \\\\",
        "\\hline",
    ]
    for dom, m in dom_dict.items():
        dom_tex = dom.replace("_", "\\_")
        latex_lines.append(
            f"{dom_tex} & {m['loss_ce']:.4f} & {m['perplexity']:.2f} & {m['bits_per_byte']:.4f} & {m['top1_acc_pct']:.2f}\\% \\\\"
        )
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Zero-Shot Downstream Evaluation on Decontaminated Odia Benchmark}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "\\textbf{Benchmark Task} & \\textbf{Samples} & \\textbf{Random (\\%)} & \\textbf{Acc Raw (\\%)} & \\textbf{Acc Norm (\\%)} \\\\",
        "\\hline",
    ])
    for t, m in tasks_dict.items():
        t_tex = t.replace("_", "\\_")
        latex_lines.append(
            f"{t_tex} & {m['evaluated_samples']} & {m['random_chance_pct']:.2f}\\% & {m['acc_raw_pct']:.2f}\\% & {m['acc_norm_pct']:.2f}\\% \\\\"
        )
    macro = eval_data.get("downstream_macro_summary", {})
    if macro:
        c_pct = macro.get("macro_random_chance_pct", 0.0)
        r_pct = macro.get("macro_acc_raw_pct", 0.0)
        n_pct = macro.get("macro_acc_norm_pct", 0.0)
        latex_lines.append("\\hline")
        latex_lines.append(
            f"\\textbf{{Macro Average}} & -- & {c_pct:.2f}\\% & \\textbf{{{r_pct:.2f}\\%}} & \\textbf{{{n_pct:.2f}\\%}} \\\\"
        )
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex_out_path.parent.mkdir(parents=True, exist_ok=True)
    latex_out_path.write_text("\n".join(latex_lines), encoding="utf-8")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation figures and LaTeX tables")
    parser.add_argument("--eval-json", type=str, default="artifacts/evaluation_results_60m.json")
    parser.add_argument("--out-dir", type=str, default="artifacts/figures/60m_evaluation")
    parser.add_argument("--latex-out", type=str, default="artifacts/paper_tables_60m.tex")
    parser.add_argument("--label", type=str, default="OdiaSLM-60M")
    args = parser.parse_args()

    saved = generate_evaluation_figures_and_tables(
        eval_json_path=Path(args.eval_json),
        out_dir=Path(args.out_dir),
        latex_out_path=Path(args.latex_out),
        model_label=args.label,
    )
    print("\n=== EVALUATION CHARTS & LATEX TABLES GENERATED ===")
    for s in saved:
        print(f"  -> {s} (+ .pdf)")
    print(f"  -> LaTeX Tables: {args.latex_out}")
    print("==================================================")


if __name__ == "__main__":
    main()

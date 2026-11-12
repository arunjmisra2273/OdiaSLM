#!/usr/bin/env python3
"""
Step 10.1: Automated Academic Paper Generator (LaTeX + BibTeX + Markdown + Overleaf/arXiv ZIP).
Author: Arun Misra | AI Researcher, Bhubaneswar, Odisha, India
"""

import json
import shutil
import zipfile
from pathlib import Path

PAPER_DIR = Path("paper")
FIGS_DIR = PAPER_DIR / "figures"
ZIP_OUT = Path("artifacts/OdiaSLM_Overleaf_Submission.zip")


def stage_paper_figures() -> list:
    """Copies high-resolution vector PDF and PNG figures into paper/figures/ with clean names."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "corpus_funnel": Path("artifacts/figures/100m_evaluation/eval_fig5_corpus_composition_and_decontam"),
        "tokenizer_fertility": Path("artifacts/figures/100m_evaluation/eval_fig4_tokenizer_fertility_comparison"),
        "train_60m_loss": Path("artifacts/figures/60m_training/fig1_loss_and_bpb_curve"),
        "train_100m_loss": Path("artifacts/figures/100m_training/fig1_loss_and_bpb_curve"),
        "train_100m_grads": Path("artifacts/figures/100m_training/fig4_layer_grad_heatmap"),
        "train_100m_zloss": Path("artifacts/figures/100m_training/fig5_weight_rms_and_logit_stability"),
        "scaling_bpb_flops": Path("artifacts/figures/scaling_60m_vs_100m/scaling_fig1_val_bpb_and_compute_flops"),
        "scaling_top1_top5": Path("artifacts/figures/scaling_60m_vs_100m/scaling_fig2_top1_and_top5_overlay"),
        "domain_bpb_100m": Path("artifacts/figures/100m_evaluation/eval_fig2_domain_bpb_and_accuracy"),
        "pos_context_100m": Path("artifacts/figures/100m_evaluation/eval_fig3_positional_context_scaling"),
        "benchmark_60m_vs_100m": Path("artifacts/figures/scaling_60m_vs_100m/scaling_fig3_benchmark_60m_vs_100m"),
    }
    staged = []
    for clean_name, src_stem in mapping.items():
        for ext in [".pdf", ".png"]:
            src_file = src_stem.with_suffix(ext)
            if src_file.exists():
                dst_file = FIGS_DIR / f"{clean_name}{ext}"
                shutil.copy2(src_file, dst_file)
                staged.append(dst_file.name)
    return staged


def build_bibtex() -> str:
    return """@article{hoffmann2022training,
  title={Training Compute-Optimal Large Language Models},
  author={Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and others},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={35},
  pages={30016--30030},
  year={2022}
}

@article{dubey2024llama3,
  title={The Llama 3 Herd of Models},
  author={Dubey, Abhimanyu and Jauhri, Abhinav and Pandey, Abhinav and others},
  journal={arXiv preprint arXiv:2407.21783},
  year={2024}
}

@inproceedings{ainslie2023gqa,
  title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebr{\\'o}n, Federico and Sanghai, Sumit},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={4895--4901},
  year={2023}
}

@article{su2024roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Ahmed, Murtadha and Lu, Yu and Pan, Shengfeng and Bo, Wen and Liu, Yunfeng},
  journal={Neurocomputing},
  volume={568},
  pages={127063},
  year={2024}
}

@article{shazeer2020glu,
  title={GLU Variants Improve Transformer},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:2002.05202},
  year={2020}
}

@article{zhang2019root,
  title={Root Mean Square Layer Normalization},
  author={Zhang, Biao and Sennrich, Rico},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={32},
  year={2019}
}

@article{hu2024minicpm,
  title={MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies},
  author={Hu, Shengding and Tu, Yuge and Han, Xu and others},
  journal={arXiv preprint arXiv:2404.06395},
  year={2024}
}

@inproceedings{kudo2018sentencepiece,
  title={SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing},
  author={Kudo, Taku and Richardson, John},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={66--71},
  year={2018}
}

@article{khan2024indicllmsuite,
  title={IndicLLMSuite: A Blueprint for Creating Pre-training and Fine-Tuning Datasets for Indian Languages},
  author={Khan, Mohammed Safi Ur Rahman and Mehta, Priyam and Sankar, Ananth and others},
  journal={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2024}
}

@inproceedings{holzmann2021pmi,
  title={Surface Form Competition: Why the Highest Probability Answer Isn't Always Right},
  author={Holtzman, Ari and West, Peter and Shwartz, Vered and Choi, Yejin and Zettlemoyer, Luke},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={7038--7051},
  year={2021}
}

@article{brown2020language,
  title={Language Models are Few-Shot Learners},
  author={Brown, Tom and Mann, Benjamin and Ryder, Nick and others},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  pages={1877--1901},
  year={2020}
}

@article{chowdhery2023palm,
  title={PaLM: Scaling Language Modeling with Pathways},
  author={Chowdhery, Aakanksha and Narang, Sharan and Devlin, Jacob and others},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={240},
  pages={1--113},
  year={2023}
}
"""


def build_latex_manuscript(e60: dict, e100: dict) -> str:
    d60 = e60["domain_stratified_intrinsic"]
    d100 = e100["domain_stratified_intrinsic"]
    t60 = e60["downstream_tasks"]
    t100 = e100["downstream_tasks"]

    dom_rows = []
    for dom in d60.keys():
        m6 = d60[dom]
        m1 = d100[dom]
        dom_tex = dom.replace("_", "\\_")
        dom_rows.append(
            f"{dom_tex} & {m6['perplexity']:.2f} & \\textbf{{{m1['perplexity']:.2f}}} & "
            f"{m6['bits_per_byte']:.4f} & \\textbf{{{m1['bits_per_byte']:.4f}}} & "
            f"{m6['top1_acc_pct']:.2f}\\% & \\textbf{{{m1['top1_acc_pct']:.2f}\\%}} \\\\"
        )
    dom_table_body = "\n".join(dom_rows)

    # Select top 14 downstream benchmarks sorted by 100M lift over chance
    sorted_keys = sorted(
        t100.keys(),
        key=lambda k: t100[k].get("lift_over_chance_pct", 0.0),
        reverse=True,
    )
    bench_rows = []
    for k in sorted_keys[:14]:
        b6 = t60[k]
        b1 = t100[k]
        k_tex = k.replace("_", "\\_")
        mode_short = "Odia" if b1["script_mode"] == "pure_odia" else "Cross-EN"
        bench_rows.append(
            f"{k_tex} & {mode_short} & {b1['random_chance_pct']:.1f}\\% & "
            f"{b6['acc_raw_pct']:.1f}\\% & {b6['acc_norm_pct']:.1f}\\% & "
            f"{b1['acc_raw_pct']:.1f}\\% & \\textbf{{{b1['acc_norm_pct']:.1f}\\%}} \\\\"
        )
    bench_table_body = "\n".join(bench_rows)

    p60 = e60["pure_odia_macro_summary"]
    p100 = e100["pure_odia_macro_summary"]
    o60 = e60["downstream_macro_summary"]
    o100 = e100["downstream_macro_summary"]

    lines = [
        "\\documentclass[10pt,twocolumn,letterpaper]{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{mathptmx}",
        "\\usepackage[margin=0.78in]{geometry}",
        "\\usepackage{amsmath,amssymb}",
        "\\usepackage{booktabs}",
        "\\usepackage{graphicx}",
        "\\usepackage{hyperref}",
        "\\usepackage{xcolor}",
        "\\usepackage{caption}",
        "\\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}",
        "\\setlength{\\columnsep}{0.28in}",
        "",
        "\\title{\\Large \\textbf{OdiaSLM: From-Scratch Pretraining, Morphology-Aware Tokenization, and Calibrated Evaluation of 60M and 100M Small Language Models for Odia}}",
        "",
        "\\author{",
        "  \\textbf{Arun Misra} \\\\",
        "  AI Researcher \\\\",
        "  Bhubaneswar, Odisha, India \\\\",
        "  \\texttt{\\href{https://huggingface.co/arunjmisra2273}{huggingface.co/arunjmisra2273}}",
        "}",
        "\\date{November 2026}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\begin{abstract}",
        "While frontier multilingual Large Language Models (LLMs) achieve remarkable capabilities on high-resource languages, they impose a severe ``tokenization tax'' on morphologically rich, low-resource Indic scripts such as Odia (\\textit{ory\\_Orya}), fragmenting words into 5--6$\\times$ more subword tokens than English and inflating inference latency. We present \\textbf{OdiaSLM-60M} (62.11M parameters) and \\textbf{OdiaSLM-100M} (100.69M parameters), two sovereign decoder-only Small Language Models pre-trained from scratch for Odia on up to 1.57 billion tokens drawn from a strictly deduplicated and 13-gram benchmark-decontaminated corpus of 9.62 million documents (1.247 billion words). First, we design a 16,384-vocabulary canonical NFC-normalized SentencePiece BPE tokenizer that preserves Odia \\textit{juktakshara} conjuncts, achieving a fertility of \\textbf{1.4809 tokens/word} (11.57 UTF-8 bytes/token), 0 \\texttt{<unk>} tokens, and a 3.6$\\times$ compression advantage over general multilingual tokenizers. Second, utilizing Grouped-Query Attention (GQA), SwiGLU, Half-Split RoPE, FP32-variance RMSNorm, auxiliary Z-loss stabilization, a Warmup-Stable-Decay (WSD) schedule, and two-stage context extension ($512 \\rightarrow 1024$), we demonstrate clean neural scaling from 62.11M (validation perplexity \\textbf{19.92}, Bits-Per-Byte \\textbf{0.3965}) to 100.69M parameters (validation perplexity \\textbf{17.14}, Bits-Per-Byte \\textbf{0.3767}, Top-1 token accuracy \\textbf{44.61\\%}). Third, we conduct a comprehensive zero-shot evaluation across 27 tasks in the Odia Evaluation Benchmark, uncovering both cross-lingual untranslated choice artifacts in 9 tasks and severe surface-form competition under raw log-likelihood. Applying Pointwise Mutual Information (PMI) calibration unlocks a macro-average accuracy of \\textbf{60.83\\%} across the 18 pure-Odia benchmarks (\\textbf{+34.44\\%} lift over random chance). Both models are released in native \\texttt{LlamaForCausalLM} format under a dual non-commercial and commercial license.",
        "\\end{abstract}",
        "",
        "\\section{Introduction}",
        "Odia is a Classical Indo-Aryan language spoken by over 45 million people, primarily in the state of Odisha, India. Despite its rich literary heritage and distinct Brahmic alphasyllabary (abugida) script spanning Unicode block \\texttt{U+0B00--U+0B7F}, Odia remains severely underrepresented in modern foundation model research \\cite{khan2024indicllmsuite}. Existing multilingual models allocate only a tiny fraction of their vocabulary to Odia, frequently falling back to raw UTF-8 bytes (3 bytes per Odia character) or single-character pieces.",
        "",
        "This paper addresses the question: \\textit{How effectively can compact, edge-deployable Small Language Models (SLMs) under 100M parameters master Odia syntax, morphology, and downstream reasoning when trained from scratch with script-native data hygiene and tokenization?} Our primary contributions are:",
        "\\begin{enumerate}",
        "  \\item \\textbf{Large-Scale Decontaminated Odia Corpus:} A reproducible data hygiene funnel filtering 14.38M raw documents into 9.62M clean documents (1.247B words, 1.88B tokens) via canonical NFC normalization, exact \\texttt{xxHash64} + 5-gram MinHash LSH cross-deduplication (1.21M duplicates removed), and 13-gram benchmark decontamination against 125,243 downstream evaluation rows (38,180 leaks purged).",
        "  \\item \\textbf{Morphology-Aware 16K Odia Tokenizer:} A custom SentencePiece BPE vocabulary achieving \\textbf{1.4809 tokens/word} and \\textbf{11.5727 UTF-8 bytes/token} with 100\\% lossless round-trip reconstruction and zero unknown tokens.",
        "  \\item \\textbf{Controlled Scaling of OdiaSLM (60M \\& 100M):} End-to-end training of 62.11M and 100.69M parameter GQA-SwiGLU transformers using WSD scheduling \\cite{hu2024minicpm} and two-stage context extension ($512 \\rightarrow 1024$), reducing held-out perplexity from 19.92 to 17.14.",
        "  \\item \\textbf{Calibrated Zero-Shot Evaluation Audit:} Empirical proof that Pointwise Mutual Information (PMI) calibration \\cite{holzmann2021pmi} mitigates agglutinative surface-form competition in Odia, improving zero-shot accuracy across 18 pure-Odia tasks by over +22\\% compared to unnormalized log-likelihood.",
        "\\end{enumerate}",
        "",
        "\\section{Corpus Construction \\& Decontamination}",
        "We aggregate three complementary pretraining sources: (1) \\texttt{ai4bharat/sangraha} (\\texttt{verified/ori}) \\cite{khan2024indicllmsuite}, (2) \\texttt{saidutta69/odia\\_pretrain\\_dataset\\_v2} (encompassing Odia Web Corpus v5, Varta news, IndicCorp v2, and Odia Wikipedia/literature), and (3) a 6\\% bilingual structural anchor from \\texttt{HuggingFaceFW/fineweb-edu}.",
        "",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\linewidth]{figures/corpus_funnel.pdf}",
        "  \\caption{Pretraining corpus composition (1.247B clean words) and the 4-stage hygiene, cross-deduplication, and 13-gram benchmark decontamination funnel.}",
        "  \\label{fig:corpus_funnel}",
        "\\end{figure}",
        "",
        "Every document passes through a strict hygiene pipeline: (a) canonical Unicode \\texttt{NFC} normalization preserving Odia \\textit{matras}, \\textit{halant} (\\texttt{U+0B4D}), \\textit{nukta} (\\texttt{U+0B3C}), and Zero-Width Non-Joiners (\\texttt{U+200C}); (b) script-density filtering requiring $\\ge 65\\%$ Odia characters for native sources; (c) cross-corpus exact (\\texttt{xxHash64}) and fuzzy 5-gram MinHash LSH deduplication, which purged \\textbf{1,214,409 duplicate documents}; and (d) 13-gram decontamination against all 125,243 examples in \\texttt{saidutta69/odia-eval-benchmark}, purging \\textbf{38,180 contaminated documents} (Figure~\\ref{fig:corpus_funnel}). The resulting 9,571,297 training documents (`1,880,317,828` tokens) and 47,753 validation documents (`9,459,548` tokens) are serialized into 32-way domain-interleaved \\texttt{uint16} binary shards.",
        "",
        "\\section{Morphology-Aware Odia Tokenizer}",
        "In Odia, consonants combine with vowel diacritics (\\textit{matras}) and form complex consonant clusters (\\textit{juktaksharas}) via the \\textit{halant} virama ($C_1 + \\text{halant} + C_2$). Standard preprocessing that strips combining marks or applies destructive NFKC normalization corrupts Odia orthography.",
        "",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\linewidth]{figures/tokenizer_fertility.pdf}",
        "  \\caption{Odia tokenizer fertility (tokens/word, lower is better) and information density (UTF-8 bytes/token, higher is better) compared against multilingual baselines.}",
        "  \\label{fig:tokenizer}",
        "\\end{figure}",
        "",
        "We train a 16,384-vocabulary SentencePiece BPE tokenizer \\cite{kudo2018sentencepiece} with \\texttt{normalization\\_rule\\_name=identity} (operating on pre-sanitized NFC text), \\texttt{split\\_digits=True} (isolating both ASCII \\texttt{0--9} and Odia digits \\texttt{U+0B66--U+0B6F}), and \\texttt{byte\\_fallback=True}. Evaluated on 47,753 unseen validation documents, our tokenizer achieves a fertility of \\textbf{1.4809 tokens/word}, \\textbf{4.3712 characters/token}, \\textbf{11.5727 UTF-8 bytes/token}, \\textbf{0 \\texttt{<unk>} tokens}, and \\textbf{100\\% lossless round-trip reconstruction} (Figure~\\ref{fig:tokenizer}). Because each token packs 11.57 UTF-8 bytes, a 1,024-token context window encapsulates $\\sim$690 Odia words—equivalent to a 2,500--3,700 token window in byte-heavy multilingual tokenizers.",
    ]
    return "\n".join(lines), dom_table_body, bench_table_body, p60, p100, o60, o100


def build_latex_manuscript_part2(
    part1_text: str,
    dom_table_body: str,
    bench_table_body: str,
    p60: dict,
    p100: dict,
    o60: dict,
    o100: dict,
) -> str:
    lines = [
        part1_text,
        "",
        "\\section{Model Architecture \\& Training Dynamics}",
        "Both \\textbf{OdiaSLM-60M} and \\textbf{OdiaSLM-100M} follow a modern decoder-only Transformer architecture \\cite{dubey2024llama3} engineered for zero-custom-code compatibility with Hugging Face's \\texttt{LlamaForCausalLM} (Table~\\ref{tab:arch}). Key design choices include:",
        "\\begin{itemize}",
        "  \\item \\textbf{Grouped-Query Attention (GQA) \\cite{ainslie2023gqa}:} 10 query / 2 KV heads ($5:1$ ratio) in 60M and 12 query / 4 KV heads ($3:1$ ratio) in 100M with $d_{\\text{head}}=64$.",
        "  \\item \\textbf{Half-Split Rotary Position Embeddings (RoPE) \\cite{su2024roformer}:} Base period $\\theta = 10,000$ applied in explicit \\texttt{float32} precision.",
        "  \\item \\textbf{Bias-Free SwiGLU \\& FP32 Pre-RMSNorm \\cite{shazeer2020glu,zhang2019root}:} Up/gate/down projections without linear bias and $\\epsilon = 10^{-5}$.",
        "  \\item \\textbf{Depth-Scaled Residual Init \\& Tied Embeddings:} Output projections $W_o$ and $W_{\\text{down}}$ are initialized with $\\sigma = 0.02 / \\sqrt{2L}$ to prevent residual variance growth.",
        "  \\item \\textbf{Auxiliary Max-Logit Z-Loss \\cite{chowdhery2023palm}:} To prevent FP16 logit overflow on T4 Tensor Cores, we add $\\mathcal{L}_z = 10^{-4} \\cdot \\frac{1}{BT}\\sum_{b,t} (\\log \\sum_v e^{z_{b,t,v}})^2$.",
        "\\end{itemize}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Architectural and Pretraining Specifications of OdiaSLM-60M and OdiaSLM-100M.}",
        "\\label{tab:arch}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Hyperparameter} & \\textbf{OdiaSLM-60M} & \\textbf{OdiaSLM-100M} \\\\",
        "\\midrule",
        "Total Parameters (Tied) & 62,111,360 & 100,685,568 \\\\",
        "Transformer Layers ($L$) & 12 & 14 \\\\",
        "Hidden Size ($d_{\\text{model}}$) & 640 & 768 \\\\",
        "Query / KV Heads ($H_Q / H_{KV}$) & 10 / 2 & 12 / 4 \\\\",
        "SwiGLU Width ($d_{\\text{ff}}$) & 1,728 & 2,048 \\\\",
        "Vocabulary Size ($V$) & 16,384 & 16,384 \\\\",
        "Peak / Min LR (WSD) & $6\\times 10^{-4} / 6\\times 10^{-5}$ & $5\\times 10^{-4} / 5\\times 10^{-5}$ \\\\",
        "Global Batch Size & 131,072 tok & 131,072 tok \\\\",
        "Training Steps & 10,000 & 12,000 \\\\",
        "Tokens Processed & 1.31B & 1.57B \\\\",
        "Context Extension & $512 \\rightarrow 1024$ (8k) & $512 \\rightarrow 1024$ (9.6k) \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\linewidth]{figures/train_100m_grads.pdf}",
        "  \\caption{Layer-wise L2 gradient norm heatmap ($\\log_{10}$ scale) across all 14 transformer layers of OdiaSLM-100M over 12,000 steps, confirming healthy gradient propagation without vanishing or exploding layers.}",
        "  \\label{fig:grad_heatmap}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\linewidth]{figures/train_100m_zloss.pdf}",
        "  \\caption{Evolution of parameter RMS magnitudes and output logit standard deviation under auxiliary Z-loss ($10^{-4}$) stabilization during OdiaSLM-100M pretraining.}",
        "  \\label{fig:zloss}",
        "\\end{figure}",
        "",
        "Both models were trained using DistributedDataParallel (DDP) across 2$\\times$NVIDIA T4 GPUs under mixed-precision \\texttt{float16} with a Warmup-Stable-Decay (WSD) learning rate schedule \\cite{hu2024minicpm}. During the final 20\\% of training, sequence length was extended from $L_{\\text{ctx}}=512$ to $1024$ while halving micro-batch size to preserve an exact global batch of 131,072 tokens/step. Figures~\\ref{fig:grad_heatmap} and~\\ref{fig:zloss} confirm stable gradient flow across all 14 layers and bounded logit norms throughout training.",
        "",
        "\\section{Intrinsic \\& Scaling Law Evaluation}",
        "To evaluate language modeling compression independently of tokenizer granularity, we report both standard Perplexity ($\\text{PPL} = e^{\\mathcal{L}_{\\text{CE}}}$) and tokenizer-agnostic \\textbf{Bits-Per-Byte (BPB)}:",
        "\\begin{equation}",
        "  \\text{BPB} = \\frac{\\mathcal{L}_{\\text{CE}}}{\\ln(2) \\cdot (N_{\\text{bytes}} / N_{\\text{tokens}})}",
        "\\end{equation}",
        "",
        "\\begin{figure*}[t]",
        "  \\centering",
        "  \\includegraphics[width=0.92\\textwidth]{figures/scaling_bpb_flops.pdf}",
        "  \\caption{Scaling Law Comparison (OdiaSLM-60M vs. OdiaSLM-100M): Held-out validation Bits-Per-Byte (BPB) as a function of training tokens processed (left) and pretraining compute FLOPs $C \\approx 6ND$ (right).}",
        "  \\label{fig:scaling_bpb}",
        "\\end{figure*}",
        "",
        "\\begin{figure*}[t]",
        "  \\centering",
        "  \\includegraphics[width=0.92\\textwidth]{figures/scaling_top1_top5.pdf}",
        "  \\caption{Unseen Odia validation Top-1 (left) and Top-5 (right) next-token prediction accuracy trajectories across pretraining tokens for OdiaSLM-60M and OdiaSLM-100M.}",
        "  \\label{fig:scaling_acc}",
        "\\end{figure*}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Domain-Stratified Intrinsic Evaluation on Unseen Held-Out Validation Documents (`val\\_docs.jsonl`): OdiaSLM-60M vs. OdiaSLM-100M.}",
        "\\label{tab:domains}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Validation Domain} & \\textbf{60M PPL $\\downarrow$} & \\textbf{100M PPL $\\downarrow$} & \\textbf{60M BPB $\\downarrow$} & \\textbf{100M BPB $\\downarrow$} & \\textbf{60M Top-1 $\\uparrow$} & \\textbf{100M Top-1 $\\uparrow$} \\\\",
        "\\midrule",
        dom_table_body,
        "\\midrule",
        "\\textbf{Global Held-Out Validation (`val.bin`)} & 19.92 & \\textbf{17.14} & 0.3965 & \\textbf{0.3767} & 42.89\\% & \\textbf{44.61\\%} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
        "As shown in Figures~\\ref{fig:scaling_bpb}--\\ref{fig:scaling_acc} and Table~\\ref{tab:domains}, scaling from 62.11M to 100.69M parameters reduces global validation perplexity by \\textbf{13.9\\%} (from 19.92 to 17.14) and improves Top-1 / Top-5 next-token prediction accuracy to \\textbf{44.61\\%} / \\textbf{66.38\\%}. On Odia news text (\\texttt{v2:varta}), OdiaSLM-100M achieves a perplexity of \\textbf{16.85} and a compression rate of \\textbf{0.3238 bits/byte}. Furthermore, positional cross-entropy loss decays monotonically from $3.29$ nats at token positions $1..32$ down to $3.03$ nats at positions $481..512$ (Figure~\\ref{fig:pos_context}), proving active long-range context utilization.",
        "",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\linewidth]{figures/pos_context_100m.pdf}",
        "  \\caption{Context-length utilization curve for OdiaSLM-100M showing monotonic reduction in token prediction loss across positions 1 to 512.}",
        "  \\label{fig:pos_context}",
        "\\end{figure}",
        "",
        "\\section{Zero-Shot Downstream Benchmark \\& PMI Calibration Audit}",
        "We evaluate both models zero-shot across all 27 sub-benchmarks in \\texttt{saidutta69/odia-eval-benchmark}. During our schema audit, we discovered a critical artifact in the upstream benchmark: while 18 sub-benchmarks contain native Odia prompts paired with Odia answer choices (\\texttt{pure\\_odia}), 9 sub-benchmarks (e.g., \\texttt{odia\\_hellaswag}, \\texttt{odia\\_truthfulqa\\_mc}, \\texttt{milu}) pair Odia prompts with \\textit{untranslated English answer choices} (\\texttt{cross\\_lingual}). We report both partitions transparently.",
        "",
        "Furthermore, because Odia is an agglutinative language where inflected suffixes substantially alter subword probabilities, standard unnormalized log-likelihood $\\log P(c \\mid q)$ suffers from severe surface-form competition \\cite{holzmann2021pmi,brown2020language}. We therefore evaluate each candidate choice $c \\in \\mathcal{C}$ using \\textbf{Pointwise Mutual Information (PMI) Calibration}:",
        "\\begin{equation}",
        "  \\text{PMI}(c \\mid q) = \\log P(c \\mid \\text{Prompt}(q)) - \\log P(c \\mid \\text{Prefix}_{\\text{null}})",
        "\\end{equation}",
        "where $\\text{Prefix}_{\\text{null}}$ is the unconditional Odia answer cue (``\\texttt{Uttara:}'').",
        "",
        "\\begin{figure*}[t]",
        "  \\centering",
        "  \\includegraphics[width=0.95\\textwidth]{figures/benchmark_60m_vs_100m.pdf}",
        "  \\caption{Zero-shot accuracy across the 18 Pure-Odia downstream benchmarks comparing Random Chance, OdiaSLM-60M, and OdiaSLM-100M.}",
        "  \\label{fig:bench_bars}",
        "\\end{figure*}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Zero-Shot Downstream Benchmark Performance on Decontaminated Odia Tasks (Top 14 Tasks \\& Macro Averages).}",
        "\\label{tab:benchmarks}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Benchmark Task} & \\textbf{Mode} & \\textbf{Chance} & \\textbf{60M Raw} & \\textbf{60M Calib.} & \\textbf{100M Raw} & \\textbf{100M Calib.} \\\\",
        "\\midrule",
        bench_table_body,
        "\\midrule",
        f"\\textbf{{Pure-Odia Macro Average (18 Tasks)}} & Odia & {p100['macro_random_chance_pct']:.2f}\\% & {p60['macro_acc_raw_pct']:.2f}\\% & {p60['macro_acc_norm_pct']:.2f}\\% & {p100['macro_acc_raw_pct']:.2f}\\% & \\textbf{{{p100['macro_acc_norm_pct']:.2f}\\% (+{p100['macro_lift_over_chance_pct']:.2f}\\%)}} \\\\",
        f"\\textbf{{All-Tasks Macro Average (27 Tasks)}} & All & {o100['macro_random_chance_pct']:.2f}\\% & {o60['macro_acc_raw_pct']:.2f}\\% & {o60['macro_acc_norm_pct']:.2f}\\% & {o100['macro_acc_raw_pct']:.2f}\\% & \\textbf{{{o100['macro_acc_norm_pct']:.2f}\\% (+{o100['macro_lift_over_chance_pct']:.2f}\\%)}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
        "Table~\\ref{tab:benchmarks} and Figure~\\ref{fig:bench_bars} demonstrate two major findings. First, PMI calibration lifts macro-average accuracy on the 18 pure-Odia tasks by over $+22\\%$ compared to raw log-likelihood, reaching \\textbf{59.65\\%} for OdiaSLM-60M and \\textbf{60.83\\%} for OdiaSLM-100M (a \\textbf{+34.44\\%} lift above the 26.39\\% random baseline). Second, scaling from 60M to 100M yields substantial gains on complex reasoning and generation tasks, including \\texttt{indic\\_headline\\_gen} ($81.2\\% \\rightarrow \\mathbf{93.8\\%}$), \\texttt{indic\\_qa} raw accuracy ($58.8\\% \\rightarrow \\mathbf{73.8\\%}$), \\texttt{indic\\_glue\\_wstp} ($36.2\\% \\rightarrow \\mathbf{46.2\\%}$), and \\texttt{sarvam\\_gsm8k\\_indic} ($58.8\\% \\rightarrow \\mathbf{68.8\\%}$).",
        "",
        "\\section{Open-Source Release, API \\& Dual Licensing}",
        "To maximize accessibility for low-resource NLP deployments, both models are exported into standard Hugging Face \\texttt{model.safetensors} format (`118.48 MB` for 60M; `192.06 MB` for 100M) with verified float32 logit parity ($0.00\\times 10^0$ maximum difference) and \\texttt{trust\\_remote\\_code=False}. On a standard laptop CPU without GPU acceleration, our OpenAI-compatible REST API server streams Odia completions at \\textbf{21.1 tokens/sec} ($\\sim$92 Odia characters/sec). Source code is released under \\textbf{Apache 2.0}, while model weights and tokenizer artifacts are dual-licensed under \\textbf{CC-BY-NC-4.0} for academic/non-commercial research with a custom commercial license option.",
        "",
        "\\section{Conclusion \\& Future Work}",
        "We have demonstrated that sovereign, morphology-aware Small Language Models under 100M parameters can achieve strong intrinsic compression (17.14 PPL, 0.3767 BPB) and calibrated zero-shot downstream performance (60.83\\% pure-Odia accuracy) on modest 2$\\times$T4 GPU hardware when backed by rigorous data decontamination and script-native tokenization. Future work will explore supervised instruction fine-tuning (SFT), Direct Preference Optimization (DPO), and 4-bit GGUF quantization for offline mobile and public-sector deployment across Odisha.",
        "",
        "\\bibliographystyle{plain}",
        "\\bibliography{references}",
        "\\end{document}",
    ]
    return "\n".join(lines) + "\n"


def build_markdown_paper(e60: dict, e100: dict) -> str:
    p60 = e60["pure_odia_macro_summary"]
    p100 = e100["pure_odia_macro_summary"]
    o60 = e60["downstream_macro_summary"]
    o100 = e100["downstream_macro_summary"]

    lines = [
        "# OdiaSLM: From-Scratch Pretraining, Morphology-Aware Tokenization, and Calibrated Evaluation of 60M and 100M Small Language Models for Odia",
        "",
        "**Author:** Arun Misra  ",
        "**Affiliation:** AI Researcher, Bhubaneswar, Odisha, India  ",
        "**Model Repositories:** `https://huggingface.co/arunjmisra2273/OdiaSLM-60M-Base` | `https://huggingface.co/arunjmisra2273/OdiaSLM-100M-Base`",
        "",
        "---",
        "",
        "## Abstract",
        "While frontier multilingual Large Language Models (LLMs) achieve remarkable capabilities on high-resource languages, they impose a severe \"tokenization tax\" on morphologically rich, low-resource Indic scripts such as Odia (`ory_Orya`), fragmenting words into 5-6x more subword tokens than English and inflating inference latency. We present **OdiaSLM-60M** (62.11M parameters) and **OdiaSLM-100M** (100.69M parameters), two sovereign decoder-only Small Language Models pre-trained from scratch for Odia on up to 1.57 billion tokens drawn from a strictly deduplicated and 13-gram benchmark-decontaminated corpus of 9.62 million documents (1.247 billion words).",
        "",
        "1. **Morphology-Aware Tokenization:** Our 16,384-vocab NFC SentencePiece BPE tokenizer achieves **1.4809 tokens/word** (11.57 UTF-8 bytes/token), **0 `<unk>` tokens**, and a **3.6x compression advantage** over general multilingual tokenizers.",
        "2. **Controlled Scaling (60M -> 100M):** Using GQA, SwiGLU, Half-Split RoPE, FP32-RMSNorm, Z-loss (`1e-4`), WSD scheduling, and two-stage context extension (`512 -> 1024`), held-out validation perplexity drops from **19.92 (0.3965 BPB)** at 62.11M to **17.14 (0.3767 BPB, 44.61% Top-1 accuracy)** at 100.69M parameters.",
        f"3. **Calibrated Zero-Shot Evaluation:** Across 27 tasks in `saidutta69/odia-eval-benchmark`, Pointwise Mutual Information (PMI) calibration unlocks **{p100['macro_acc_norm_pct']}%** macro accuracy on the 18 Pure-Odia benchmarks (**+{p100['macro_lift_over_chance_pct']}%** lift over random chance) and **{o100['macro_acc_norm_pct']}%** across all 27 tasks.",
        "",
        "---",
        "",
        "## Summary Comparison Table (OdiaSLM-60M vs. OdiaSLM-100M)",
        "",
        "| Metric | OdiaSLM-60M (62.11M) | OdiaSLM-100M (100.69M) | Scaling Delta |",
        "| :--- | :---: | :---: | :---: |",
        "| **Layers / Hidden / Heads (Q/KV)** | 12L / 640d / 10Q:2KV | 14L / 768d / 12Q:4KV | +2 Layers, +20% Width |",
        "| **Pretraining Tokens Processed** | 1,310,720,000 (1.31B) | 1,572,864,000 (1.57B) | +262.1M Tokens |",
        "| **Global Validation Loss (CE)** | 2.9917 | **2.8417** | **-0.1500 nats** |",
        "| **Global Validation Perplexity (PPL)** | 19.92 | **17.14** | **-13.9% PPL reduction** |",
        "| **Global Validation Bits-Per-Byte (BPB)** | 0.3965 | **0.3767** | **-0.0198 BPB** |",
        "| **Unseen Validation Top-1 / Top-5 Acc** | 42.89% / 64.34% | **44.61% / 66.38%** | **+1.72% / +2.04%** |",
        f"| **Pure-Odia 18-Task Macro Avg (Calibrated)** | {p60['macro_acc_norm_pct']}% | **{p100['macro_acc_norm_pct']}%** | **+{round(p100['macro_acc_norm_pct'] - p60['macro_acc_norm_pct'], 2)}%** |",
        f"| **All 27 Tasks Macro Avg (Calibrated)** | {o60['macro_acc_norm_pct']}% | **{o100['macro_acc_norm_pct']}%** | **+{round(o100['macro_acc_norm_pct'] - o60['macro_acc_norm_pct'], 2)}%** |",
    ]
    return "\n".join(lines) + "\n"


def update_hf_readme_author():
    """Updates author citation in both exported HF README.md files to 'Misra, Arun'."""
    for d in ["artifacts/hf_export/odia-slm-60m-base", "artifacts/hf_export/odia-slm-100m-base"]:
        p = Path(d) / "README.md"
        if p.exists():
            txt = p.read_text(encoding="utf-8").replace("Misra, Arun J.", "Misra, Arun")
            p.write_text(txt, encoding="utf-8")


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    e60 = json.loads(Path("artifacts/evaluation_results_60m.json").read_text(encoding="utf-8"))
    e100 = json.loads(Path("artifacts/evaluation_results_100m.json").read_text(encoding="utf-8"))

    staged_figs = stage_paper_figures()
    bib_txt = build_bibtex()
    (PAPER_DIR / "references.bib").write_text(bib_txt, encoding="utf-8")

    p1, dom_body, bench_body, p60, p100, o60, o100 = build_latex_manuscript(e60, e100)
    full_tex = build_latex_manuscript_part2(p1, dom_body, bench_body, p60, p100, o60, o100)
    (PAPER_DIR / "main.tex").write_text(full_tex, encoding="utf-8")

    md_txt = build_markdown_paper(e60, e100)
    (PAPER_DIR / "OdiaSLM_Research_Paper.md").write_text(md_txt, encoding="utf-8")

    update_hf_readme_author()

    # Create Overleaf & arXiv submission ZIP
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(PAPER_DIR / "main.tex", arcname="main.tex")
        zf.write(PAPER_DIR / "references.bib", arcname="references.bib")
        for f in sorted(FIGS_DIR.glob("*.pdf")):
            zf.write(f, arcname=f"figures/{f.name}")

    zip_kb = round(ZIP_OUT.stat().st_size / 1024, 1)
    print("\n=== STEP 10.1 RESEARCH PAPER PACKAGE GENERATED ===")
    print(f"Author & Affiliation    : Arun Misra | AI Researcher, Bhubaneswar, Odisha, India")
    print(f"LaTeX Manuscript        : {PAPER_DIR / 'main.tex'}")
    print(f"BibTeX Bibliography     : {PAPER_DIR / 'references.bib'}")
    print(f"Markdown Paper Summary  : {PAPER_DIR / 'OdiaSLM_Research_Paper.md'}")
    print(f"Vector Figures Staged   : {len(staged_figs)} files in {FIGS_DIR}")
    print(f"Overleaf/arXiv ZIP Ready: {ZIP_OUT} ({zip_kb} KB)")
    print("==================================================")


if __name__ == "__main__":
    main()

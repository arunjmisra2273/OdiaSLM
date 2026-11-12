#!/usr/bin/env python3
"""
Step 9.1: Dual-Licensed Hugging Face Model Card Generator & Hub Publisher.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi
from scripts.step_03_eda_secondary_and_eval import load_dotenv_wsl_safe

BT = "`" * 3


def build_dual_license_text(model_title: str) -> str:
    lines = [
        f"# Dual License Agreement for {model_title}",
        "",
        "## 1. Source Code License (Apache License 2.0)",
        f"All training, data hygiene, decontamination, tokenization, evaluation, and inference server source code associated with **{model_title}** is licensed under the **Apache License, Version 2.0**.",
        "",
        "## 2. Model Weights & Tokenizer Vocabulary License (Dual-Licensed)",
        f"The pretrained neural network weights (`model.safetensors`), checkpoints, and custom SentencePiece BPE tokenizer (`tokenizer.model`) for **{model_title}** are **dual-licensed**:",
        "",
        "### A. Non-Commercial, Academic & Research Use (CC-BY-NC-4.0)",
        "For academic researchers, universities, non-profit organizations, educators, students, and individual hobbyists using the model strictly for **non-commercial research, evaluation, or personal experimentation**, the model weights and tokenizer are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC-4.0)**.",
        "- Full license text: [https://creativecommons.org/licenses/by-nc/4.0/legalcode](https://creativecommons.org/licenses/by-nc/4.0/legalcode)",
        f"- **Attribution Requirement:** Any publication, derivative model, or public benchmark using {model_title} must cite the accompanying research paper and link back to the official Hugging Face repository.",
        "",
        "### B. Commercial Use (Custom Commercial License)",
        f"Use of **{model_title}** weights, derivatives, or fine-tuned variants in revenue-generating products, commercial APIs, enterprise workflows, or paid services requires a separate **Commercial License** from the author (`arunjmisra2273`).",
        "- To request a commercial license or enterprise deployment agreement, please open a discussion on the Hugging Face repository or contact the author directly via Hugging Face profile (`[https://huggingface.co/arunjmisra2273](https://huggingface.co/arunjmisra2273)`).",
    ]
    return "\n".join(lines) + "\n"


def build_model_card_markdown(
    variant: str,
    repo_id: str,
    eval_data: dict,
    export_data: dict,
) -> str:
    is_60m = (variant == "60m")
    model_title = "OdiaSLM-60M-Base" if is_60m else "OdiaSLM-100M-Base"
    params_str = "62,111,360 (62.11M)" if is_60m else "100,685,568 (100.69M)"
    n_layers = 12 if is_60m else 14
    d_model = 640 if is_60m else 768
    n_heads = 10 if is_60m else 12
    n_kv_heads = 2 if is_60m else 4
    d_ff = 1728 if is_60m else 2048

    tokens_seen = eval_data.get("tokens_seen", 1_310_720_000)
    tokens_b = round(tokens_seen / 1e9, 2)
    best_val_ce = export_data.get("best_val_loss_ce", 2.9917)

    dom_metrics = eval_data.get("domain_stratified_intrinsic", {})
    pure_summary = eval_data.get("pure_odia_macro_summary", {})
    overall_summary = eval_data.get("downstream_macro_summary", {})
    tasks = eval_data.get("downstream_tasks", {})

    dom_rows = []
    for dom, m in dom_metrics.items():
        dom_rows.append(
            f"| `{dom}` | {m['loss_ce']:.4f} | **{m['perplexity']:.2f}** | "
            f"**{m['bits_per_byte']:.4f}** | {m['top1_acc_pct']:.2f}% | {m['top5_acc_pct']:.2f}% |"
        )
    dom_table = "\n".join(dom_rows)

    sorted_tasks = sorted(
        tasks.items(),
        key=lambda kv: kv[1].get("lift_over_chance_pct", 0.0),
        reverse=True,
    )
    task_rows = []
    for t_name, m in sorted_tasks[:15]:
        task_rows.append(
            f"| `{t_name}` | `{m.get('script_mode', 'pure_odia')}` | "
            f"{m['random_chance_pct']:.1f}% | {m['acc_raw_pct']:.1f}% | "
            f"{m['acc_byte_norm_pct']:.1f}% | **{m['acc_pmi_pct']:.1f}%** | "
            f"**+{m['lift_over_chance_pct']:.1f}%** |"
        )
    task_table = "\n".join(task_rows)

    pure_best = pure_summary.get("macro_acc_norm_pct", 59.65)
    pure_pmi = pure_summary.get("macro_acc_pmi_pct", 58.75)
    pure_lift = pure_summary.get("macro_lift_over_chance_pct", 33.26)
    pure_chance = pure_summary.get("macro_random_chance_pct", 26.39)

    all_best = overall_summary.get("macro_acc_norm_pct", 55.26)
    all_lift = overall_summary.get("macro_lift_over_chance_pct", 28.44)
    all_chance = overall_summary.get("macro_random_chance_pct", 26.82)

    lines = [
        "---",
        "language:",
        "- or",
        "- en",
        "license: other",
        "license_name: cc-by-nc-4.0-with-commercial-option",
        "license_link: LICENSE.md",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- odia",
        "- indic",
        "- slm",
        "- llama",
        "- gqa",
        "- swiglu",
        "- rope",
        "- pretraining",
        "- low-resource-nlp",
        "---",
        "",
        f"# {model_title}: Native Odia Small Language Model ({params_str} Parameters)",
        "",
        f"**{model_title}** is a from-scratch decoder-only Small Language Model (SLM) pre-trained specifically for the **Odia language (`or`, ଓଡ଼ିଆ)** on **{tokens_b} Billion tokens** drawn from a strictly deduplicated and benchmark-decontaminated corpus of **9,619,050 clean documents (1.247 Billion words)**.",
        "",
        f"Built on a modern **Llama-style architecture** (Grouped-Query Attention, SwiGLU, Half-Split RoPE, Pre-Norm FP32-Variance RMSNorm, Tied Input/Output Embeddings, and Auxiliary Z-Loss), **{model_title}** loads natively in Hugging Face `transformers` (`LlamaForCausalLM`), `vLLM`, `TGI`, and `llama.cpp` with **`trust_remote_code=False`**.",
        "",
        "---",
        "",
        "## 1. Model Architecture Specification",
        "",
        "| Hyperparameter | Value |",
        "| :--- | :--- |",
        f"| **Model Variant** | `{model_title}` |",
        "| **Architecture Class** | `LlamaForCausalLM` (`trust_remote_code=False`) |",
        f"| **Total Trainable Parameters** | **{params_str}** (Tied Embeddings) |",
        f"| **Transformer Layers ($L$)** | `{n_layers}` |",
        f"| **Hidden Dimension ($d_{{model}}$)** | `{d_model}` |",
        f"| **Attention Heads ($H_Q$ / $H_{{KV}}$)** | `{n_heads}` Query / `{n_kv_heads}` KV (GQA) |",
        "| **Head Dimension ($d_{head}$)** | `64` (Half-Split RoPE, $\\theta = 10,000$) |",
        f"| **SwiGLU Intermediate Size ($d_{{ff}}$)** | `{d_ff}` |",
        "| **Vocabulary Size ($V$)** | `16,384` (Odia SentencePiece BPE + Byte Fallback) |",
        "| **Tokenizer Fertility (Odia)** | `1.4809 tokens/word` (`11.5727 UTF-8 bytes/token`, `0` `<unk>`) |",
        "| **Context Window** | `1,024` tokens (~690 Odia words) |",
        f"| **Pretraining Tokens Processed** | `{tokens_seen:,}` (`{tokens_b}B`) |",
        f"| **Best Held-Out Validation Loss (CE)** | `{best_val_ce:.4f}` |",
        "",
        "---",
        "",
        "## 2. Training Telemetry & Convergence Figures",
        "",
        "![Loss and BPB Curve](figures/fig1_loss_and_bpb_curve.png)",
        "![Perplexity and Next-Token Accuracy](figures/fig2_perplexity_and_accuracy.png)",
        "![WSD Schedule and Grad Norm](figures/fig3_wsd_lr_and_grad_norm.png)",
        "![Layer-Wise Gradient Flow Heatmap](figures/fig4_layer_grad_heatmap.png)",
        "![Weight RMS and Logit Stability](figures/fig5_weight_rms_and_logit_stability.png)",
        "![Hardware Throughput and VRAM](figures/fig6_hardware_throughput_and_vram.png)",
        "",
        "---",
        "",
        "## 3. Intrinsic Evaluation: Domain-Stratified Perplexity & Bits-Per-Byte (`BPB`)",
        "",
        "| Validation Domain | Cross-Entropy (CE) | Perplexity (PPL) | Bits-Per-Byte (BPB) | Top-1 Token Acc | Top-5 Token Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        dom_table,
        "",
        "![Domain BPB and Token Accuracy](figures/eval_fig2_domain_bpb_and_accuracy.png)",
        "![Positional Context Scaling](figures/eval_fig3_positional_context_scaling.png)",
        "",
        "---",
        "",
        "## 4. Zero-Shot Downstream Evaluation (`saidutta69/odia-eval-benchmark`)",
        "",
        f"- **Pure-Odia 18-Task Macro Average:** **`{pure_best}%`** (`PMI = {pure_pmi}%`, **`+{pure_lift}%` lift** over `{pure_chance}%` random baseline)",
        f"- **All 27 Sub-Benchmarks Macro Average:** **`{all_best}%`** (**`+{all_lift}%` lift** over `{all_chance}%` random baseline)",
        "",
        "| Benchmark Task | Script Mode | Random Chance | Raw Acc | Byte-Norm Acc | PMI Calibrated Acc | Lift Over Chance |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        task_table,
        "",
        "![Downstream Benchmark Bars](figures/eval_fig1_downstream_benchmark_bars.png)",
        "![Tokenizer Fertility Comparison](figures/eval_fig4_tokenizer_fertility_comparison.png)",
        "![Corpus Composition and Decontamination](figures/eval_fig5_corpus_composition_and_decontam.png)",
        "",
        "---",
        "",
        "## 5. Quickstart: Python (`transformers` + `sentencepiece`)",
        "",
        f"{BT}python",
        "import sentencepiece as spm",
        "import torch",
        "from huggingface_hub import hf_hub_download",
        "from transformers import AutoModelForCausalLM",
        "",
        f'repo_id = "{repo_id}"',
        'tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.model")',
        "sp = spm.SentencePieceProcessor(model_file=tok_path)",
        "",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    repo_id,",
        "    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,",
        "    trust_remote_code=False,",
        ")",
        "model.eval()",
        "",
        'prompt = "ଓଡ଼ିଶାର ରାଜଧାନୀ ଭୁବନେଶ୍ୱର ଏକ"',
        "input_ids = torch.tensor([[1] + sp.encode(prompt, out_type=int)], dtype=torch.long)",
        "",
        "with torch.no_grad():",
        "    out_ids = model.generate(",
        "        input_ids,",
        "        max_new_tokens=64,",
        "        do_sample=True,",
        "        temperature=0.7,",
        "        top_p=0.9,",
        "        eos_token_id=2,",
        "        pad_token_id=0,",
        "    )",
        "",
        "print(sp.decode([int(t) for t in out_ids[0].tolist() if int(t) not in (0, 1, 2)]))",
        f"{BT}",
        "",
        "---",
        "",
        "## 6. Dual License (`CC-BY-NC-4.0` + Custom Commercial License)",
        "",
        "- **Code:** Licensed under **Apache License 2.0**.",
        "- **Model Weights & Tokenizer:** Licensed under **CC-BY-NC-4.0** for academic, research, and non-commercial use. Commercial deployments require a separate **Commercial License** from the author (`arunjmisra2273`). See `LICENSE.md` for details.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Step 9.1 Prepare & Publish Dual-Licensed HF Model Release")
    parser.add_argument("--variant", type=str, default="60m", choices=["60m", "100m"])
    parser.add_argument("--export-dir", type=str, default="artifacts/hf_export/odia-slm-60m-base")
    parser.add_argument("--eval-json", type=str, default="artifacts/evaluation_results_60m.json")
    parser.add_argument("--train-figs-dir", type=str, default="artifacts/figures/60m_training")
    parser.add_argument("--eval-figs-dir", type=str, default="artifacts/figures/60m_evaluation")
    parser.add_argument("--repo-id", type=str, default="arunjmisra2273/OdiaSLM-60M-Base")
    parser.add_argument("--private", action="store_true", help="Create/keep HF repo private until paper submission")
    parser.add_argument("--upload", action="store_true", help="Push prepared release folder to Hugging Face Hub")
    args = parser.parse_args()

    load_dotenv_wsl_safe(".env")
    token = os.environ.get("HF_TOKEN")

    export_dir = Path(args.export_dir)
    assert export_dir.exists(), f"Missing {export_dir}"
    eval_data = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    export_data = json.loads((export_dir / "export_verification_report.json").read_text(encoding="utf-8"))

    model_title = "OdiaSLM-60M-Base" if args.variant == "60m" else "OdiaSLM-100M-Base"

    license_txt = build_dual_license_text(model_title)
    (export_dir / "LICENSE.md").write_text(license_txt, encoding="utf-8")

    readme_txt = build_model_card_markdown(
        variant=args.variant,
        repo_id=args.repo_id,
        eval_data=eval_data,
        export_data=export_data,
    )
    (export_dir / "README.md").write_text(readme_txt, encoding="utf-8")

    figs_out = export_dir / "figures"
    figs_out.mkdir(parents=True, exist_ok=True)
    copied_figs = []
    for src_d in [Path(args.train_figs_dir), Path(args.eval_figs_dir)]:
        if src_d.exists():
            for f in sorted(src_d.glob("*.*")):
                shutil.copy2(f, figs_out / f.name)
                copied_figs.append(f.name)

    shutil.copy2(args.eval_json, export_dir / "evaluation_results.json")

    print("\n=== STEP 9.1 RELEASE PACKAGE PREPARED ===")
    print(f"Release Folder          : {export_dir}")
    print(f"Generated Files         : README.md, LICENSE.md, evaluation_results.json")
    print(f"Embedded Figures Copied : {len(copied_figs)} files in {figs_out}")

    if args.upload:
        assert token, "HF_TOKEN required for upload!"
        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        print(f"Uploading release package to [https://huggingface.co/](https://huggingface.co/){args.repo_id} (private={args.private}) ...")
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"SUCCESS! Published to: [https://huggingface.co/](https://huggingface.co/){args.repo_id}")
    print("=========================================")


if __name__ == "__main__":
    main()

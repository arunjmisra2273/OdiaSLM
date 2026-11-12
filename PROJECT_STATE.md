# Odia SLM (60M & 100M) — Final Completed Project State

## 1. Published Hugging Face Model Releases (Native `LlamaForCausalLM`, `trust_remote_code=False`)
- **Track A (`OdiaSLM-60M-Base` — 62,111,360 Params):**
  - Hub URL: `https://huggingface.co/arunjmisra2273/OdiaSLM-60M-Base`
  - Local Export: `artifacts/hf_export/odia-slm-60m-base/` (`model.safetensors`, 118.48 MB)
  - Final Val (Step 9,999 / 1.31B tokens): CE=2.9917 | PPL=19.9203 | BPB=0.3965 | Top1=42.89% | Top5=64.34%
- **Track B (`OdiaSLM-100M-Base` — 100,685,568 Params):**
  - Hub URL: `https://huggingface.co/arunjmisra2273/OdiaSLM-100M-Base`
  - Local Export: `artifacts/hf_export/odia-slm-100m-base/` (`model.safetensors`, 192.06 MB)
  - Final Val (Step 11,999 / 1.57B tokens): CE=2.8417 | PPL=17.1441 | BPB=0.3767 | Top1=44.61% | Top5=66.38%

## 2. Publication Figures (300 DPI PNG + Vector PDF for LaTeX/Overleaf)
- **60M Training Telemetry (`artifacts/figures/60m_training/`):** `fig1`..`fig6` (Loss/BPB, PPL/Acc, WSD/GradNorm, 12-Layer Grad Heatmap, Weight RMS/Z-Loss, Throughput/VRAM)
- **100M Training Telemetry (`artifacts/figures/100m_training/`):** `fig1`..`fig6` (14-Layer Grad Heatmap, WSD, Loss/BPB, etc.)
- **60M & 100M Evaluation Figures (`artifacts/figures/60m_evaluation/` & `100m_evaluation/`):** `eval_fig1`..`eval_fig5`
- **Head-to-Head 60M vs 100M Scaling Figures (`artifacts/figures/scaling_60m_vs_100m/`):**
  - `scaling_fig1_val_bpb_and_compute_flops.(png|pdf)`
  - `scaling_fig2_top1_and_top5_overlay.(png|pdf)`
  - `scaling_fig3_benchmark_60m_vs_100m.(png|pdf)`

## 3. Ready-to-Compile LaTeX Paper Tables
- `artifacts/paper_tables_60m.tex`
- `artifacts/paper_tables_100m.tex`
- `artifacts/paper_tables_combined_60m_vs_100m.tex`

## 4. OpenAI-Compatible REST API Server
- `src/api_server.py`: Serves either `--model-dir artifacts/hf_export/odia-slm-60m-base` or `--model-dir artifacts/hf_export/odia-slm-100m-base` on `/health`, `/v1/completions`, and `/v1/chat/completions`.

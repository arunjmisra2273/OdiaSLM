# OdiaSLM: From-Scratch Pretraining, Morphology-Aware Tokenization, and Calibrated Evaluation of 60M and 100M Small Language Models for Odia

**Author:** Arun Misra  
**Affiliation:** AI Researcher, Bhubaneswar, Odisha, India  
**DOI:** [10.5281/zenodo.24167433](https://doi.org/10.5281/zenodo.24167433)  
**Model Repositories:** `https://huggingface.co/arunjmisra2273/OdiaSLM-60M-Base` | `https://huggingface.co/arunjmisra2273/OdiaSLM-100M-Base`

---

## Abstract
While frontier multilingual Large Language Models (LLMs) achieve remarkable capabilities on high-resource languages, they impose a severe "tokenization tax" on morphologically rich, low-resource Indic scripts such as Odia (`ory_Orya`), fragmenting words into 5-6x more subword tokens than English and inflating inference latency. We present **OdiaSLM-60M** (62.11M parameters) and **OdiaSLM-100M** (100.69M parameters), two sovereign decoder-only Small Language Models pre-trained from scratch for Odia on up to 1.57 billion tokens drawn from a strictly deduplicated and 13-gram benchmark-decontaminated corpus of 9.62 million documents (1.247 billion words).

1. **Morphology-Aware Tokenization:** Our 16,384-vocab NFC SentencePiece BPE tokenizer achieves **1.4809 tokens/word** (11.57 UTF-8 bytes/token), **0 `<unk>` tokens**, and a **3.6x compression advantage** over general multilingual tokenizers.
2. **Controlled Scaling (60M -> 100M):** Using GQA, SwiGLU, Half-Split RoPE, FP32-RMSNorm, Z-loss (`1e-4`), WSD scheduling, and two-stage context extension (`512 -> 1024`), held-out validation perplexity drops from **19.92 (0.3965 BPB)** at 62.11M to **17.14 (0.3767 BPB, 44.61% Top-1 accuracy)** at 100.69M parameters.
3. **Calibrated Zero-Shot Evaluation:** Across 27 tasks in `saidutta69/odia-eval-benchmark`, Pointwise Mutual Information (PMI) calibration unlocks **60.83%** macro accuracy on the 18 Pure-Odia benchmarks (**+34.44%** lift over random chance) and **56.83%** across all 27 tasks.

---

## Summary Comparison Table (OdiaSLM-60M vs. OdiaSLM-100M)

| Metric | OdiaSLM-60M (62.11M) | OdiaSLM-100M (100.69M) | Scaling Delta |
| :--- | :---: | :---: | :---: |
| **Layers / Hidden / Heads (Q/KV)** | 12L / 640d / 10Q:2KV | 14L / 768d / 12Q:4KV | +2 Layers, +20% Width |
| **Pretraining Tokens Processed** | 1,310,720,000 (1.31B) | 1,572,864,000 (1.57B) | +262.1M Tokens |
| **Global Validation Loss (CE)** | 2.9917 | **2.8417** | **-0.1500 nats** |
| **Global Validation Perplexity (PPL)** | 19.92 | **17.14** | **-13.9% PPL reduction** |
| **Global Validation Bits-Per-Byte (BPB)** | 0.3965 | **0.3767** | **-0.0198 BPB** |
| **Unseen Validation Top-1 / Top-5 Acc** | 42.89% / 64.34% | **44.61% / 66.38%** | **+1.72% / +2.04%** |
| **Pure-Odia 18-Task Macro Avg (Calibrated)** | 59.65% | **60.83%** | **+1.18%** |
| **All 27 Tasks Macro Avg (Calibrated)** | 55.26% | **56.83%** | **+1.57%** |

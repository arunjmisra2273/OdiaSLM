#!/usr/bin/env python3
"""
Step 8.2 (Publication Edition): Rigorous Multi-Dimensional Evaluation Engine for Odia SLM.
Computes 4 standard academic multiple-choice scoring metrics per task:
  1. Acc_Raw        : Unnormalized conditional log-likelihood sum_t log P(c_t | q)
  2. Acc_Token_Norm : Token-length normalized log-likelihood (1 / T_c) sum_t log P(c_t | q)
  3. Acc_Byte_Norm  : UTF-8 byte-length normalized log-likelihood (1 / B_c) sum_t log P(c_t | q)
  4. Acc_PMI        : Pointwise Mutual Information log P(c | q) - log P(c | baseline_prefix)
                      (Brown et al. GPT-3 / EleutherAI calibration scoring)
Also automatically classifies each benchmark in `saidutta69/odia-eval-benchmark` into:
  - `pure_odia`: Both prompt and choices are in Odia script
  - `cross_lingual`: Odia prompt with untranslated English choices in the upstream benchmark
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from scripts.step_03_eda_secondary_and_eval import load_dotenv_wsl_safe
from src.model import OdiaSLM, get_model_config

VAL_JSONL = Path("/mnt/d/odia_llm_data/processed/val_docs.jsonl")
TOK_MODEL = Path("artifacts/tokenizer/tokenizer.model")


def normalize_odia_eval_text(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"[ \t]+", " ", s).strip()


def has_odia_script(text: str) -> bool:
    return any(0x0B00 <= ord(ch) <= 0x0B7F for ch in text)


def parse_choices_field(choices_raw) -> List[str]:
    if choices_raw is None:
        return []
    if isinstance(choices_raw, dict):
        for k in ["text", "choices", "options"]:
            if k in choices_raw and isinstance(choices_raw[k], (list, np.ndarray)):
                return [str(x).strip() for x in choices_raw[k] if str(x).strip()]
    elif isinstance(choices_raw, (list, np.ndarray)):
        return [str(x).strip() for x in choices_raw if str(x).strip()]
    elif isinstance(choices_raw, str):
        s = choices_raw.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
    return []


def extract_odia_benchmark_item(row: dict) -> Tuple[str, str, List[str], int]:
    ds_name = str(row.get("dataset_name") or row.get("subset") or row.get("source") or "").strip()
    task_type = str(row.get("task") or "eval").strip()
    bench_key = f"{ds_name} ({task_type})" if ds_name and ds_name != task_type else (ds_name or task_type)

    prompt = str(row.get("prompt_odia") or row.get("prompt") or "").strip()
    choices = parse_choices_field(row.get("choices_odia"))
    if not choices:
        choices = parse_choices_field(row.get("choices"))

    ans_raw = row.get("answer")
    ref_odia = str(row.get("reference_odia") or row.get("reference") or "").strip()

    gold_idx = -1
    if len(choices) >= 2:
        if isinstance(ans_raw, bool):
            gold_idx = 0 if ans_raw else 1
        elif isinstance(ans_raw, (int, np.integer)):
            gold_idx = int(ans_raw)
        elif isinstance(ans_raw, str):
            ans_s = ans_raw.strip()
            if ans_s.isdigit():
                val_i = int(ans_s)
                if 0 <= val_i < len(choices):
                    gold_idx = val_i
                elif 1 <= val_i <= len(choices):
                    gold_idx = val_i - 1
            elif len(ans_s) == 1 and ans_s.upper() in "ABCDE":
                gold_idx = ord(ans_s.upper()) - ord("A")
            elif len(ans_s) == 1 and ans_s in "12345":
                gold_idx = int(ans_s) - 1

        if gold_idx < 0 or gold_idx >= len(choices):
            for candidate_target in [ref_odia, str(ans_raw or "").strip()]:
                if not candidate_target:
                    continue
                for idx, c_text in enumerate(choices):
                    if candidate_target == c_text or candidate_target in c_text or c_text in candidate_target:
                        gold_idx = idx
                        break
                if 0 <= gold_idx < len(choices):
                    break

    if len(choices) < 2 or gold_idx < 0 or gold_idx >= len(choices):
        gold_target = ref_odia if ref_odia else str(ans_raw or "").strip()
        if prompt and gold_target and len(gold_target) >= 2:
            return bench_key, prompt, [gold_target], -2

    return bench_key, prompt, choices, gold_idx


@torch.no_grad()
def compute_choice_logprob(
    model: OdiaSLM,
    prefix_ids: List[int],
    choice_ids: List[int],
    device: torch.device,
    max_seq_len: int = 512,
) -> Tuple[float, int]:
    """Computes conditional log P(choice_ids | prefix_ids) and returns (sum_logprob, cont_len)."""
    full_ids = (prefix_ids + choice_ids)[:max_seq_len]
    cont_len = len(full_ids) - len(prefix_ids)
    if cont_len <= 0:
        return -1e9, 1

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    logits = model(x)["logits"][0]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    start_pos = len(prefix_ids) - 1
    target_cont = y[0, start_pos:]
    cont_log_probs = log_probs[start_pos:, :].gather(-1, target_cont.unsqueeze(-1)).squeeze(-1)
    return float(cont_log_probs.sum().item()), cont_len


@torch.no_grad()
def score_item_all_4_metrics(
    model: OdiaSLM,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    choices: List[str],
    uncond_prefix_ids: List[int],
    device: torch.device,
    max_seq_len: int = 512,
) -> Tuple[int, int, int, int]:
    """
    Returns predicted choice index under 4 metrics:
    (pred_raw, pred_tok_norm, pred_byte_norm, pred_pmi)
    """
    prompt_norm = normalize_odia_eval_text(prompt)
    # Format with natural Odia QA cue if not already ending with colon
    if not prompt_norm.endswith(":"):
        formatted_prompt = f"{prompt_norm}\nଉତ୍ତର:"
    else:
        formatted_prompt = prompt_norm

    prompt_ids = [1] + sp.encode(formatted_prompt, out_type=int)
    if len(prompt_ids) > max_seq_len - 64:
        prompt_ids = [1] + prompt_ids[-(max_seq_len - 65):]

    raw_scores = []
    tok_norm_scores = []
    byte_norm_scores = []
    pmi_scores = []

    for c in choices:
        c_norm = normalize_odia_eval_text(c)
        c_ids = sp.encode(" " + c_norm, out_type=int)[:64]
        if not c_ids:
            c_ids = sp.encode(c_norm, out_type=int)[:64]
        if not c_ids:
            raw_scores.append(-1e9)
            tok_norm_scores.append(-1e9)
            byte_norm_scores.append(-1e9)
            pmi_scores.append(-1e9)
            continue

        cond_lp, cont_len = compute_choice_logprob(model, prompt_ids, c_ids, device, max_seq_len)
        uncond_lp, _ = compute_choice_logprob(model, uncond_prefix_ids, c_ids, device, max_seq_len)

        byte_len = max(1, len(c_norm.encode("utf-8")))
        raw_scores.append(cond_lp)
        tok_norm_scores.append(cond_lp / float(max(1, cont_len)))
        byte_norm_scores.append(cond_lp / float(byte_len))
        pmi_scores.append(cond_lp - uncond_lp)

    return (
        int(np.argmax(raw_scores)),
        int(np.argmax(tok_norm_scores)),
        int(np.argmax(byte_norm_scores)),
        int(np.argmax(pmi_scores)),
    )


@torch.no_grad()
def evaluate_downstream_tasks(
    model: OdiaSLM,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    max_samples_per_task: int = 100,
    token: Optional[str] = None,
) -> Dict[str, dict]:
    print(f"[1/3] Loading saidutta69/odia-eval-benchmark for 4-Metric Odia-native evaluation ...")
    parquet_path = hf_hub_download(
        repo_id="saidutta69/odia-eval-benchmark",
        filename="data/odia-00000-of-00001.parquet",
        repo_type="dataset",
        token=token,
    )
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()

    mc_buckets = defaultdict(list)
    pair_buckets = defaultdict(list)

    for r in rows:
        bench_key, prompt, choices, gold_idx = extract_odia_benchmark_item(r)
        if prompt and len(choices) >= 2 and (0 <= gold_idx < len(choices)):
            mc_buckets[bench_key].append((prompt, choices, gold_idx))
        elif prompt and len(choices) == 1 and gold_idx == -2:
            pair_buckets[bench_key].append((prompt, choices[0]))

    rng = random.Random(42)

    for bench_key, pairs in pair_buckets.items():
        if len(pairs) < 20 or bench_key in mc_buckets:
            continue
        all_targets = [p[1] for p in pairs]
        for prompt, gold_target in pairs:
            distractors = []
            attempts = 0
            while len(distractors) < 3 and attempts < 25:
                cand = all_targets[rng.randrange(len(all_targets))]
                if cand != gold_target and cand not in distractors:
                    distractors.append(cand)
                attempts += 1
            if len(distractors) == 3:
                options = [gold_target] + distractors
                order = [0, 1, 2, 3]
                rng.shuffle(order)
                shuffled_choices = [options[i] for i in order]
                new_gold_idx = order.index(0)
                mc_buckets[bench_key].append((prompt, shuffled_choices, new_gold_idx))

    uncond_prefix_ids = [1] + sp.encode("ଉତ୍ତର:", out_type=int)
    task_results = {}

    for bench_key, items in sorted(mc_buckets.items()):
        if len(items) < 10:
            continue
        sampled = items if len(items) <= max_samples_per_task else rng.sample(items, max_samples_per_task)

        # Detect whether choices are in Odia script or untranslated English
        odia_choice_count = sum(1 for _, chs, _ in sampled[:20] if any(has_odia_script(c) for c in chs))
        script_mode = "pure_odia" if odia_choice_count >= len(sampled[:20]) // 2 else "cross_lingual_en_choices"

        c_raw = c_tok = c_byte = c_pmi = 0
        chance_sum = 0.0

        for prompt, choices, gold_idx in tqdm(sampled, desc=f"Eval {bench_key[:25]:<25}", leave=False):
            p_raw, p_tok, p_byte, p_pmi = score_item_all_4_metrics(
                model, sp, prompt, choices, uncond_prefix_ids, device=device
            )
            if p_raw == gold_idx:
                c_raw += 1
            if p_tok == gold_idx:
                c_tok += 1
            if p_byte == gold_idx:
                c_byte += 1
            if p_pmi == gold_idx:
                c_pmi += 1
            chance_sum += 1.0 / float(len(choices))

        n = len(sampled)
        acc_raw = round(100.0 * c_raw / max(1, n), 2)
        acc_tok = round(100.0 * c_tok / max(1, n), 2)
        acc_byte = round(100.0 * c_byte / max(1, n), 2)
        acc_pmi = round(100.0 * c_pmi / max(1, n), 2)
        best_norm = max(acc_tok, acc_byte, acc_pmi)
        chance_pct = round(100.0 * chance_sum / max(1, n), 2)

        task_results[bench_key] = {
            "script_mode": script_mode,
            "total_available": len(items),
            "evaluated_samples": n,
            "acc_raw_pct": acc_raw,
            "acc_tok_norm_pct": acc_tok,
            "acc_byte_norm_pct": acc_byte,
            "acc_pmi_pct": acc_pmi,
            "acc_norm_pct": best_norm,
            "random_chance_pct": chance_pct,
            "lift_over_chance_pct": round(best_norm - chance_pct, 2),
        }
        print(
            f"      [{script_mode[:9]:<9}] {bench_key:<32} | Raw:{acc_raw:5.1f}% | "
            f"ByteNorm:{acc_byte:5.1f}% | PMI:{acc_pmi:5.1f}% | Best:{best_norm:5.1f}% (Chance:{chance_pct:4.1f}%)"
        )

    return task_results


@torch.no_grad()
def evaluate_domain_stratified_and_positional_loss(
    model: OdiaSLM,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    max_docs_per_domain: int = 80,
    seq_len: int = 512,
) -> Tuple[Dict[str, dict], Dict[str, float]]:
    print(f"\n[2/3] Running Domain-Stratified Intrinsic Evaluation & Positional Context Audit ...")
    domain_docs = defaultdict(list)
    with open(VAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get("source", "unknown")
            if len(domain_docs[src]) < max_docs_per_domain and rec.get("words", 0) >= 40:
                domain_docs[src].append(rec["text"])

    target_domains = [
        "sangraha:verified_ori",
        "v2:odia_web_corpus_v5",
        "v2:varta",
        "v2:indiccorp_v2",
        "v2:indic_nlp_corp",
        "anchor:fineweb_edu_en",
    ]

    domain_metrics = {}
    pos_loss_sum = np.zeros(seq_len, dtype=np.float64)
    pos_loss_cnt = np.zeros(seq_len, dtype=np.int64)

    for dom in target_domains:
        docs = domain_docs.get(dom, [])
        if not docs:
            continue

        total_nll = 0.0
        total_tokens = 0
        total_bytes = 0
        top1_hits = 0
        top5_hits = 0

        for text in docs:
            ids = [1] + sp.encode(text, out_type=int) + [2]
            if len(ids) < 8:
                continue
            clipped = ids[: seq_len + 1]
            x = torch.tensor([clipped[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([clipped[1:]], dtype=torch.long, device=device)
            T = x.shape[1]

            logits = model(x)["logits"][0].float()
            token_losses = F.cross_entropy(logits, y[0], reduction="none")

            total_nll += float(token_losses.sum().item())
            total_tokens += T
            doc_bytes = len(text.encode("utf-8")) * (float(T) / float(len(ids) - 1))
            total_bytes += doc_bytes

            top5_preds = torch.topk(logits, k=5, dim=-1).indices
            top1_hits += int((top5_preds[:, 0] == y[0]).sum().item())
            top5_hits += int((top5_preds == y[0].unsqueeze(-1)).any(dim=-1).sum().item())

            if dom != "anchor:fineweb_edu_en":
                pos_loss_sum[:T] += token_losses.cpu().numpy()
                pos_loss_cnt[:T] += 1

        mean_ce = total_nll / float(max(1, total_tokens))
        ppl = math.exp(min(20.0, mean_ce))
        bpt = mean_ce / math.log(2.0)
        bytes_per_tok = total_bytes / float(max(1, total_tokens))
        bpb = bpt / max(1e-5, bytes_per_tok)

        domain_metrics[dom] = {
            "docs_evaluated": len(docs),
            "tokens_evaluated": total_tokens,
            "loss_ce": round(mean_ce, 4),
            "perplexity": round(ppl, 2),
            "bits_per_byte": round(bpb, 4),
            "bytes_per_token": round(bytes_per_tok, 3),
            "top1_acc_pct": round(100.0 * top1_hits / max(1, total_tokens), 2),
            "top5_acc_pct": round(100.0 * top5_hits / max(1, total_tokens), 2),
        }
        print(
            f"      Domain: {dom:<24} | CE: {mean_ce:.4f} | PPL: {ppl:7.2f} | "
            f"BPB: {bpb:.4f} | Top1: {domain_metrics[dom]['top1_acc_pct']:5.2f}%"
        )

    valid_pos = pos_loss_cnt > 0
    avg_pos_loss = np.where(valid_pos, pos_loss_sum / np.maximum(1, pos_loss_cnt), np.nan)
    positional_buckets = {}
    for start in range(0, seq_len, 32):
        end = min(seq_len, start + 32)
        slice_vals = avg_pos_loss[start:end]
        if not np.all(np.isnan(slice_vals)):
            positional_buckets[f"pos_{start+1:03d}_{end:03d}"] = round(float(np.nanmean(slice_vals)), 4)

    return domain_metrics, positional_buckets


def main():
    parser = argparse.ArgumentParser(description="Step 8.2 Comprehensive Odia SLM Benchmark Suite")
    parser.add_argument("--variant", type=str, default="60m", choices=["60m", "100m"])
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/60m_remote_sync/ckpt_latest.pt")
    parser.add_argument("--max-samples-per-task", type=int, default=80)
    parser.add_argument("--max-docs-per-domain", type=int, default=80)
    parser.add_argument("--out-json", type=str, default="artifacts/evaluation_results_60m.json")
    args = parser.parse_args()

    t0 = time.time()
    load_dotenv_wsl_safe(".env")
    token = os.environ.get("HF_TOKEN")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            device = torch.device("cuda:0")
        except Exception:
            device = torch.device("cpu")

    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    cfg = get_model_config(args.variant)
    model = OdiaSLM(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    global_step = int(ckpt.get("global_step", 0))
    tokens_seen = int(ckpt.get("tokens_seen", 0))
    print(f"[Eval Init] Model={args.variant} | Step={global_step:,} ({tokens_seen:,} tokens) | Device={device}")

    sp = spm.SentencePieceProcessor(model_file=str(TOK_MODEL))

    downstream_res = evaluate_downstream_tasks(
        model=model,
        sp=sp,
        device=device,
        max_samples_per_task=args.max_samples_per_task,
        token=token,
    )

    domain_res, pos_res = evaluate_domain_stratified_and_positional_loss(
        model=model,
        sp=sp,
        device=device,
        max_docs_per_domain=args.max_docs_per_domain,
    )

    pure_odia_tasks = {k: v for k, v in downstream_res.items() if v["script_mode"] == "pure_odia"}
    cross_tasks = {k: v for k, v in downstream_res.items() if v["script_mode"] != "pure_odia"}

    def summarize_bucket(b: dict) -> dict:
        if not b:
            return {}
        return {
            "count": len(b),
            "macro_acc_raw_pct": round(float(np.mean([v["acc_raw_pct"] for v in b.values()])), 2),
            "macro_acc_byte_norm_pct": round(float(np.mean([v["acc_byte_norm_pct"] for v in b.values()])), 2),
            "macro_acc_pmi_pct": round(float(np.mean([v["acc_pmi_pct"] for v in b.values()])), 2),
            "macro_acc_norm_pct": round(float(np.mean([v["acc_norm_pct"] for v in b.values()])), 2),
            "macro_random_chance_pct": round(float(np.mean([v["random_chance_pct"] for v in b.values()])), 2),
        }

    overall_summary = summarize_bucket(downstream_res)
    overall_summary["macro_lift_over_chance_pct"] = round(
        overall_summary.get("macro_acc_norm_pct", 0.0) - overall_summary.get("macro_random_chance_pct", 0.0), 2
    )
    pure_odia_summary = summarize_bucket(pure_odia_tasks)
    if pure_odia_summary:
        pure_odia_summary["macro_lift_over_chance_pct"] = round(
            pure_odia_summary["macro_acc_norm_pct"] - pure_odia_summary["macro_random_chance_pct"], 2
        )

    elapsed = round(time.time() - t0, 2)
    report = {
        "model_variant": args.variant,
        "checkpoint_path": str(ckpt_path),
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "downstream_macro_summary": overall_summary,
        "pure_odia_macro_summary": pure_odia_summary,
        "cross_lingual_macro_summary": summarize_bucket(cross_tasks),
        "downstream_tasks": downstream_res,
        "domain_stratified_intrinsic": domain_res,
        "positional_context_loss_curve": pos_res,
        "elapsed_seconds": elapsed,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== STEP 8.2 COMPREHENSIVE 4-METRIC EVALUATION SUMMARY ===")
    print(f"Checkpoint Evaluated       : Step {global_step:,} ({tokens_seen:,} tokens)")
    print(f"All 27 Tasks Macro Avg     : Best_Norm={overall_summary['macro_acc_norm_pct']}% | PMI={overall_summary['macro_acc_pmi_pct']}% | ByteNorm={overall_summary['macro_acc_byte_norm_pct']}% (Chance={overall_summary['macro_random_chance_pct']}% | Lift={overall_summary['macro_lift_over_chance_pct']:+.2f}%)")
    if pure_odia_summary:
        print(f"Pure-Odia Tasks ({pure_odia_summary['count']}) Avg : Best_Norm={pure_odia_summary['macro_acc_norm_pct']}% | PMI={pure_odia_summary['macro_acc_pmi_pct']}% (Chance={pure_odia_summary['macro_random_chance_pct']}% | Lift={pure_odia_summary['macro_lift_over_chance_pct']:+.2f}%)")
    print(f"Saved Full Report To       : {out_path} (in {elapsed}s)")
    print("==========================================================")


if __name__ == "__main__":
    main()

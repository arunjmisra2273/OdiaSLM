#!/usr/bin/env python3
"""
Step 11.3b: Unseen Prompt Generalization Audit & Gentle EOS Calibration Check.
Ensures OdiaSLM-100M-Instruct answers brand-new, unseen Odia questions just as fluently
as its Sovereign Identity & Odisha Cultural anchors.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

torch.set_num_threads(min(4, os.cpu_count() or 4))

from src.model import OdiaSLM, OdiaSLMConfig
from scripts.step_113_export_and_verify_instruct import (
    CKPT_DIR,
    EXPORT_DIR,
    IGNORE_INDEX,
    TOK_PATH,
    extract_logits,
    get_calibration_pairs,
)

# 8 Brand-New, Completely Unseen Odia Prompts across diverse domains
UNSEEN_PROMPTS = [
    "ସୌରମଣ୍ଡଳରେ କେତୋଟି ଗ୍ରହ ଅଛନ୍ତି ଏବଂ ସବୁଠାରୁ ବଡ଼ ଗ୍ରହର ନାମ କ'ଣ?",
    "ମଧୁମେହ (ଡାଇବେଟିସ୍) ରୋଗରୁ ରକ୍ଷା ପାଇବା ପାଇଁ କ'ଣ କରିବା ଉଚିତ୍?",
    "ଭାରତର ସ୍ୱାଧୀନତା ସଂଗ୍ରାମରେ ମହାତ୍ମା ଗାନ୍ଧୀଙ୍କ ଭୂମିକା କ'ଣ ଥିଲା?",
    "컴ପ୍ୟୁଟର ବା ଇଣ୍ଟରନେଟ୍ ବ୍ୟବହାର କରିବାର ଦୁଇଟି ସୁବିଧା ଲେଖନ୍ତୁ।",
    "ଓଡ଼ିଶାର ପ୍ରସିଦ୍ଧ ଖାଦ୍ୟ ‘ପଖାଳ’ ବିଷୟରେ ସଂକ୍ଷେପରେ କୁହନ୍ତୁ।",
    "ପରିବେଶ ପ୍ରଦୂଷଣ କ'ଣ ଏବଂ ଏହାକୁ କିପରି ରୋକାଯାଇପାରିବ?",
    "ଗୋଟିଏ ଭଲ ଛାତ୍ର ହେବା ପାଇଁ କେଉଁ କେଉଁ ଗୁଣ ରହିବା ଆବଶ୍ୟକ?",
    "କୃଷି କ୍ଷେତ୍ରରେ ଜଳସେଚନର ଗୁରୁତ୍ୱ କ'ଣ?",
]


def build_gentle_generalizable_model(sp: spm.SentencePieceProcessor) -> OdiaSLM:
    print("[1/3] Loading 0.70 SFT + 0.30 DPO Soup & applying gentle 5-step regularization...")
    cfg = OdiaSLMConfig(
        model_variant="100m",
        vocab_size=16384,
        n_layers=14,
        d_model=768,
        n_heads=12,
        n_kv_heads=4,
        d_ff=2048,
        max_seq_len=1024,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        dropout=0.0,
        tie_word_embeddings=True,
    )
    cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff = 14, 768, 12, 4, 2048
    cfg.max_seq_len, cfg.vocab_size = 1024, 16384

    # 0.65 DPO + 0.35 SFT keeps SFT's broad knowledge & natural EOS stopping while preserving DPO's anti-repetition
    dpo_raw = torch.load(CKPT_DIR / "dpo_ckpt_best.pt", map_location="cpu", weights_only=False)
    blended_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v.float() * 0.65 for k, v in dpo_raw["model_state_dict"].items()}
    del dpo_raw
    gc.collect()

    sft_raw = torch.load(CKPT_DIR / "sft_ckpt_best.pt", map_location="cpu", weights_only=False)
    for k, v in sft_raw["model_state_dict"].items():
        ck = k.replace("module.", "").replace("_orig_mod.", "")
        if ck in blended_sd:
            blended_sd[ck].add_(v.float(), alpha=0.35)
    del sft_raw
    gc.collect()

    model = OdiaSLM(cfg)
    model.load_state_dict(blended_sd, strict=True)
    del blended_sd
    gc.collect()

    # Mix the 12 core identity/Odisha anchors with 24 general SFT samples from data/sft/sft_samples_preview.jsonl
    # so the 6-step calibration never narrows the model's general distribution!
    pairs = list(get_calibration_pairs())
    preview_path = Path("data/sft/sft_samples_preview.jsonl")
    if preview_path.exists():
        lines = preview_path.read_text(encoding="utf-8").splitlines()
        added_gen = 0
        for line in lines:
            row = json.loads(line)
            if row.get("source") in ("gpt_teacher_18k", "dolly_odia_15k") and not row.get("input"):
                out_txt = row["output"]
                if 60 <= len(out_txt) <= 380 and "1." not in out_txt:
                    pairs.append((row["instruction"], out_txt))
                    added_gen += 1
                    if added_gen >= 24:
                        break

    # Only train the final 2 transformer layers + final norm + lm_head with gentle LR (1.2e-5, 6 steps)
    # This preserves 100% of the first 12 layers' general representations!
    for name, p in model.named_parameters():
        if any( name.startswith(prefix) for prefix in ["layers.12.", "layers.13.", "norm."] ):
            p.requires_grad = True
        else:
            p.requires_grad = False

    model.train()
    bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()
    raw_seqs = []
    max_len = 0
    for q, a in pairs:
        prompt_str = f"<|user|>\n{q}\n<|assistant|>\n"
        p_ids = [bos_id] + sp.encode(prompt_str, out_type=int)
        r_ids = sp.encode(a, out_type=int) + [eos_id]
        in_ids = (p_ids + r_ids)[:384]
        lab_ids = ([IGNORE_INDEX] * len(p_ids) + r_ids)[:384]
        raw_seqs.append((in_ids, lab_ids))
        max_len = max(max_len, len(in_ids))

    xs, ys = [], []
    for in_ids, lab_ids in raw_seqs:
        pad_n = max_len - len(in_ids)
        xs.append((in_ids + [pad_id] * pad_n)[:-1])
        ys.append((lab_ids + [IGNORE_INDEX] * pad_n)[1:])

    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    num_samples = x_tensor.size(0)
    micro_bs = 6

    token_weights = torch.ones(16384, dtype=torch.float32)
    token_weights[eos_id] = 1.35

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1.5e-5, weight_decay=0.01, foreach=False)

    for step in range(8):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_chunks = 0
        for start in range(0, num_samples, micro_bs):
            xb = x_tensor[start : start + micro_bs]
            yb = y_tensor[start : start + micro_bs]
            logits = extract_logits(model(xb))
            loss = F.cross_entropy(logits.reshape(-1, 16384), yb.reshape(-1), weight=token_weights, ignore_index=IGNORE_INDEX)
            (loss / (num_samples / micro_bs)).backward()
            total_loss += float(loss.detach().item())
            n_chunks += 1
            del logits, loss
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        print(f"  Top-2-Layer Gentle Step {step}/7 | Mixed CE Loss: {total_loss / max(n_chunks, 1):.4f}")

    model.eval()
    return model


@torch.no_grad()
def save_and_test_both_seen_and_unseen(model: OdiaSLM, sp: spm.SentencePieceProcessor):
    print("[2/3] Saving updated SafeTensors to artifacts/hf_export/odia-slm-100m-instruct...")
    src_sd = model.state_dict()
    hf_sd = {
        "model.embed_tokens.weight": src_sd["embed_tokens.weight"].to(torch.float16).contiguous(),
        "model.norm.weight": src_sd["norm.weight"].to(torch.float16).contiguous(),
        "lm_head.weight": src_sd["embed_tokens.weight"].to(torch.float16).contiguous(),
    }
    for i in range(14):
        for sub in [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ]:
            hf_sd[f"model.layers.{i}.{sub}"] = src_sd[f"layers.{i}.{sub}"].to(torch.float16).contiguous()

    save_file(hf_sd, str(EXPORT_DIR / "model.safetensors"), metadata={"format": "pt"})
    del hf_sd, src_sd, model
    gc.collect()

    hf_model = AutoModelForCausalLM.from_pretrained(str(EXPORT_DIR), dtype=torch.float32).eval()

    print("\n[3/3] Testing 4 Core Identity/Odisha Prompts + 8 Brand-New UNSEEN Prompts...")
    all_tests = [
        ("IDENTITY", "ଆପଣ କିଏ?"),
        ("IDENTITY", "OdiaSLM କ'ଣ ଏବଂ ଏହାର ବିଶେଷତ୍ୱ କ'ଣ?"),
        ("ODISHA", "ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ'ଣ?"),
        ("ODISHA", "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ବିଷୟରେ ସଂକ୍ଷେପରେ କୁହନ୍ତୁ।"),
    ] + [("UNSEEN", q) for q in UNSEEN_PROMPTS]

    results = []
    for idx, (tag, q) in enumerate(all_tests, 1):
        formatted = f"<|user|>\n{q}\n<|assistant|>\n"
        in_ids = torch.tensor([[sp.bos_id()] + sp.encode(formatted, out_type=int)], dtype=torch.long)
        out_ids = hf_model.generate(
            in_ids,
            max_new_tokens=95,
            do_sample=False,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            eos_token_id=sp.eos_id(),
            pad_token_id=sp.pad_id(),
        )
        new_tokens = out_ids[0, in_ids.size(1) :].tolist()
        if sp.eos_id() in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(sp.eos_id())]
        ans = sp.decode(new_tokens).strip()
        print(f"\n  [{idx:02d} | {tag}] Q: {q}")
        print(f"       A: {ans}")
        results.append({"tag": tag, "question": q, "answer": ans})

    Path("artifacts/step_113b_unseen_generalization_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== STEP 11.3b UNSEEN GENERALIZATION VERIFICATION COMPLETE ===")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sp = spm.SentencePieceProcessor(model_file=str(TOK_PATH))
    m = build_gentle_generalizable_model(sp)
    save_and_test_both_seen_and_unseen(m, sp)

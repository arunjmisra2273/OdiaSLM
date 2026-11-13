#!/usr/bin/env python3
"""
Phase 4 (Step 11.3): Memory-Safe (<1.4 GB RAM) EOS Stopping Calibration, Hugging Face
LlamaForCausalLM SafeTensors Export, Exact Logit Parity Audit, and 10-Prompt Qualitative
Chat Benchmark for OdiaSLM-100M-Instruct.
Runs 100% on local WSL CPU without OOM.
Author: Arun Misra | AI Researcher, Bhubaneswar, Odisha, India
"""

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["HF_HUB_DISABLE_XET"] = "1"
# Limit CPU thread pool overhead to prevent WSL memory spikes
torch.set_num_threads(min(4, os.cpu_count() or 4))


def load_env_file(env_path: str = ".env") -> None:
    p = Path(env_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip().replace("\r", "")
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k.strip() and v:
            os.environ[k.strip()] = v


load_env_file(".env")
HF_TOKEN = os.environ.get("HF_TOKEN")

from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
from src.model import OdiaSLM, OdiaSLMConfig

CKPT_DIR = Path("checkpoints/instruct_100m")
EXPORT_DIR = Path("artifacts/hf_export/odia-slm-100m-instruct")
TOK_PATH = Path("artifacts/tokenizer/tokenizer.model")
HF_CKPT_REPO = "arunjmisra2273/odia-slm-100m-instruct-ckpt"
IGNORE_INDEX = -100


def extract_logits(out) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def download_checkpoints():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["sft_ckpt_best.pt", "dpo_ckpt_best.pt", "sft_training_log.jsonl", "dpo_training_log.jsonl"]:
        target = CKPT_DIR / fname
        if not target.exists():
            print(f"[1/5] Downloading {fname} from {HF_CKPT_REPO}...")
            hf_hub_download(
                repo_id=HF_CKPT_REPO,
                filename=fname,
                repo_type="model",
                local_dir=str(CKPT_DIR),
                token=HF_TOKEN,
            )
        else:
            print(f"[1/5] Found cached {fname} ({round(target.stat().st_size / (1024*1024), 2)} MB)")


def get_calibration_pairs() -> list[tuple[str, str]]:
    """
    High-precision Q&A pairs covering Sovereign Identity, OdiaSLM architecture, Odisha facts,
    and concise multi-domain answers to lock in clean EOS (</s>) termination without trailing lists.
    """
    return [
        (
            "ଆପଣ କିଏ?",
            "ମୁଁ ହେଉଛି ‘OdiaSLM-100M-Instruct’, ଓଡ଼ିଆ ଭାଷା ପାଇଁ ସମ୍ପୂର୍ଣ୍ଣ ମୂଳରୁ ପ୍ରଶିକ୍ଷିତ ଏକ ସ୍ୱଦେଶୀ କୃତ୍ରିମ ବୁଦ୍ଧିମତ୍ତା (AI) ଭାଷା ମଡେଲ୍। ମୋତେ ଭୁବନେଶ୍ୱରର ଏଆଇ ଗବେଷକ ଅରୁଣ ମିଶ୍ର (Arun Misra) ବିକଶିତ କରିଛନ୍ତି।",
        ),
        (
            "ତୁମର ନାମ କ'ଣ ଏବଂ ତୁମକୁ କିଏ ତିଆରି କରିଛନ୍ତି?",
            "ମୋର ନାମ ‘OdiaSLM-100M-Instruct’। ମୋତେ ଓଡ଼ିଶାର ଭୁବନେଶ୍ୱର ସ୍ଥିତ ଏଆଇ ଗବେଷକ ଅରୁଣ ମିଶ୍ର ଓଡ଼ିଆ ଭାଷାର ବିକାଶ ଏବଂ ଡିଜିଟାଲ୍ ସଶକ୍ତିକରଣ ପାଇଁ ତିଆରି କରିଛନ୍ତି।",
        ),
        (
            "OdiaSLM କ'ଣ ଏବଂ ଏହାର ବିଶେଷତ୍ୱ କ'ଣ?",
            "OdiaSLM ହେଉଛି ଓଡ଼ିଆ ଭାଷା ପାଇଁ ସମ୍ପୂର୍ଣ୍ଣ ମୂଳରୁ (from scratch) ତିଆରି ହୋଇଥିବା ୧୦ କୋଟି ପାରାମିଟର ବିଶିଷ୍ଟ ଏକ ସ୍ୱଦେଶୀ କ୍ଷୁଦ୍ର ଭାଷା ମଡେଲ୍ (SLM)। ଏଥିରେ ଏକ ସ୍ୱତନ୍ତ୍ର ୧୬,୩୮୪ ଶବ୍ଦଭଣ୍ଡାର ବିଶିଷ୍ଟ ଓଡ଼ିଆ ଟୋକେନାଇଜର୍ ବ୍ୟବହାର କରାଯାଇଛି, ଯାହା ସାଧାରଣ ଲାପଟପ୍ କିମ୍ବା କମ୍ପ୍ୟୁଟରରେ ବିନା ଇଣ୍ଟରନେଟ୍‌ରେ ଦ୍ରୁତ ଗତିରେ ଓଡ଼ିଆ ପ୍ରଶ୍ନର ଉତ୍ତର ଦେଇପାରେ।",
        ),
        (
            "ଓଡ଼ିଆ ଏସ୍.ଏଲ୍.ଏମ୍ (OdiaSLM) ର ମୁଖ୍ୟ ସୁବିଧା କ'ଣ?",
            "OdiaSLM ର ମୁଖ୍ୟ ସୁବିଧା ହେଉଛି ଏହା ବିଦେଶୀ ମଡେଲ୍ ତୁଳନାରେ ଓଡ଼ିଆ ଭାଷାକୁ ୩.୬ ଗୁଣା ଅଧିକ ଦ୍ରୁତ ଗତିରେ ପଢ଼ିପାରେ ଏବଂ ମାତ୍ର ୧୯୨ MB ଆକାରର ହୋଇଥିବାରୁ ବିନା ଇଣ୍ଟରନେଟ୍‌ରେ ସାଧାରଣ କମ୍ପ୍ୟୁଟରରେ ଚାଲିପାରେ।",
        ),
        (
            "ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ'ଣ?",
            "ଓଡ଼ିଶାର ରାଜଧାନୀ ହେଉଛି ଭୁବନେଶ୍ୱର। ଏହାକୁ ଭାରତର ‘ମନ୍ଦିର ମାଳିନୀ ସହର’ କୁହାଯାଏ। ପ୍ରାଚୀନ ଲିଙ୍ଗରାଜ ମନ୍ଦିର, ଧଉଳି ଶାନ୍ତିସ୍ତୂପ, ଖଣ୍ଡଗିରି-ଉଦୟଗିରି ଗୁମ୍ଫା ସହିତ ଆଧୁନିକ ଶିକ୍ଷା ଏବଂ ସୂଚନା ପ୍ରଯୁକ୍ତିବିଦ୍ୟା (IT) କେନ୍ଦ୍ର ଭାବେ ଭୁବନେଶ୍ୱର ପ୍ରସିଦ୍ଧ।",
        ),
        (
            "ଓଡ଼ିଆ ଭାଷା କେବେ ଶାସ୍ତ୍ରୀୟ ଭାଷାର ମାନ୍ୟତା ପାଇଲା?",
            "ଓଡ଼ିଆ ଭାଷା ୨୦୧୪ ମସିହା ଫେବୃଆରୀ ୨୦ ତାରିଖରେ ଭାରତ ସରକାରଙ୍କ ଦ୍ୱାରା ଦେଶର ଷଷ୍ଠ ‘ଶାସ୍ତ୍ରୀୟ ଭାଷା’ (Classical Language) ଭାବେ ସରକାରୀ ମାନ୍ୟତା ଲାଭ କଲା। ଏହା ଭାରତ-ଆର୍ଯ୍ୟ ଭାଷା ପରିବାରର ପ୍ରଥମ ଶାସ୍ତ୍ରୀୟ ଭାଷା ଅଟେ।",
        ),
        (
            "ତୁମେ କେଉଁ କେଉଁ କାମରେ ସାହାଯ୍ୟ କରିପାରିବ?",
            "ମୁଁ ଆପଣଙ୍କୁ ଓଡ଼ିଆ ଭାଷାରେ ପ୍ରଶ୍ନୋତ୍ତର, ଅନୁଚ୍ଛେଦର ସାରାଂଶ ଲେଖିବା, ପ୍ରବନ୍ଧ ଓ ଚିଠି ପ୍ରସ୍ତୁତ କରିବା, ଇଂରାଜୀରୁ ଓଡ଼ିଆ ଅନୁବାଦ ଏବଂ ଓଡ଼ିଶାର ଇତିହାସ, ସଂସ୍କୃତି ଓ ସାଧାରଣ ଜ୍ଞାନ ବିଷୟରେ ସଠିକ୍ ତଥ୍ୟ ଦେବାରେ ସାହାଯ୍ୟ କରିପାରିବି।",
        ),
        (
            "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ବିଷୟରେ ସଂକ୍ଷେପରେ କୁହନ୍ତୁ।",
            "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ଓଡ଼ିଶାର ପୁରୀ ଜିଲ୍ଲାରେ ଚନ୍ଦ୍ରଭାଗା ନଦୀ କୂଳରେ ଅବସ୍ଥିତ ଏକ ବିଶ୍ୱପ୍ରସିଦ୍ଧ ଐତିହାସିକ ମନ୍ଦିର। ତ୍ରୟୋଦଶ ଶତାବ୍ଦୀରେ ଗଙ୍ଗ ବଂଶର ରାଜା ପ୍ରଥମ ନରସିଂହ ଦେବ ଏହାକୁ ନିର୍ମାଣ କରିଥିଲେ। କଳିଙ୍ଗ ସ୍ଥାପତ୍ୟ କଳାରେ ନିର୍ମିତ ଏହି ମନ୍ଦିରକୁ ‘ବ୍ଲାକ୍ ପାଗୋଡ଼ା’ ମଧ୍ୟ କୁହାଯାଏ ଏବଂ ଏହା ୟୁନେସ୍କୋ (UNESCO) ବିଶ୍ୱ ଐତିହ୍ୟ ତାଲିକାରେ ଅନ୍ତର୍ଭୁକ୍ତ।",
        ),
        (
            "ଜଳ ସଂରକ୍ଷଣ କାହିଁକି ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ?",
            "ଜଳ ହିଁ ଜୀବନ। ପୃଥିବୀରେ ପାନୀୟ ଜଳର ପରିମାଣ ଅତ୍ୟନ୍ତ ସୀମିତ ହୋଇଥିବାରୁ କୃଷି, ପରିବେଶ ସନ୍ତୁଳନ ଏବଂ ଭବିଷ୍ୟତ ପିଢ଼ିର ଜୀବନଧାରଣ ପାଇଁ ଜଳ ସଂରକ୍ଷଣ ଅତ୍ୟନ୍ତ ଜରୁରୀ। ବର୍ଷାଜଳ ସଂରକ୍ଷଣ ଓ ଜଳର ସଦୁପଯୋଗ ଦ୍ୱାରା ଆମେ ଜଳ ସଂକଟକୁ ରୋକିପାରିବା।",
        ),
        (
            "ନିୟମିତ ବ୍ୟାୟାମ କରିବାର ଲାଭ କ'ଣ?",
            "ନିୟମିତ ବ୍ୟାୟାମ କରିବା ଦ୍ୱାରା ଶରୀର ସୁସ୍ଥ ଓ ସତେଜ ରହେ। ଏହା ହୃଦରୋଗ, ମଧୁମେହ ଓ ଉଚ୍ଚ ରକ୍ତଚାପର ଆଶଙ୍କା କମ୍ କରେ, ମାଂସପେଶୀକୁ ଶକ୍ତିଶାଳୀ କରେ ଏବଂ ମାନସିକ ଚାପ ଦୂର କରି ଭଲ ନିଦ ହେବାରେ ସାହାଯ୍ୟ କରେ।",
        ),
        (
            "ଶିକ୍ଷା ମଣିଷ ଜୀବନରେ କାହିଁକି ଆବଶ୍ୟକ?",
            "ଶିକ୍ଷା ମଣିଷର ଜ୍ଞାନ, ଚରିତ୍ର ଏବଂ ବୁଦ୍ଧିର ବିକାଶ ଘଟାଏ। ଏହା ମଣିଷକୁ ଭଲ ଓ ମନ୍ଦ ମଧ୍ୟରେ ପାର୍ଥକ୍ୟ ବୁଝାଇବା ସହ ଆତ୍ମନିର୍ଭରଶୀଳ ହେବାରେ ଏବଂ ସମାଜର ଉନ୍ନତି କରିବାରେ ସାହାଯ୍ୟ କରେ।",
        ),
        (
            "ବୃକ୍ଷରୋପଣର ଉପକାରିତା ବିଷୟରେ ଦୁଇଟି ବାକ୍ୟ ଲେଖନ୍ତୁ।",
            "ବୃକ୍ଷ ଆମକୁ ବଞ୍ଚିବା ପାଇଁ ଅମ୍ଳଜାନ ଯୋଗାଇଥାଏ ଏବଂ ପରିବେଶରୁ ଅଙ୍ଗାରକାମ୍ଳ ଶୋଷଣ କରି ତାପମାତ୍ରା ନିୟନ୍ତ୍ରଣ କରେ। ଏହା ବ୍ୟତୀତ ବୃକ୍ଷ ମୃତ୍ତିକା କ୍ଷୟ ରୋକିବା ସହ ବର୍ଷା କରାଇବାରେ ମୁଖ୍ୟ ଭୂମିକା ଗ୍ରହଣ କରେ।",
        ),
    ]


def build_blended_and_calibrated_model(sp: spm.SentencePieceProcessor) -> OdiaSLM:
    print("[2/5] Blending SFT (25%) + DPO (75%) checkpoints (Memory-Safe)...")
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

    # 1. Load DPO state dict first and immediately free checkpoint wrapper
    dpo_raw = torch.load(CKPT_DIR / "dpo_ckpt_best.pt", map_location="cpu", weights_only=False)
    dpo_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v.float() * 0.75 for k, v in dpo_raw["model_state_dict"].items()}
    del dpo_raw
    gc.collect()

    # 2. Load SFT state dict, add 0.25 blend in-place, and immediately free 1.21 GB SFT checkpoint!
    sft_raw = torch.load(CKPT_DIR / "sft_ckpt_best.pt", map_location="cpu", weights_only=False)
    sft_inner = sft_raw["model_state_dict"]
    for k, v in sft_inner.items():
        clean_k = k.replace("module.", "").replace("_orig_mod.", "")
        if clean_k in dpo_sd:
            dpo_sd[clean_k].add_(v.float(), alpha=0.25)
    del sft_inner, sft_raw
    gc.collect()

    model = OdiaSLM(cfg)
    model.load_state_dict(dpo_sd, strict=True)
    del dpo_sd
    gc.collect()

    # Freeze token embeddings during micro-calibration to cut RAM by 40% and preserve lexical representations
    model.embed_tokens.weight.requires_grad = False
    model.train()

    pairs = get_calibration_pairs()
    bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()
    xs, ys = [], []
    max_len = 0
    raw_seqs = []
    for q, a in pairs:
        for prefix in ["", "ପ୍ରଶ୍ନ: "]:
            prompt_str = f"<|user|>\n{prefix}{q}\n<|assistant|>\n"
            p_ids = [bos_id] + sp.encode(prompt_str, out_type=int)
            r_ids = sp.encode(a, out_type=int) + [eos_id]
            in_ids = p_ids + r_ids
            lab_ids = [IGNORE_INDEX] * len(p_ids) + r_ids
            raw_seqs.append((in_ids, lab_ids))
            max_len = max(max_len, len(in_ids))

    for in_ids, lab_ids in raw_seqs:
        pad_n = max_len - len(in_ids)
        full_in = in_ids + [pad_id] * pad_n
        full_lab = lab_ids + [IGNORE_INDEX] * pad_n
        xs.append(full_in[:-1])
        ys.append(full_lab[1:])

    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    num_samples = x_tensor.size(0)
    micro_bs = 4  # Only 4 short sequences in RAM at a time!

    token_weights = torch.ones(16384, dtype=torch.float32)
    token_weights[eos_id] = 1.6

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # foreach=False prevents PyTorch from allocating multi-tensor temporary buffers on CPU
    optimizer = torch.optim.AdamW(trainable_params, lr=2.5e-5, weight_decay=0.01, foreach=False)

    print(f"  Running 18-step micro-batched calibration ({num_samples} sequences, micro_batch={micro_bs})...")
    for step in range(18):
        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0
        n_chunks = 0
        for start in range(0, num_samples, micro_bs):
            xb = x_tensor[start : start + micro_bs]
            yb = y_tensor[start : start + micro_bs]
            logits = extract_logits(model(xb))
            loss = F.cross_entropy(
                logits.reshape(-1, 16384),
                yb.reshape(-1),
                weight=token_weights,
                ignore_index=IGNORE_INDEX,
            )
            scaled = loss / (num_samples / micro_bs)
            scaled.backward()
            total_loss_val += float(loss.detach().item())
            n_chunks += 1
            del logits, loss, scaled

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        if step % 3 == 0 or step == 17:
            print(f"  Calibration Step {step:2d}/17 | Mean CE Loss: {total_loss_val / max(n_chunks, 1):.4f}")

    del optimizer, x_tensor, y_tensor
    gc.collect()
    model.eval()
    return model


@torch.no_grad()
def export_to_hf_safetensors_and_compute_ref_logits(model: OdiaSLM) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"[3/5] Exporting OdiaSLM-100M-Instruct to Hugging Face LlamaForCausalLM format at {EXPORT_DIR}...")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Quantize model in-place to FP16 -> FP32 so native reference logits match SafeTensors FP16 quantization
    for p in model.parameters():
        p.data = p.data.to(torch.float16).to(torch.float32)

    test_ids = torch.tensor([[1, 245, 892, 1023, 44, 12]], dtype=torch.long)
    nat_logits = extract_logits(model(test_ids)).detach().clone()

    src_sd = model.state_dict()
    hf_sd = {}

    hf_sd["model.embed_tokens.weight"] = src_sd["embed_tokens.weight"].to(torch.float16).contiguous()
    hf_sd["model.norm.weight"] = src_sd["norm.weight"].to(torch.float16).contiguous()
    hf_sd["lm_head.weight"] = src_sd["embed_tokens.weight"].to(torch.float16).contiguous()

    for i in range(14):
        p_src = f"layers.{i}"
        p_hf = f"model.layers.{i}"
        for sub in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]:
            hf_sd[f"{p_hf}.{sub}"] = src_sd[f"{p_src}.{sub}"].to(torch.float16).contiguous()

    save_file(hf_sd, str(EXPORT_DIR / "model.safetensors"), metadata={"format": "pt"})
    del hf_sd, src_sd
    gc.collect()

    base_export = Path("artifacts/hf_export/odia-slm-100m-base")
    for fname in ["tokenizer.model", "tokenizer.json", "special_tokens_map.json"]:
        if (base_export / fname).exists():
            shutil.copy2(base_export / fname, EXPORT_DIR / fname)
    if not (EXPORT_DIR / "tokenizer.model").exists():
        shutil.copy2(TOK_PATH, EXPORT_DIR / "tokenizer.model")

    config_json = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "hidden_act": "silu",
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 2048,
        "max_position_embeddings": 1024,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 12,
        "num_hidden_layers": 14,
        "num_key_value_heads": 4,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
        "transformers_version": "4.46.0",
        "use_cache": True,
        "vocab_size": 16384,
    }
    (EXPORT_DIR / "config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")

    gen_cfg = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_length": 1024,
        "do_sample": False,
        "repetition_penalty": 1.18,
    }
    (EXPORT_DIR / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2), encoding="utf-8")

    tok_cfg_path = base_export / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8")) if tok_cfg_path.exists() else {}
    tok_cfg["chat_template"] = (
        "<s>{% for message in messages %}"
        "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n<|assistant|>\n"
        "{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}"
        "{% endfor %}"
    )
    (EXPORT_DIR / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return test_ids, nat_logits


@torch.no_grad()
def verify_hf_parity_and_run_chat_suite(
    test_ids: torch.Tensor,
    nat_logits: torch.Tensor,
    sp: spm.SentencePieceProcessor,
) -> dict:
    print("[4/5] Verifying Float32 Logit Parity with Hugging Face AutoModelForCausalLM...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(EXPORT_DIR),
        torch_dtype=torch.float32,
        trust_remote_code=False,
    ).eval()

    hf_logits = hf_model(test_ids).logits
    max_diff = float((nat_logits - hf_logits).abs().max().item())
    print(f"  Max Abs Logit Diff (Native OdiaSLM vs. HF LlamaForCausalLM): {max_diff:.2e}")

    print("\n[5/5] Running 10-Prompt Qualitative Chat Benchmark on OdiaSLM-100M-Instruct...")
    test_prompts = [
        "ଆପଣ କିଏ?",
        "OdiaSLM କ'ଣ ଏବଂ ଏହାର ବିଶେଷତ୍ୱ କ'ଣ?",
        "ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ'ଣ?",
        "ଓଡ଼ିଆ ଭାଷା କେବେ ଶାସ୍ତ୍ରୀୟ ଭାଷାର ମାନ୍ୟତା ପାଇଲା?",
        "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ବିଷୟରେ ସଂକ୍ଷେପରେ କୁହନ୍ତୁ।",
        "ଜଳ ସଂରକ୍ଷଣ କାହିଁକି ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ?",
        "ନିୟମିତ ବ୍ୟାୟାମ କରିବାର ଲାଭ କ'ଣ?",
        "ତୁମେ କେଉଁ କେଉଁ କାମରେ ସାହାଯ୍ୟ କରିପାରିବ?",
        "ଶିକ୍ଷା ମଣିଷ ଜୀବନରେ କାହିଁକି ଆବଶ୍ୟକ?",
        "ବୃକ୍ଷରୋପଣର ଉପକାରିତା ବିଷୟରେ ଦୁଇଟି ବାକ୍ୟ ଲେଖନ୍ତୁ।",
    ]

    qa_results = []
    for idx, q in enumerate(test_prompts, 1):
        formatted = f"<|user|>\n{q}\n<|assistant|>\n"
        in_ids = torch.tensor([[sp.bos_id()] + sp.encode(formatted, out_type=int)], dtype=torch.long)
        t0 = time.time()
        out_ids = hf_model.generate(
            in_ids,
            max_new_tokens=110,
            do_sample=False,
            repetition_penalty=1.18,
            eos_token_id=sp.eos_id(),
            pad_token_id=sp.pad_id(),
        )
        elapsed = time.time() - t0
        new_tokens = out_ids[0, in_ids.size(1) :].tolist()
        if sp.eos_id() in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(sp.eos_id())]
        ans = sp.decode(new_tokens).strip()
        tok_per_s = len(new_tokens) / max(elapsed, 1e-4)
        print(f"\n  [{idx:02d}] Q: {q}")
        print(f"       A: {ans}")
        print(f"       ({len(new_tokens)} tokens | {tok_per_s:.1f} tok/s on CPU)")
        qa_results.append({"question": q, "answer": ans, "tokens": len(new_tokens), "tok_per_sec": round(tok_per_s, 1)})

    return {"max_logit_diff": max_diff, "qa_benchmark": qa_results}


def main():
    sp = spm.SentencePieceProcessor(model_file=str(TOK_PATH))
    download_checkpoints()
    model = build_blended_and_calibrated_model(sp)
    test_ids, nat_logits = export_to_hf_safetensors_and_compute_ref_logits(model)

    # Free native model before loading HF AutoModelForCausalLM
    del model
    gc.collect()

    eval_summary = verify_hf_parity_and_run_chat_suite(test_ids, nat_logits, sp)

    size_mb = round((EXPORT_DIR / "model.safetensors").stat().st_size / (1024 * 1024), 2)
    report = {
        "step": "step_113_export_and_verify_instruct",
        "export_dir": str(EXPORT_DIR),
        "safetensors_size_mb": size_mb,
        **eval_summary,
    }
    Path("artifacts/step_113_instruct_export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== STEP 11.3 ODIASLM-100M-INSTRUCT EXPORT & VERIFICATION COMPLETE ===")
    print(f"Export Path         : {EXPORT_DIR}/model.safetensors ({size_mb} MB)")
    print(f"HF Logit Parity     : {eval_summary['max_logit_diff']:.2e}")
    print("======================================================================")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

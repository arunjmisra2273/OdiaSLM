#!/usr/bin/env python3
"""
Phase 3: Direct Preference Optimization (DPO) + Chosen-NLL Anchor Engine for OdiaSLM-100M-Instruct.
Eliminates clause/numbered-list repetition and locks in Sovereign Identity & Odisha factual grounding.
Supports Single-GPU (Google Colab 1x T4) and Multi-GPU DDP (Kaggle 2x T4).
Author: Arun Misra | AI Researcher, Bhubaneswar, Odisha, India
"""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi, hf_hub_download

from src.model import OdiaSLM, OdiaSLMConfig


def extract_logits(out) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class DPODataset:
    """Memory-mapped loader for 512-token (chosen, rejected) preference pairs with -100 prompt masks."""

    def __init__(
        self,
        c_in_path: Path,
        c_lab_path: Path,
        r_in_path: Path,
        r_lab_path: Path,
    ):
        self.c_in = np.load(c_in_path, mmap_mode="r")
        self.c_lab = np.load(c_lab_path, mmap_mode="r")
        self.r_in = np.load(r_in_path, mmap_mode="r")
        self.r_lab = np.load(r_lab_path, mmap_mode="r")
        assert self.c_in.shape == self.c_lab.shape == self.r_in.shape == self.r_lab.shape
        self.num_rows, self.seq_len = self.c_in.shape

    def __len__(self) -> int:
        return self.num_rows

    def get_batch(self, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
        cx = torch.from_numpy(np.asarray(self.c_in[indices, :-1], dtype=np.int64)).to(device, non_blocking=True)
        cy = torch.from_numpy(np.asarray(self.c_lab[indices, 1:], dtype=np.int64)).to(device, non_blocking=True)
        rx = torch.from_numpy(np.asarray(self.r_in[indices, :-1], dtype=np.int64)).to(device, non_blocking=True)
        ry = torch.from_numpy(np.asarray(self.r_lab[indices, 1:], dtype=np.int64)).to(device, non_blocking=True)
        return cx, cy, rx, ry


def compute_sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes length-normalized and sum log-probabilities over active assistant response tokens
    (where labels != -100), plus standard NLL loss on the sequence.
    """
    logits_f = logits.float()
    log_probs = F.log_softmax(logits_f, dim=-1)

    mask = labels != ignore_index
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    per_token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token_logps = per_token_logps * mask.float()

    token_counts = mask.float().sum(dim=-1).clamp(min=1.0)
    seq_logp_sum = per_token_logps.sum(dim=-1)
    seq_logp_mean = seq_logp_sum / token_counts
    nll_loss = -seq_logp_sum.sum() / mask.float().sum().clamp(min=1.0)
    return seq_logp_sum, seq_logp_mean, nll_loss


def compute_dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
    nll_weight: float = 0.15,
) -> tuple[torch.Tensor, dict]:
    """
    Computes DPO loss + Chosen-NLL anchor loss to prevent log-prob drift and eliminate repetition.
    Uses length-calibrated log-ratios so short truncated answers cannot game sum log-probabilities.
    """
    _, pi_c_mean, chosen_nll = compute_sequence_logps(policy_chosen_logits, chosen_labels)
    _, pi_r_mean, _ = compute_sequence_logps(policy_rejected_logits, rejected_labels)

    with torch.no_grad():
        _, ref_c_mean, _ = compute_sequence_logps(ref_chosen_logits, chosen_labels)
        _, ref_r_mean, _ = compute_sequence_logps(ref_rejected_logits, rejected_labels)

    # Scale mean log-prob by average response length (~45 tokens) so beta=0.1 has standard sharpness
    scale_factor = 40.0
    chosen_rewards = beta * scale_factor * (pi_c_mean - ref_c_mean)
    rejected_rewards = beta * scale_factor * (pi_r_mean - ref_r_mean)
    reward_margins = chosen_rewards - rejected_rewards

    dpo_loss = -F.logsigmoid(reward_margins).mean()
    total_loss = dpo_loss + nll_weight * chosen_nll

    pref_acc = (chosen_rewards > rejected_rewards).float().mean().item() * 100.0
    metrics = {
        "total_loss": float(total_loss.detach().item()),
        "dpo_loss": float(dpo_loss.detach().item()),
        "chosen_nll": float(chosen_nll.detach().item()),
        "reward_margin": float(reward_margins.detach().mean().item()),
        "pref_acc_pct": float(pref_acc),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_dpo_validation(
    policy_model: OdiaSLM,
    ref_model: OdiaSLM,
    val_ds: DPODataset,
    batch_size: int,
    device: torch.device,
    beta: float = 0.1,
    nll_weight: float = 0.15,
) -> dict:
    policy_model.eval()
    losses, dpo_losses, nlls, margins, accs = [], [], [], [], []
    for start in range(0, len(val_ds), batch_size):
        idx = np.arange(start, min(start + batch_size, len(val_ds)))
        cx, cy, rx, ry = val_ds.get_batch(idx, device)
        cat_x = torch.cat([cx, rx], dim=0)
        b = cx.size(0)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            pi_logits = extract_logits(policy_model(cat_x))
            ref_logits = extract_logits(ref_model(cat_x))
            _, m = compute_dpo_loss(
                pi_logits[:b],
                pi_logits[b:],
                ref_logits[:b],
                ref_logits[b:],
                cy,
                ry,
                beta=beta,
                nll_weight=nll_weight,
            )
        losses.append(m["total_loss"])
        dpo_losses.append(m["dpo_loss"])
        nlls.append(m["chosen_nll"])
        margins.append(m["reward_margin"])
        accs.append(m["pref_acc_pct"])

    policy_model.train()
    return {
        "val_total_loss": round(float(np.mean(losses)), 4),
        "val_dpo_loss": round(float(np.mean(dpo_losses)), 4),
        "val_chosen_nll": round(float(np.mean(nlls)), 4),
        "val_reward_margin": round(float(np.mean(margins)), 4),
        "val_pref_acc_pct": round(float(np.mean(accs)), 2),
    }


@torch.no_grad()
def generate_chat_sample(
    raw_model: OdiaSLM,
    sp: spm.SentencePieceProcessor,
    user_prompt: str,
    device: torch.device,
    max_new_tokens: int = 90,
) -> str:
    raw_model.eval()
    formatted = f"<|user|>\n{user_prompt}\n<|assistant|>\n"
    ids = [sp.bos_id()] + sp.encode(formatted, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            logits = extract_logits(raw_model(x[:, -1023:]))
        next_logits = logits[0, -1, :].float()
        for prev_id in set(generated[-40:]):
            if next_logits[prev_id] > 0:
                next_logits[prev_id] /= 1.18
            else:
                next_logits[prev_id] *= 1.18
        next_id = int(torch.argmax(next_logits).item())
        if next_id == sp.eos_id() or next_id == sp.pad_id():
            break
        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    raw_model.train()
    return sp.decode(generated).strip()


def get_cosine_lr(step: int, max_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def ensure_dpo_data_and_sft_ckpt(
    data_dir: Path,
    sft_ckpt_path: Path,
    dpo_data_repo: str,
    hf_ckpt_repo: str,
    hf_token: str | None,
):
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "dpo_train_chosen_inputs.npy",
        "dpo_train_chosen_labels.npy",
        "dpo_train_rejected_inputs.npy",
        "dpo_train_rejected_labels.npy",
        "dpo_val_chosen_inputs.npy",
        "dpo_val_chosen_labels.npy",
        "dpo_val_rejected_inputs.npy",
        "dpo_val_rejected_labels.npy",
        "tokenizer.model",
    ]
    for fname in files:
        if not (data_dir / fname).exists():
            print(f"Downloading {fname} from {dpo_data_repo}...")
            hf_hub_download(
                repo_id=dpo_data_repo,
                filename=fname,
                repo_type="dataset",
                local_dir=str(data_dir),
                token=hf_token,
            )

    if not sft_ckpt_path.exists():
        sft_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SFT best checkpoint from {hf_ckpt_repo}...")
        hf_hub_download(
            repo_id=hf_ckpt_repo,
            filename="sft_ckpt_best.pt",
            repo_type="model",
            local_dir=str(sft_ckpt_path.parent),
            token=hf_token,
        )


def main():
    parser = argparse.ArgumentParser(description="OdiaSLM-100M-Instruct Phase 3 DPO Alignment Engine")
    parser.add_argument("--data-dir", type=str, default="data/dpo")
    parser.add_argument("--out-dir", type=str, default="checkpoints/odia_100m_dpo")
    parser.add_argument("--sft-ckpt", type=str, default="checkpoints/odia_100m_sft/sft_ckpt_best.pt")
    parser.add_argument("--dpo-data-repo", type=str, default="arunjmisra2273/odia-slm-dpo-bin")
    parser.add_argument("--hf-ckpt-repo", type=str, default="arunjmisra2273/odia-slm-100m-instruct-ckpt")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=6e-6)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll-weight", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master = rank == 0
    hf_token = os.environ.get("HF_TOKEN")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    sft_ckpt_path = Path(args.sft_ckpt)

    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
        ensure_dpo_data_and_sft_ckpt(data_dir, sft_ckpt_path, args.dpo_data_repo, args.hf_ckpt_repo, hf_token)
    if ddp:
        dist.barrier()

    tok_file = data_dir / "tokenizer.model"
    if not tok_file.exists():
        tok_file = Path("artifacts/tokenizer/tokenizer.model")
    sp = spm.SentencePieceProcessor(model_file=str(tok_file))

    train_ds = DPODataset(
        data_dir / "dpo_train_chosen_inputs.npy",
        data_dir / "dpo_train_chosen_labels.npy",
        data_dir / "dpo_train_rejected_inputs.npy",
        data_dir / "dpo_train_rejected_labels.npy",
    )
    val_ds = DPODataset(
        data_dir / "dpo_val_chosen_inputs.npy",
        data_dir / "dpo_val_chosen_labels.npy",
        data_dir / "dpo_val_rejected_inputs.npy",
        data_dir / "dpo_val_rejected_labels.npy",
    )

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
    cfg.n_layers = 14
    cfg.d_model = 768
    cfg.n_heads = 12
    cfg.n_kv_heads = 4
    cfg.d_ff = 2048
    cfg.max_seq_len = 1024
    cfg.vocab_size = 16384

    raw_policy = OdiaSLM(cfg).to(device)
    sft_ckpt = torch.load(sft_ckpt_path, map_location=device, weights_only=False)
    sd = sft_ckpt.get("model_state_dict") or sft_ckpt.get("model") or sft_ckpt
    clean_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    raw_policy.load_state_dict(clean_sd, strict=True)

    # Create frozen FP16 reference model (uses only ~192 MB VRAM)
    ref_model = copy.deepcopy(raw_policy).to(device)
    if device.type == "cuda":
        ref_model = ref_model.half()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if is_master:
        print(f"Loaded SFT checkpoint from {sft_ckpt_path} (Step {sft_ckpt.get('step', 1219)}) into Policy & Frozen Ref Model.")

    optimizer = torch.optim.AdamW(raw_policy.parameters(), lr=args.peak_lr, betas=(0.9, 0.95), weight_decay=0.01)
    model = DDP(raw_policy, device_ids=[local_rank]) if ddp else raw_policy
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    api = HfApi(token=hf_token) if (is_master and hf_token) else None
    rng = np.random.default_rng(108 + rank)
    num_rows = len(train_ds)
    perm = rng.permutation(num_rows)
    ptr = 0

    best_val_loss = float("inf")
    log_path = out_dir / "dpo_training_log.jsonl"
    dpo_latest = out_dir / "dpo_ckpt_latest.pt"
    dpo_best = out_dir / "dpo_ckpt_best.pt"

    if is_master:
        print(f"\n=== STARTING ODIASLM-100M-INSTRUCT PHASE 3 DPO (Steps 0 -> {args.max_steps}) ===")

    t0 = time.time()
    for step in range(args.max_steps):
        lr = get_cosine_lr(step, args.max_steps, args.warmup_steps, args.peak_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_metrics = {"total_loss": 0.0, "dpo_loss": 0.0, "chosen_nll": 0.0, "reward_margin": 0.0, "pref_acc_pct": 0.0}

        for micro_step in range(args.grad_accum_steps):
            if ptr + args.micro_batch_size > num_rows:
                perm = rng.permutation(num_rows)
                ptr = 0
            idx = perm[ptr : ptr + args.micro_batch_size]
            ptr += args.micro_batch_size

            cx, cy, rx, ry = train_ds.get_batch(idx, device)
            cat_x = torch.cat([cx, rx], dim=0)
            b = cx.size(0)

            sync_ctx = model.no_sync() if (ddp and micro_step < args.grad_accum_steps - 1) else torch.enable_grad()
            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                    pi_logits = extract_logits(model(cat_x))
                    with torch.no_grad():
                        ref_logits = extract_logits(ref_model(cat_x))
                    loss, m = compute_dpo_loss(
                        pi_logits[:b],
                        pi_logits[b:],
                        ref_logits[:b],
                        ref_logits[b:],
                        cy,
                        ry,
                        beta=args.beta,
                        nll_weight=args.nll_weight,
                    )
                    scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()

            for k in step_metrics:
                step_metrics[k] += m[k] / args.grad_accum_steps

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_policy.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()

        if is_master and (step % 10 == 0 or step == args.max_steps - 1):
            elapsed = time.time() - t0
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
            print(
                f"Step {step:3d}/{args.max_steps} | DPO: {step_metrics['dpo_loss']:.4f} | NLL: {step_metrics['chosen_nll']:.4f} | "
                f"Margin: {step_metrics['reward_margin']:+6.3f} | PrefAcc: {step_metrics['pref_acc_pct']:5.1f}% | "
                f"Grad: {grad_norm:.3f} | VRAM: {vram_gb:.2f}GB | {elapsed:.1f}s"
            )

        is_eval = (step > 0 and step % args.eval_every == 0) or (step == args.max_steps - 1)
        val_m = None
        if is_eval and is_master:
            val_m = evaluate_dpo_validation(raw_policy, ref_model, val_ds, args.micro_batch_size, device, args.beta, args.nll_weight)
            q1 = "ଆପଣ କିଏ?"
            a1 = generate_chat_sample(raw_policy, sp, q1, device)
            q2 = "ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ'ଣ?"
            a2 = generate_chat_sample(raw_policy, sp, q2, device)
            q3 = "OdiaSLM କ'ଣ ଏବଂ ଏହାର ବିଶେଷତ୍ୱ କ'ଣ?"
            a3 = generate_chat_sample(raw_policy, sp, q3, device)
            print(
                f"\n  [VAL @ Step {step}] Val DPO: {val_m['val_dpo_loss']:.4f} | Val NLL: {val_m['val_chosen_nll']:.4f} | "
                f"Val Margin: {val_m['val_reward_margin']:+.3f} | Val PrefAcc: {val_m['val_pref_acc_pct']:.2f}%"
            )
            print(f"  [SAMPLE 1] Q: {q1} -> A: {a1}")
            print(f"  [SAMPLE 2] Q: {q2} -> A: {a2}")
            print(f"  [SAMPLE 3] Q: {q3} -> A: {a3}\n")

        if is_master:
            entry = {"step": step, "lr": lr, "grad_norm": round(grad_norm, 4), **{k: round(v, 4) for k, v in step_metrics.items()}}
            if val_m:
                entry.update(val_m)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        is_save = (step > 0 and step % args.save_every == 0) or (step == args.max_steps - 1)
        if is_save and is_master:
            payload = {
                "step": step,
                "model_state_dict": raw_policy.state_dict(),
                "config": cfg.__dict__,
                "val_metrics": val_m,
            }
            torch.save(payload, dpo_latest)
            if val_m and val_m["val_total_loss"] <= best_val_loss:
                best_val_loss = val_m["val_total_loss"]
                torch.save(payload, dpo_best)

            if api is not None:
                try:
                    api.upload_file(path_or_fileobj=str(dpo_latest), path_in_repo="dpo_ckpt_latest.pt", repo_id=args.hf_ckpt_repo, repo_type="model")
                    api.upload_file(path_or_fileobj=str(log_path), path_in_repo="dpo_training_log.jsonl", repo_id=args.hf_ckpt_repo, repo_type="model")
                    if dpo_best.exists():
                        api.upload_file(path_or_fileobj=str(dpo_best), path_in_repo="dpo_ckpt_best.pt", repo_id=args.hf_ckpt_repo, repo_type="model")
                    print(f"  [HF SYNC] Uploaded DPO checkpoint & log at step {step} -> {args.hf_ckpt_repo}")
                except Exception as e:
                    print(f"  [HF SYNC WARNING] {e}")

    if ddp:
        dist.destroy_process_group()
    if is_master:
        print("\n=== ODIASLM-100M-INSTRUCT PHASE 3 DPO ALIGNMENT COMPLETE ===")


if __name__ == "__main__":
    main()

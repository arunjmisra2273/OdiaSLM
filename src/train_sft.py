#!/usr/bin/env python3
"""
Phase 2: Full-Parameter Supervised Fine-Tuning (SFT) Engine for OdiaSLM-100M-Instruct.
Supports both Single-GPU (Google Colab 1x T4) and Multi-GPU DDP (Kaggle 2x T4).
Features:
  - Assistant-Only Loss Masking (ignore_index = -100)
  - NEFTune Embedding Noise Regularization (alpha = 5.0)
  - Auxiliary Max-Logit Z-Loss (1e-4) for FP16 Tensor Core stability
  - Cosine LR schedule with warmup (5e-5 -> 5e-6)
  - Atomic checkpoint & telemetry sync to Hugging Face Hub
Author: Arun Misra | AI Researcher, Bhubaneswar, Odisha, India
"""

import argparse
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


class PackedSFTDataset:
    """Memory-mapped loader for 1024-token packed SFT inputs (uint16) and loss-masked labels (int32)."""

    def __init__(self, inputs_path: Path, labels_path: Path):
        self.inputs = np.load(inputs_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        assert self.inputs.shape == self.labels.shape, "Mismatch between inputs and labels shapes!"
        self.num_rows, self.seq_len = self.inputs.shape

    def __len__(self) -> int:
        return self.num_rows

    def get_batch(self, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Standard causal shift: x = tokens[:-1], y = labels[1:]
        x_np = np.asarray(self.inputs[indices, :-1], dtype=np.int64)
        y_np = np.asarray(self.labels[indices, 1:], dtype=np.int64)
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)
        return x, y


def compute_sft_loss_with_zloss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    z_loss_weight: float = 1e-4,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float, float, int]:
    """
    Computes assistant-only Cross-Entropy loss (ignoring prompt/pad tokens where targets == -100)
    plus PaLM-style auxiliary Z-loss over active positions.
    """
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_targets = targets.reshape(-1)

    active_mask = flat_targets != ignore_index
    active_count = int(active_mask.sum().item())

    if active_count == 0:
        zero = logits.sum() * 0.0
        return zero, 0.0, 0.0, 0

    ce_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="mean")

    if z_loss_weight > 0.0:
        active_logits = flat_logits[active_mask]
        log_z = torch.logsumexp(active_logits, dim=-1)
        z_loss = z_loss_weight * torch.mean(log_z ** 2)
        total_loss = ce_loss + z_loss
        z_val = float(z_loss.detach().item())
    else:
        total_loss = ce_loss
        z_val = 0.0

    return total_loss, float(ce_loss.detach().item()), z_val, active_count


@torch.no_grad()
def evaluate_sft_validation(
    raw_model: OdiaSLM,
    val_ds: PackedSFTDataset,
    batch_size: int,
    device: torch.device,
) -> dict:
    raw_model.eval()
    total_ce_nats = 0.0
    total_active_tokens = 0
    total_top1_correct = 0

    num_rows = len(val_ds)
    for start in range(0, num_rows, batch_size):
        idx = np.arange(start, min(start + batch_size, num_rows))
        x, y = val_ds.get_batch(idx, device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            out = raw_model(x)
            logits = extract_logits(out)

        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_y = y.reshape(-1)
        mask = flat_y != -100
        n_act = int(mask.sum().item())
        if n_act == 0:
            continue

        ce_sum = F.cross_entropy(flat_logits[mask], flat_y[mask], reduction="sum")
        preds = flat_logits[mask].argmax(dim=-1)
        correct = int((preds == flat_y[mask]).sum().item())

        total_ce_nats += float(ce_sum.item())
        total_active_tokens += n_act
        total_top1_correct += correct

    mean_ce = total_ce_nats / max(total_active_tokens, 1)
    ppl = math.exp(min(mean_ce, 20.0))
    top1_acc = 100.0 * total_top1_correct / max(total_active_tokens, 1)
    raw_model.train()
    return {
        "val_ce_loss": round(mean_ce, 4),
        "val_ppl": round(ppl, 2),
        "val_top1_acc_pct": round(top1_acc, 2),
        "val_active_tokens": total_active_tokens,
    }


@torch.no_grad()
def generate_chat_sample(
    raw_model: OdiaSLM,
    sp: spm.SentencePieceProcessor,
    user_prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
) -> str:
    raw_model.eval()
    formatted = f"<|user|>\n{user_prompt}\n<|assistant|>\n"
    ids = [sp.bos_id()] + sp.encode(formatted, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            out = raw_model(x[:, -1023:])
            logits = extract_logits(out)
        next_logits = logits[0, -1, :].float()
        # Mild repetition penalty (1.15) + greedy/low-temp decoding for deterministic sanity check
        for prev_id in set(generated[-32:]):
            if next_logits[prev_id] > 0:
                next_logits[prev_id] /= 1.15
            else:
                next_logits[prev_id] *= 1.15
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


def ensure_data_and_base_weights(
    data_dir: Path,
    base_ckpt_path: Path,
    sft_data_repo: str,
    base_ckpt_repo: str,
    hf_token: str | None,
):
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["sft_train_inputs.npy", "sft_train_labels.npy", "sft_val_inputs.npy", "sft_val_labels.npy", "tokenizer.model"]:
        target = data_dir / fname
        if not target.exists():
            print(f"Downloading {fname} from dataset {sft_data_repo}...")
            hf_hub_download(
                repo_id=sft_data_repo,
                filename=fname,
                repo_type="dataset",
                local_dir=str(data_dir),
                token=hf_token,
            )

    if not base_ckpt_path.exists():
        base_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pretrained 100M base checkpoint from {base_ckpt_repo}...")
        hf_hub_download(
            repo_id=base_ckpt_repo,
            filename="ckpt_latest.pt",
            repo_type="model",
            local_dir=str(base_ckpt_path.parent),
            token=hf_token,
        )


def main():
    parser = argparse.ArgumentParser(description="OdiaSLM-100M-Instruct Full-Parameter SFT Engine")
    parser.add_argument("--data-dir", type=str, default="data/sft")
    parser.add_argument("--out-dir", type=str, default="checkpoints/odia_100m_sft")
    parser.add_argument("--base-ckpt", type=str, default="checkpoints/base_100m/ckpt_latest.pt")
    parser.add_argument("--sft-data-repo", type=str, default="arunjmisra2273/odia-slm-sft-bin")
    parser.add_argument("--base-ckpt-repo", type=str, default="arunjmisra2273/odia-slm-100m-ckpt")
    parser.add_argument("--hf-ckpt-repo", type=str, default="arunjmisra2273/odia-slm-100m-instruct-ckpt")
    parser.add_argument("--max-steps", type=int, default=1220)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=4e-5)
    parser.add_argument("--min-lr", type=float, default=4e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--neftune-alpha", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
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
    base_ckpt_path = Path(args.base_ckpt)

    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
        ensure_data_and_base_weights(data_dir, base_ckpt_path, args.sft_data_repo, args.base_ckpt_repo, hf_token)
    if ddp:
        dist.barrier()

    tok_file = data_dir / "tokenizer.model"
    if not tok_file.exists():
        tok_file = Path("artifacts/tokenizer/tokenizer.model")
    sp = spm.SentencePieceProcessor(model_file=str(tok_file))

    train_ds = PackedSFTDataset(data_dir / "sft_train_inputs.npy", data_dir / "sft_train_labels.npy")
    val_ds = PackedSFTDataset(data_dir / "sft_val_inputs.npy", data_dir / "sft_val_labels.npy")

    # Load base checkpoint metadata first so OdiaSLMConfig matches exact saved field names
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = base_ckpt.get("config", {})
    if isinstance(saved_cfg, OdiaSLMConfig):
        cfg = saved_cfg
    elif isinstance(saved_cfg, dict) and saved_cfg:
        valid_keys = {f.name for f in __import__("dataclasses").fields(OdiaSLMConfig)}
        filtered_cfg = {k: v for k, v in saved_cfg.items() if k in valid_keys}
        cfg = OdiaSLMConfig(**filtered_cfg)
    else:
        cfg = OdiaSLMConfig(
            model_variant="OdiaSLM-100M",
            vocab_size=16384,
            n_layers=14,
            d_model=768,
            n_heads=12,
            n_kv_heads=4,
            d_ff=2048,
            max_seq_len=1024,
            dropout=0.05,
        )
    if hasattr(cfg, "dropout"):
        cfg.dropout = 0.05
    raw_model = OdiaSLM(cfg).to(device)

    # Check if we are resuming an existing SFT checkpoint or starting fresh from Base 100M
    sft_latest = out_dir / "sft_ckpt_latest.pt"
    if is_master and not sft_latest.exists() and hf_token:
        try:
            hf_hub_download(
                repo_id=args.hf_ckpt_repo,
                filename="sft_ckpt_latest.pt",
                repo_type="model",
                local_dir=str(out_dir),
                token=hf_token,
            )
            print("Downloaded existing SFT checkpoint from HF Hub to resume!")
        except Exception:
            pass
    if ddp:
        dist.barrier()

    start_step = 0
    best_val_loss = float("inf")

    if sft_latest.exists():
        ckpt = torch.load(sft_latest, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        start_step = int(ckpt.get("step", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if is_master:
            print(f"Resumed SFT from step {start_step} (best_val_loss={best_val_loss:.4f})")
    else:
        base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        state_dict = base_ckpt.get("model_state_dict", base_ckpt)
        raw_model.load_state_dict(state_dict, strict=True)
        if is_master:
            print(f"Loaded pretrained OdiaSLM-100M base weights from {base_ckpt_path} (Step {base_ckpt.get('step', '11999')})")

    # Register NEFTune forward hook on token embeddings during training
    def neftune_hook(module, inp, out):
        if module.training and args.neftune_alpha > 0.0:
            dims = torch.tensor(out.size(1) * out.size(2), dtype=torch.float32, device=out.device)
            mag = args.neftune_alpha / torch.sqrt(dims)
            noise = torch.zeros_like(out).uniform_(-mag.item(), mag.item())
            return out + noise
        return out

    emb_layer = getattr(raw_model, "tok_embeddings", None)
    if emb_layer is None:
        for m in raw_model.modules():
            if isinstance(m, torch.nn.Embedding):
                emb_layer = m
                break
    if emb_layer is not None:
        emb_layer.register_forward_hook(neftune_hook)

    # Separate weight decay groups
    decay_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    if sft_latest.exists() and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    api = HfApi(token=hf_token) if (is_master and hf_token) else None
    if api is not None:
        try:
            api.create_repo(repo_id=args.hf_ckpt_repo, repo_type="model", private=True, exist_ok=True)
        except Exception as e:
            print(f"HF repo init warning: {e}")

    rng = np.random.default_rng(42 + rank)
    num_train_rows = len(train_ds)
    perm = rng.permutation(num_train_rows)
    ptr = (start_step * args.micro_batch_size * args.grad_accum_steps) % num_train_rows

    log_path = out_dir / "sft_training_log.jsonl"
    if is_master:
        print(f"\n=== STARTING ODIASLM-100M-INSTRUCT SFT (Steps {start_step} -> {args.max_steps}) ===")
        print(f"Device: {device} | World Size: {world_size} | Micro-Batch: {args.micro_batch_size} | Grad-Accum: {args.grad_accum_steps}")

    t0 = time.time()
    for step in range(start_step, args.max_steps):
        lr = get_cosine_lr(step, args.max_steps, args.warmup_steps, args.peak_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_ce_sum = 0.0
        step_z_sum = 0.0
        step_active_tokens = 0

        for micro_step in range(args.grad_accum_steps):
            if ptr + args.micro_batch_size > num_train_rows:
                perm = rng.permutation(num_train_rows)
                ptr = 0
            batch_idx = perm[ptr : ptr + args.micro_batch_size]
            ptr += args.micro_batch_size

            x, y = train_ds.get_batch(batch_idx, device)
            sync_ctx = model.no_sync() if (ddp and micro_step < args.grad_accum_steps - 1) else torch.enable_grad()
            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                    out = model(x)
                    logits = extract_logits(out)
                    loss, ce_val, z_val, act_cnt = compute_sft_loss_with_zloss(logits, y)
                    scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()

            step_ce_sum += ce_val
            step_z_sum += z_val
            step_active_tokens += act_cnt

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()

        mean_ce = step_ce_sum / args.grad_accum_steps
        mean_z = step_z_sum / args.grad_accum_steps
        ppl = math.exp(min(mean_ce, 20.0))

        if is_master and (step % 10 == 0 or step == args.max_steps - 1):
            elapsed = time.time() - t0
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
            print(
                f"Step {step:4d}/{args.max_steps} | CE: {mean_ce:.4f} | PPL: {ppl:6.2f} | "
                f"Grad: {grad_norm:.3f} | LR: {lr:.2e} | ActTok: {step_active_tokens:5d} | VRAM: {vram_gb:.2f}GB | {elapsed:.1f}s"
            )

        is_eval_step = (step > 0 and step % args.eval_every == 0) or (step == args.max_steps - 1)
        val_metrics = None
        if is_eval_step and is_master:
            val_metrics = evaluate_sft_validation(raw_model, val_ds, args.micro_batch_size, device)
            q1 = "ଆପଣ କିଏ?"
            a1 = generate_chat_sample(raw_model, sp, q1, device)
            q2 = "ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ'ଣ?"
            a2 = generate_chat_sample(raw_model, sp, q2, device)
            print(f"\n  [VAL @ Step {step}] Val CE: {val_metrics['val_ce_loss']:.4f} | Val PPL: {val_metrics['val_ppl']:.2f} | Val Top-1: {val_metrics['val_top1_acc_pct']:.2f}%")
            print(f"  [SAMPLE 1] Q: {q1} -> A: {a1}")
            print(f"  [SAMPLE 2] Q: {q2} -> A: {a2}\n")

        if is_master:
            log_entry = {
                "step": step,
                "train_ce": round(mean_ce, 4),
                "train_ppl": round(ppl, 2),
                "z_loss": round(mean_z, 5),
                "grad_norm": round(grad_norm, 4),
                "lr": lr,
                "active_tokens": step_active_tokens,
            }
            if val_metrics:
                log_entry.update(val_metrics)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        is_save_step = (step > 0 and step % args.save_every == 0) or (step == args.max_steps - 1)
        if is_save_step and is_master:
            ckpt_payload = {
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
                "best_val_loss": best_val_loss,
            }
            torch.save(ckpt_payload, sft_latest)
            if val_metrics and val_metrics["val_ce_loss"] <= best_val_loss:
                best_val_loss = val_metrics["val_ce_loss"]
                ckpt_payload["best_val_loss"] = best_val_loss
                torch.save(ckpt_payload, out_dir / "sft_ckpt_best.pt")

            if api is not None:
                try:
                    api.upload_file(path_or_fileobj=str(sft_latest), path_in_repo="sft_ckpt_latest.pt", repo_id=args.hf_ckpt_repo, repo_type="model")
                    api.upload_file(path_or_fileobj=str(log_path), path_in_repo="sft_training_log.jsonl", repo_id=args.hf_ckpt_repo, repo_type="model")
                    if (out_dir / "sft_ckpt_best.pt").exists():
                        api.upload_file(path_or_fileobj=str(out_dir / "sft_ckpt_best.pt"), path_in_repo="sft_ckpt_best.pt", repo_id=args.hf_ckpt_repo, repo_type="model")
                    print(f"  [HF SYNC] Uploaded SFT checkpoint & log at step {step} -> {args.hf_ckpt_repo}")
                except Exception as e:
                    print(f"  [HF SYNC WARNING] {e}")

    if ddp:
        dist.destroy_process_group()
    if is_master:
        print("\n=== ODIASLM-100M-INSTRUCT SFT TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()

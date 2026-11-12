#!/usr/bin/env python3
"""
Step 5.1 (Upgraded with Publication Telemetry): Production Training Engine for Odia SLM (60M & 100M).
Collects comprehensive research telemetry for paper figures & Hugging Face model cards:
- Step-level CE loss, Perplexity, Bits-Per-Token (BPT), Bits-Per-Byte (BPB), Z-Loss
- Top-1 and Top-5 Next-Token Accuracy (Train & Validation)
- Logit distribution stats (max, mean, std) proving Z-Loss stability
- Per-Layer Gradient L2 Norms (embed, layer_0..L-1, norm) for gradient-flow heatmaps
- Per-Component Weight L2 Norms & RMSNorm gamma drift across training
- Hardware throughput telemetry (tokens/sec, step ms, VRAM GB, estimated MFU %).
"""

import argparse
import contextlib
import json
import math
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dataset import MemmapDataset
from src.model import OdiaSLM, get_model_config


@dataclass
class TrainConfig:
    model_variant: str = "60m"
    bin_dir: str = "data/bin"
    out_dir: str = "checkpoints/60m"
    hf_repo_id: Optional[str] = None

    # Step & Token Schedule
    max_steps: int = 10_000
    warmup_steps: int = 400
    decay_start_step: int = 8_500
    context_switch_step: int = 8_000

    # Batch & Sequence Geometry
    stage1_seq_len: int = 512
    stage2_seq_len: int = 1024
    stage1_micro_batch_size: int = 32
    stage2_micro_batch_size: int = 16
    grad_accum_steps: int = 4

    # Optimizer & Regularization
    max_lr: float = 6.0e-4
    min_lr: float = 6.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Precision & Execution
    dtype: str = "float16"  # "float16", "bfloat16", or "float32"
    compile_model: bool = False
    seed: int = 1337
    overfit_single_batch: bool = False

    # Logging, Eval & Checkpointing
    log_interval: int = 20
    layer_telemetry_interval: int = 100
    val_interval: int = 250
    val_batches: int = 50
    ckpt_interval: int = 500
    milestone_interval: int = 2_000
    max_session_seconds: int = 40_500  # 11h 15m watchdog

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_wsd_lr(
    step: int,
    warmup_steps: int,
    decay_start_step: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Warmup-Stable-Decay (WSD) schedule with 1-minus-sqrt cooldown."""
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step < decay_start_step:
        return max_lr
    if step >= max_steps:
        return min_lr

    decay_len = max(1, max_steps - decay_start_step)
    progress = float(step - decay_start_step) / float(decay_len)
    progress = min(1.0, max(0.0, progress))
    return min_lr + (max_lr - min_lr) * (1.0 - math.sqrt(progress))


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    """Initializes DDP if launched via torchrun, else falls back to single process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
        is_ddp = True
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_ddp = False
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                device = torch.device("cuda:0")
            except Exception:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

    return is_ddp, rank, local_rank, world_size, device


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def unwrap_model(model: torch.nn.Module) -> OdiaSLM:
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m  # type: ignore


@torch.no_grad()
def collect_layer_and_weight_telemetry(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Extracts per-layer gradient L2 norms and weight RMS norms for research plots.
    Called after scaler.unscale_(optimizer) so gradients are in true scale.
    """
    raw = unwrap_model(model)
    grad_norms: Dict[str, float] = {}
    weight_rms: Dict[str, float] = {}

    if raw.embed_tokens.weight.grad is not None:
        grad_norms["embed"] = round(float(raw.embed_tokens.weight.grad.norm().item()), 5)
    weight_rms["embed"] = round(
        float(raw.embed_tokens.weight.pow(2).mean().sqrt().item()), 5
    )

    attn_w_norms = []
    mlp_w_norms = []
    norm_gammas = []

    for idx, layer in enumerate(raw.layers):
        # Compute combined L2 grad norm across all parameters in layer `idx`
        sq_sum = 0.0
        for p in layer.parameters():
            if p.grad is not None:
                sq_sum += float(p.grad.pow(2).sum().item())
        grad_norms[f"layer_{idx:02d}"] = round(math.sqrt(sq_sum), 5)

        attn_w_norms.append(float(layer.self_attn.q_proj.weight.pow(2).mean().sqrt().item()))
        mlp_w_norms.append(float(layer.mlp.up_proj.weight.pow(2).mean().sqrt().item()))
        norm_gammas.append(float(layer.input_layernorm.weight.mean().item()))

    if raw.norm.weight.grad is not None:
        grad_norms["final_norm"] = round(float(raw.norm.weight.grad.norm().item()), 5)

    weight_rms["attn_q_mean_rms"] = round(float(np.mean(attn_w_norms)), 5)
    weight_rms["mlp_up_mean_rms"] = round(float(np.mean(mlp_w_norms)), 5)
    weight_rms["rmsnorm_gamma_mean"] = round(float(np.mean(norm_gammas)), 5)

    return {
        "layer_grad_norms": grad_norms,
        "weight_rms_summary": weight_rms,
    }


@torch.no_grad()
def compute_logit_and_accuracy_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Computes Top-1 accuracy, Top-5 accuracy, and logit max/std on a batch."""
    # logits: (B, T, V), targets: (B, T)
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)

    # Top-1 & Top-5 accuracy
    top5_preds = torch.topk(flat_logits, k=5, dim=-1).indices  # (N, 5)
    top1_correct = (top5_preds[:, 0] == flat_targets).float().mean().item()
    top5_correct = (top5_preds == flat_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

    l_max = float(flat_logits.max().item())
    l_std = float(flat_logits.float().std().item())

    return {
        "top1_acc": round(top1_correct, 4),
        "top5_acc": round(top5_correct, 4),
        "logit_max": round(l_max, 3),
        "logit_std": round(l_std, 4),
    }


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    dataset: MemmapDataset,
    val_batches: int,
    batch_size: int,
    device: torch.device,
    amp_ctx,
) -> Dict[str, float]:
    """
    Evaluates model across `val_batches` deterministic validation batches.
    Returns CE loss, Perplexity, BPT, BPB, Z-Loss, Top-1 Acc, Top-5 Acc, and Logit stats.
    """
    raw_model = unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()

    total_ce = 0.0
    total_z = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    max_logit_seen = -1e9
    total_logit_std = 0.0

    for b_idx in range(val_batches):
        xv, yv = dataset.get_val_batch(batch_idx=b_idx, batch_size=batch_size, device=device)
        with amp_ctx:
            out = raw_model(xv, targets=yv)
        total_ce += float(out["loss_ce"].item())
        total_z += float(out["z_loss"].item())

        b_stats = compute_logit_and_accuracy_stats(out["logits"], yv)
        total_top1 += b_stats["top1_acc"]
        total_top5 += b_stats["top5_acc"]
        max_logit_seen = max(max_logit_seen, b_stats["logit_max"])
        total_logit_std += b_stats["logit_std"]

    if was_training:
        raw_model.train()

    denom = float(max(1, val_batches))
    mean_ce = total_ce / denom
    metrics = dataset.compute_val_metrics_from_ce(mean_ce)
    metrics["val_z_loss"] = round(total_z / denom, 6)
    metrics["val_top1_acc"] = round(total_top1 / denom, 4)
    metrics["val_top5_acc"] = round(total_top5 / denom, 4)
    metrics["val_logit_max"] = round(max_logit_seen, 3)
    metrics["val_logit_std"] = round(total_logit_std / denom, 4)
    return metrics


def save_atomic_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    global_step: int,
    epoch: int,
    tokens_seen: int,
    best_val_loss: float,
    train_cfg: TrainConfig,
    is_milestone: bool = False,
    is_best: bool = False,
    hf_repo_id: Optional[str] = None,
) -> Path:
    """Serializes the mandatory 10-key checkpoint dictionary using atomic os.replace()."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)

    cuda_rng = None
    if torch.cuda.is_available():
        try:
            cuda_rng = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_rng = None

    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "tokens_seen": int(tokens_seen),
        "best_val_loss": float(best_val_loss),
        "config": train_cfg.to_dict(),
        "rng_state_cpu": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "rng_state_device": cuda_rng,
    }

    tmp_path = out_dir / "ckpt_tmp.pt"
    latest_path = out_dir / "ckpt_latest.pt"

    torch.save(state, tmp_path)
    os.replace(tmp_path, latest_path)

    if is_best:
        shutil.copy2(latest_path, out_dir / "ckpt_best.pt")

    if is_milestone:
        milestone_path = out_dir / f"ckpt_step_{global_step:06d}.pt"
        shutil.copy2(latest_path, milestone_path)

    if hf_repo_id and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(repo_id=hf_repo_id, private=True, exist_ok=True)
            for fname in ["ckpt_latest.pt", "training_log.jsonl", "val_history.jsonl"]:
                fpath = out_dir / fname
                if fpath.exists():
                    api.upload_file(
                        path_or_fileobj=str(fpath),
                        path_in_repo=fname,
                        repo_id=hf_repo_id,
                        repo_type="model",
                    )
        except Exception as e:
            print(f"[WARN] HF Hub checkpoint upload failed (local ckpt is safe): {e}")

    return latest_path


def load_checkpoint_if_exists(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Dict[str, Any]:
    """Restores all 10 keys from a saved checkpoint for bit-exact continuation."""
    if not ckpt_path.exists():
        return {
            "resumed": False,
            "start_step": 0,
            "epoch": 0,
            "tokens_seen": 0,
            "best_val_loss": float("inf"),
        }

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt and ckpt["scaler"]:
        scaler.load_state_dict(ckpt["scaler"])

    rng_cpu = ckpt.get("rng_state_cpu")
    if rng_cpu:
        random.setstate(rng_cpu["python"])
        np.random.set_state(rng_cpu["numpy"])
        torch_rng = rng_cpu["torch"]
        if isinstance(torch_rng, torch.Tensor):
            torch.set_rng_state(torch_rng.cpu())

    rng_dev = ckpt.get("rng_state_device")
    if rng_dev is not None and device.type == "cuda":
        try:
            torch.cuda.set_rng_state_all([t.cpu() for t in rng_dev])
        except Exception:
            pass

    return {
        "resumed": True,
        "start_step": int(ckpt["global_step"]) + 1,
        "epoch": int(ckpt.get("epoch", 0)),
        "tokens_seen": int(ckpt.get("tokens_seen", 0)),
        "best_val_loss": float(ckpt.get("best_val_loss", float("inf"))),
    }


def run_training(train_cfg: TrainConfig) -> Dict[str, Any]:
    """Main training loop entry point with publication-grade telemetry collection."""
    t_session_start = time.monotonic()
    is_ddp, rank, local_rank, world_size, device = setup_distributed()
    is_master = (rank == 0)

    seed_everything(train_cfg.seed + rank)

    out_dir = Path(train_cfg.out_dir)
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    val_log_path = out_dir / "val_history.jsonl"

    model_cfg = get_model_config(train_cfg.model_variant)
    model = OdiaSLM(model_cfg).to(device)
    param_counts = model.count_parameters()
    num_params = param_counts["total_trainable_params"]

    optimizer = model.configure_optimizers(
        learning_rate=train_cfg.max_lr,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    use_fp16 = (train_cfg.dtype == "float16" and device.type == "cuda")
    use_bf16 = (train_cfg.dtype == "bfloat16" and device.type in ("cuda", "cpu"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    if device.type == "cuda":
        amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else torch.float32)
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_fp16 or use_bf16))
    else:
        amp_ctx = contextlib.nullcontext()

    latest_ckpt_path = out_dir / "ckpt_latest.pt"
    resume_info = load_checkpoint_if_exists(
        ckpt_path=latest_ckpt_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
    )

    if train_cfg.compile_model and device.type == "cuda":
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    dataset = MemmapDataset(
        bin_dir=train_cfg.bin_dir,
        seq_len=train_cfg.stage1_seq_len,
        seed=train_cfg.seed,
    )

    start_step = resume_info["start_step"]
    tokens_seen = resume_info["tokens_seen"]
    best_val_loss = resume_info["best_val_loss"]
    current_epoch = resume_info["epoch"]

    stage1_tokens_per_step = (
        world_size * train_cfg.grad_accum_steps * train_cfg.stage1_micro_batch_size * train_cfg.stage1_seq_len
    )
    stage2_token_offset = train_cfg.context_switch_step * stage1_tokens_per_step

    if is_master:
        print(
            f"[Train Init] Variant={train_cfg.model_variant} ({num_params:,} params) | "
            f"Device={device} (World={world_size}) | "
            f"Resumed={resume_info['resumed']} (start_step={start_step}, tokens_seen={tokens_seen:,})"
        )

    model.train()
    step_losses: List[float] = []
    last_val_metrics: Dict[str, Any] = {}
    t_step_start = time.monotonic()

    for step in range(start_step, train_cfg.max_steps):
        if step < train_cfg.context_switch_step:
            cur_seq_len = train_cfg.stage1_seq_len
            cur_micro_bsz = train_cfg.stage1_micro_batch_size
            rel_step = step
            tok_offset = 0
        else:
            cur_seq_len = train_cfg.stage2_seq_len
            cur_micro_bsz = train_cfg.stage2_micro_batch_size
            rel_step = step - train_cfg.context_switch_step
            tok_offset = stage2_token_offset

        if dataset.seq_len != cur_seq_len:
            dataset.set_seq_len(cur_seq_len)

        lr = compute_wsd_lr(
            step=step,
            warmup_steps=train_cfg.warmup_steps,
            decay_start_step=train_cfg.decay_start_step,
            max_steps=train_cfg.max_steps,
            max_lr=train_cfg.max_lr,
            min_lr=train_cfg.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        accum_ce = 0.0
        accum_z = 0.0
        last_batch_logit_stats: Dict[str, float] = {}

        for m_step in range(train_cfg.grad_accum_steps):
            fetch_step = 0 if train_cfg.overfit_single_batch else rel_step
            fetch_mstep = 0 if train_cfg.overfit_single_batch else m_step
            fetch_offset = 0 if train_cfg.overfit_single_batch else tok_offset

            x, y, current_epoch = dataset.get_train_micro_batch(
                global_step=fetch_step,
                micro_step=fetch_mstep,
                grad_accum_steps=train_cfg.grad_accum_steps,
                micro_batch_size=cur_micro_bsz,
                rank=rank,
                world_size=world_size,
                token_offset=fetch_offset,
                device=device,
            )

            sync_ctx = (
                model.no_sync()
                if (is_ddp and m_step < train_cfg.grad_accum_steps - 1)
                else contextlib.nullcontext()
            )
            with sync_ctx:
                with amp_ctx:
                    out = model(x, targets=y)
                    loss_scaled = out["loss_total"] / float(train_cfg.grad_accum_steps)
                    step_ce_val = float(out["loss_ce"].item())
                    step_z_val = float(out["z_loss"].item())
                    if is_master and m_step == train_cfg.grad_accum_steps - 1 and (
                        step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1
                    ):
                        last_batch_logit_stats = compute_logit_and_accuracy_stats(
                            out["logits"].detach(), y
                        )
                    del out

                scaler.scale(loss_scaled).backward()
                del loss_scaled

            accum_ce += step_ce_val / float(train_cfg.grad_accum_steps)
            accum_z += step_z_val / float(train_cfg.grad_accum_steps)

        # Unscale gradients before collecting layer telemetry and clipping
        scaler.unscale_(optimizer)

        layer_telemetry = {}
        if is_master and (
            step % train_cfg.layer_telemetry_interval == 0 or step == train_cfg.max_steps - 1
        ):
            layer_telemetry = collect_layer_and_weight_telemetry(model)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip).item()
        )

        scaler.step(optimizer)
        scaler.update()

        tokens_this_step = world_size * train_cfg.grad_accum_steps * cur_micro_bsz * cur_seq_len
        tokens_seen += tokens_this_step

        if is_ddp:
            loss_tensor = torch.tensor([accum_ce, accum_z], dtype=torch.float64, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            accum_ce = float(loss_tensor[0].item())
            accum_z = float(loss_tensor[1].item())

        step_losses.append(accum_ce)

        # Periodic step logging with full research telemetry
        if is_master and (step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1):
            dt = max(1e-6, time.monotonic() - t_step_start)
            steps_elapsed = train_cfg.log_interval if step > start_step else 1
            tok_per_sec = (tokens_this_step * steps_elapsed) / dt
            step_ms = (dt / steps_elapsed) * 1000.0

            train_metrics = dataset.compute_val_metrics_from_ce(accum_ce)
            vram_alloc_gb = (
                round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3)
                if device.type == "cuda"
                else 0.0
            )
            vram_res_gb = (
                round(torch.cuda.memory_reserved(device) / (1024 ** 3), 3)
                if device.type == "cuda"
                else 0.0
            )
            # Approximate FLOPs/sec = 6 * N * tok_per_sec (T4 FP16 peak = 65 TFLOPS per GPU)
            est_tflops = (6.0 * num_params * tok_per_sec) / 1e12
            mfu_pct = round(100.0 * est_tflops / (65.0 * world_size), 2) if device.type == "cuda" else 0.0

            log_rec = {
                "type": "train_step",
                "step": step,
                "epoch": current_epoch,
                "seq_len": cur_seq_len,
                "tokens_seen": tokens_seen,
                "train_loss_ce": round(accum_ce, 5),
                "train_ppl": train_metrics["val_perplexity"],
                "train_bpt": train_metrics["val_bits_per_token"],
                "train_bpb": train_metrics["val_bits_per_byte"],
                "train_z_loss": round(accum_z, 6),
                "grad_norm": round(grad_norm, 4),
                "lr": round(lr, 8),
                "scaler_scale": scaler.get_scale(),
                "tok_per_sec": round(tok_per_sec, 1),
                "step_time_ms": round(step_ms, 1),
                "vram_allocated_gb": vram_alloc_gb,
                "vram_reserved_gb": vram_res_gb,
                "est_mfu_pct": mfu_pct,
                **last_batch_logit_stats,
                **layer_telemetry,
            }
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(json.dumps(log_rec) + "\n")
            print(
                f"step {step:05d}/{train_cfg.max_steps:05d} | loss_ce: {accum_ce:.4f} | "
                f"bpb: {train_metrics['val_bits_per_byte']:.4f} | top1: {last_batch_logit_stats.get('top1_acc', 0.0):.3f} | "
                f"grad_norm: {grad_norm:.3f} | lr: {lr:.2e} | tok/s: {tok_per_sec:,.0f}"
            )
            t_step_start = time.monotonic()

        # Periodic validation
        if (step > 0 and step % train_cfg.val_interval == 0) or (step == train_cfg.max_steps - 1):
            val_metrics = evaluate_validation(
                model=model,
                dataset=dataset,
                val_batches=train_cfg.val_batches,
                batch_size=cur_micro_bsz,
                device=device,
                amp_ctx=amp_ctx,
            )
            last_val_metrics = val_metrics
            is_best = val_metrics["val_loss_ce"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["val_loss_ce"]

            if is_master:
                val_rec = {
                    "type": "validation",
                    "step": step,
                    "epoch": current_epoch,
                    "seq_len": cur_seq_len,
                    "tokens_seen": tokens_seen,
                    "lr": round(lr, 8),
                    "is_best": is_best,
                    **val_metrics,
                }
                with open(log_path, "a", encoding="utf-8") as flog:
                    flog.write(json.dumps(val_rec) + "\n")
                with open(val_log_path, "a", encoding="utf-8") as fval:
                    fval.write(json.dumps(val_rec) + "\n")
                print(
                    f"---> [VAL step {step:05d}] val_ce: {val_metrics['val_loss_ce']:.4f} | "
                    f"val_ppl: {val_metrics['val_perplexity']:.2f} | "
                    f"val_bpb: {val_metrics['val_bits_per_byte']:.4f} | "
                    f"top1: {val_metrics['val_top1_acc']:.3f} | top5: {val_metrics['val_top5_acc']:.3f}"
                )

        # Watchdog check
        elapsed_session = time.monotonic() - t_session_start
        watchdog_triggered = (elapsed_session >= train_cfg.max_session_seconds)
        if is_ddp:
            wd_tensor = torch.tensor([1 if watchdog_triggered else 0], dtype=torch.int32, device=device)
            dist.broadcast(wd_tensor, src=0)
            watchdog_triggered = bool(wd_tensor.item() == 1)

        should_ckpt = (
            (step > 0 and step % train_cfg.ckpt_interval == 0)
            or (step == train_cfg.context_switch_step - 1)
            or (step == train_cfg.decay_start_step - 1)
            or (step == train_cfg.max_steps - 1)
            or watchdog_triggered
        )

        if should_ckpt:
            is_milestone = (
                (step > 0 and step % train_cfg.milestone_interval == 0)
                or (step == train_cfg.context_switch_step - 1)
                or (step == train_cfg.decay_start_step - 1)
                or (step == train_cfg.max_steps - 1)
            )
            if is_master:
                saved_path = save_atomic_checkpoint(
                    out_dir=out_dir,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=step,
                    epoch=current_epoch,
                    tokens_seen=tokens_seen,
                    best_val_loss=best_val_loss,
                    train_cfg=train_cfg,
                    is_milestone=is_milestone,
                    is_best=(last_val_metrics.get("val_loss_ce") == best_val_loss),
                    hf_repo_id=train_cfg.hf_repo_id,
                )
                print(f"---> [CKPT step {step:05d}] Saved atomic checkpoint to {saved_path}")
            if is_ddp:
                dist.barrier()

        if watchdog_triggered:
            if is_master:
                print(
                    f"[WATCHDOG] Session elapsed {elapsed_session:.1f}s >= limit {train_cfg.max_session_seconds}s. "
                    f"Cleanly exiting at step {step} after atomic checkpoint."
                )
            break

    if is_ddp:
        dist.destroy_process_group()

    return {
        "start_step": start_step,
        "final_step": step if "step" in locals() else start_step,
        "tokens_seen": tokens_seen,
        "step_losses": step_losses,
        "best_val_loss": best_val_loss,
        "last_val_metrics": last_val_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 5.1 Odia SLM GPU/CPU Training Engine")
    parser.add_argument("--variant", type=str, default="60m", choices=["60m", "100m"])
    parser.add_argument("--bin-dir", type=str, default="data/bin")
    parser.add_argument("--out-dir", type=str, default="checkpoints/60m")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--decay-start-step", type=int, default=8_500)
    parser.add_argument("--context-switch-step", type=int, default=8_000)
    parser.add_argument("--stage1-seq-len", type=int, default=512)
    parser.add_argument("--stage2-seq-len", type=int, default=1024)
    parser.add_argument("--stage1-micro-bsz", type=int, default=32)
    parser.add_argument("--stage2-micro-bsz", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--overfit-single-batch", action="store_true")
    parser.add_argument("--max-session-seconds", type=int, default=40_500)
    args = parser.parse_args()

    cfg = TrainConfig(
        model_variant=args.variant,
        bin_dir=args.bin_dir,
        out_dir=args.out_dir,
        hf_repo_id=args.hf_repo_id,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        decay_start_step=args.decay_start_step,
        context_switch_step=args.context_switch_step,
        stage1_seq_len=args.stage1_seq_len,
        stage2_seq_len=args.stage2_seq_len,
        stage1_micro_batch_size=args.stage1_micro_bsz,
        stage2_micro_batch_size=args.stage2_micro_bsz,
        grad_accum_steps=args.grad_accum,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        dtype=args.dtype,
        compile_model=args.compile,
        overfit_single_batch=args.overfit_single_batch,
        max_session_seconds=args.max_session_seconds,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()

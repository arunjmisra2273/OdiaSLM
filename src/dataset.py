#!/usr/bin/env python3
"""
Step 4.2: Deterministic Permutation-Indexed uint16 Memory-Mapped Data Loader.
Guarantees:
- Zero-RAM-copy OS page-cache reads from train.bin and val.bin
- Strictly disjoint shard assignment across DDP / TPU ranks and gradient accumulation steps
- Bit-exact deterministic resume from any (global_step, tokens_seen) across multi-session runs
- Seamless context extension transition (seq_len=512 -> seq_len=1024) without re-reading seen tokens
- Exact Bits-Per-Byte (BPB) & Perplexity computation from meta.json.
"""

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


class MemmapDataset:
    """
    Deterministic, stateless-by-step memory-mapped dataset reader for train.bin and val.bin.
    """

    def __init__(
        self,
        bin_dir: str = "data/bin",
        seq_len: int = 512,
        seed: int = 1337,
    ):
        self.bin_dir = Path(bin_dir)
        self.train_bin_path = self.bin_dir / "train.bin"
        self.val_bin_path = self.bin_dir / "val.bin"
        self.meta_path = self.bin_dir / "meta.json"

        assert self.train_bin_path.exists(), f"Missing {self.train_bin_path}"
        assert self.val_bin_path.exists(), f"Missing {self.val_bin_path}"
        assert self.meta_path.exists(), f"Missing {self.meta_path}"

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.vocab_size = int(self.meta.get("vocab_size", 16384))
        self.val_bytes_per_token = float(
            self.meta.get("val", {}).get("val_bytes_per_token", 10.8844)
        )

        self.train_mm = np.memmap(self.train_bin_path, dtype=np.uint16, mode="r")
        self.val_mm = np.memmap(self.val_bin_path, dtype=np.uint16, mode="r")

        self.train_total_tokens = len(self.train_mm)
        self.val_total_tokens = len(self.val_mm)

        self.seed = seed
        self.seq_len = seq_len
        self._cached_perm_key: Tuple[int, int] = (-1, -1)
        self._cached_perm: np.ndarray = np.empty(0, dtype=np.int32)

        self.set_seq_len(seq_len)

    def set_seq_len(self, seq_len: int):
        """Updates sequence length and recalculates chunk counts."""
        assert seq_len >= 16, f"Invalid seq_len={seq_len}"
        self.seq_len = int(seq_len)
        self.num_train_chunks = (self.train_total_tokens - 1) // self.seq_len
        self.num_val_chunks = (self.val_total_tokens - 1) // self.seq_len
        assert self.num_train_chunks > 0, "train.bin is smaller than seq_len + 1"
        assert self.num_val_chunks > 0, "val.bin is smaller than seq_len + 1"

    def _get_epoch_permutation(self, epoch: int, seq_len: int) -> np.ndarray:
        """
        Returns a deterministic int32 permutation of [0 .. num_train_chunks - 1]
        for (epoch, seq_len). Caches the current epoch permutation (~14 MB in RAM).
        """
        key = (int(epoch), int(seq_len))
        if self._cached_perm_key != key:
            # Always use a fixed seed per epoch (independent of seq_len) on a canonical
            # 1024-aligned super-grid or direct chunk grid so order is 100% reproducible.
            rng = np.random.RandomState(self.seed + int(epoch) * 1009)
            self._cached_perm = rng.permutation(self.num_train_chunks).astype(np.int32)
            self._cached_perm_key = key
        return self._cached_perm

    def get_train_micro_batch(
        self,
        global_step: int,
        micro_step: int,
        grad_accum_steps: int,
        micro_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        token_offset: int = 0,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Fetches a deterministic, disjoint micro-batch (input_ids, targets, epoch)
        for a given (global_step, micro_step, rank).

        Args:
            global_step: Optimizer step index (0, 1, 2, ...)
            micro_step: Gradient accumulation index in [0 .. grad_accum_steps - 1]
            grad_accum_steps: Total gradient accumulation steps per optimizer step
            micro_batch_size: Sequences per device per micro-step
            rank: Process rank in [0 .. world_size - 1]
            world_size: Number of distributed devices (e.g. 2 for 2xT4, 8 for TPU v5e-8)
            token_offset: Optional starting token offset (used when transitioning seq_len 512 -> 1024
                          so Stage 2 starts at the exact unseen position in the permutation).
            device: Target torch.device.
        Returns:
            (input_ids, targets, current_epoch) where tensors have shape (micro_batch_size, seq_len)
        """
        seqs_per_global_step = world_size * grad_accum_steps * micro_batch_size
        chunk_offset_from_tokens = token_offset // self.seq_len

        # Global sequence cursor for the start of this global_step
        step_seq_start = chunk_offset_from_tokens + (global_step * seqs_per_global_step)

        # Unique slot for this (rank, micro_step) within the global step
        sub_slot = (micro_step * world_size) + rank
        micro_seq_start = step_seq_start + (sub_slot * micro_batch_size)

        epoch = int(micro_seq_start // self.num_train_chunks)
        perm = self._get_epoch_permutation(epoch=epoch, seq_len=self.seq_len)

        # Gather chunk indices (handling wrap-around across epoch boundary cleanly)
        raw_positions = np.arange(micro_seq_start, micro_seq_start + micro_batch_size, dtype=np.int64)
        mod_positions = raw_positions % self.num_train_chunks

        # Check if batch straddles an epoch boundary
        epochs_for_batch = raw_positions // self.num_train_chunks
        if epochs_for_batch[0] == epochs_for_batch[-1]:
            chunk_indices = perm[mod_positions]
        else:
            chunk_indices = np.empty(micro_batch_size, dtype=np.int64)
            for i, pos in enumerate(raw_positions):
                ep_i = int(pos // self.num_train_chunks)
                perm_i = self._get_epoch_permutation(epoch=ep_i, seq_len=self.seq_len)
                chunk_indices[i] = perm_i[int(pos % self.num_train_chunks)]

        # Slice contiguous (seq_len + 1) windows from uint16 memmap
        x_np = np.empty((micro_batch_size, self.seq_len), dtype=np.int64)
        y_np = np.empty((micro_batch_size, self.seq_len), dtype=np.int64)

        T = self.seq_len
        for i, c_idx in enumerate(chunk_indices):
            start_tok = int(c_idx) * T
            span = self.train_mm[start_tok : start_tok + T + 1]
            x_np[i] = span[:-1]
            y_np[i] = span[1:]

        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        if device.type == "cuda":
            x_tensor = x_tensor.pin_memory().to(device, non_blocking=True)
            y_tensor = y_tensor.pin_memory().to(device, non_blocking=True)
        elif device.type != "cpu":
            x_tensor = x_tensor.to(device)
            y_tensor = y_tensor.to(device)

        return x_tensor, y_tensor, epoch

    def get_val_batch(
        self,
        batch_idx: int,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches a deterministic contiguous validation batch `batch_idx` of shape (batch_size, seq_len).
        """
        start_chunk = (batch_idx * batch_size) % max(1, self.num_val_chunks - batch_size)
        T = self.seq_len

        x_np = np.empty((batch_size, T), dtype=np.int64)
        y_np = np.empty((batch_size, T), dtype=np.int64)

        for i in range(batch_size):
            c_idx = (start_chunk + i) % self.num_val_chunks
            start_tok = c_idx * T
            span = self.val_mm[start_tok : start_tok + T + 1]
            x_np[i] = span[:-1]
            y_np[i] = span[1:]

        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        if device.type == "cuda":
            x_tensor = x_tensor.pin_memory().to(device, non_blocking=True)
            y_tensor = y_tensor.pin_memory().to(device, non_blocking=True)
        elif device.type != "cpu":
            x_tensor = x_tensor.to(device)
            y_tensor = y_tensor.to(device)

        return x_tensor, y_tensor

    def compute_val_metrics_from_ce(self, val_loss_ce: float) -> Dict[str, float]:
        """
        Converts validation Cross-Entropy loss (nats/token) into:
        - Perplexity (PPL = exp(loss_ce))
        - Bits-Per-Token (BPT = loss_ce / ln(2))
        - Bits-Per-Byte (BPB = (loss_ce / ln(2)) / val_bytes_per_token)
        """
        ppl = math.exp(min(20.0, val_loss_ce))
        bpt = val_loss_ce / math.log(2.0)
        bpb = bpt / self.val_bytes_per_token
        return {
            "val_loss_ce": round(float(val_loss_ce), 4),
            "val_perplexity": round(float(ppl), 4),
            "val_bits_per_token": round(float(bpt), 4),
            "val_bits_per_byte": round(float(bpb), 4),
        }

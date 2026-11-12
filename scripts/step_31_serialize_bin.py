#!/usr/bin/env python3
"""
Step 3.1: Multi-Core Streaming uint16 Binary Serializer (train.bin, val.bin, meta.json).
Features:
- Multi-process C++ SentencePiece tokenization across all CPU cores
- 32-way deterministic bucket interleaving so v2, Sangraha, and FineWeb-Edu
  are uniformly mixed even inside individual 512/1024-token windows
- Zero-RAM-spike streaming disk writes to /mnt/d/odia_llm_data/bin (symlinked to data/bin)
- Exact UTF-8 byte & token accounting in meta.json for Bits-Per-Byte (BPB) evaluation.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

TRAIN_JSONL = Path("/mnt/d/odia_llm_data/processed/train_docs.jsonl")
VAL_JSONL = Path("/mnt/d/odia_llm_data/processed/val_docs.jsonl")
TOK_MODEL = Path("artifacts/tokenizer/tokenizer.model")
REPORT_PATH = Path("artifacts/step_31_bin_serialization_report.json")

BOS_ID = 1
EOS_ID = 2

# Worker-global SentencePiece instance
_WORKER_SP = None


def _init_worker(tok_model_path: str):
    global _WORKER_SP
    _WORKER_SP = spm.SentencePieceProcessor(model_file=tok_model_path)


def _encode_jsonl_batch(lines: list) -> tuple:
    """
    Worker function: parses a batch of JSONL lines, encodes text into
    [BOS_ID, ...tokens..., EOS_ID] as uint16, and returns:
    (list_of_uint16_byte_buffers, total_tokens, total_utf8_bytes, total_words, doc_count)
    """
    global _WORKER_SP
    encoded_Buffers = []
    total_tokens = 0
    total_bytes = 0
    total_words = 0
    doc_count = 0

    for line in lines:
        if not line:
            continue
        rec = json.loads(line)
        text = rec["text"]
        ids = _WORKER_SP.encode(text, out_type=int)
        # Pack with BOS (1) and EOS (2)
        packed = [BOS_ID]
        packed.extend(ids)
        packed.append(EOS_ID)

        arr = np.asarray(packed, dtype=np.uint16)
        encoded_Buffers.append(arr.tobytes())

        total_tokens += len(packed)
        total_bytes += len(text.encode("utf-8"))
        total_words += rec.get("words", len(text.split()))
        doc_count += 1

    return encoded_Buffers, total_tokens, total_bytes, total_words, doc_count


def line_batch_generator(jsonl_path: Path, batch_size: int = 2000):
    """Yields lists of raw lines from a JSONL file."""
    batch = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def serialize_validation_split(val_jsonl: Path, val_bin_path: Path, num_workers: int) -> dict:
    print(f"\n[1/3] Serializing validation split ({val_jsonl}) -> {val_bin_path} ...")
    total_tokens = 0
    total_bytes = 0
    total_words = 0
    total_docs = 0

    with open(val_bin_path, "wb") as fout:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(str(TOK_MODEL),)) as pool:
            batches = line_batch_generator(val_jsonl, batch_size=1000)
            for bufs, n_tok, n_bytes, n_words, n_docs in tqdm(
                pool.imap(_encode_jsonl_batch, batches, chunksize=2),
                total=48,
                desc="Tokenizing val.bin",
            ):
                for b in bufs:
                    fout.write(b)
                total_tokens += n_tok
                total_bytes += n_bytes
                total_words += n_words
                total_docs += n_docs

    file_size_bytes = val_bin_path.stat().st_size
    assert file_size_bytes == total_tokens * 2, f"Size mismatch: {file_size_bytes} != {total_tokens * 2}"

    return {
        "val_bin_path": str(val_bin_path),
        "val_docs": total_docs,
        "val_words": total_words,
        "val_tokens": total_tokens,
        "val_utf8_bytes": total_bytes,
        "val_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "val_bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
    }


def serialize_training_split_interleaved(
    train_jsonl: Path,
    train_bin_path: Path,
    temp_shard_dir: Path,
    num_shards: int,
    num_workers: int,
) -> dict:
    """
    Tokenizes train_docs.jsonl across CPU workers, routes each document's uint16 token buffer
    pseudo-randomly into 1 of `num_shards` temporary shard files (mixing v2, Sangraha, and
    FineWeb-Edu uniformly), and then concatenates the shards into a contiguous train.bin.
    """
    print(f"\n[2/3] Tokenizing & 32-way interleaving {train_jsonl} across {num_workers} CPU cores ...")
    temp_shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [temp_shard_dir / f"shard_{i:02d}.bin" for i in range(num_shards)]
    shard_files = [open(sp, "wb") for sp in shard_paths]

    rng = random.Random(1337)
    total_tokens = 0
    total_bytes = 0
    total_words = 0
    total_docs = 0

    # 9,571,297 docs / 2500 docs per batch = ~3829 batches
    est_batches = (9_571_297 // 2500) + 1

    try:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(str(TOK_MODEL),)) as pool:
            batches = line_batch_generator(train_jsonl, batch_size=2500)
            for bufs, n_tok, n_bytes, n_words, n_docs in tqdm(
                pool.imap(_encode_jsonl_batch, batches, chunksize=2),
                total=est_batches,
                desc="Tokenizing train shards",
            ):
                # Route each document into a random shard file to interleave domains
                for b in bufs:
                    s_idx = rng.randrange(num_shards)
                    shard_files[s_idx].write(b)

                total_tokens += n_tok
                total_bytes += n_bytes
                total_words += n_words
                total_docs += n_docs
    finally:
        for sf in shard_files:
            sf.close()

    print(f"\n[3/3] Concatenating {num_shards} interleaved shards into contiguous {train_bin_path} ...")
    with open(train_bin_path, "wb") as fout:
        for sp in tqdm(shard_paths, desc="Merging shards -> train.bin"):
            with open(sp, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)  # 16 MB buffer copy
            sp.unlink()  # Immediately free temporary shard disk space

    try:
        temp_shard_dir.rmdir()
    except OSError:
        pass

    file_size_bytes = train_bin_path.stat().st_size
    assert file_size_bytes == total_tokens * 2, f"Size mismatch: {file_size_bytes} != {total_tokens * 2}"

    return {
        "train_bin_path": str(train_bin_path),
        "train_docs": total_docs,
        "train_words": total_words,
        "train_tokens": total_tokens,
        "train_utf8_bytes": total_bytes,
        "train_size_gb": round(file_size_bytes / (1024 ** 3), 3),
        "train_bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
    }


def verify_memmap_integrity(bin_path: Path, expected_tokens: int, vocab_size: int = 16384) -> dict:
    """Opens the binary file with np.memmap and verifies bounds,BOS/EOS presence, and dtype."""
    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    assert len(mm) == expected_tokens, f"Memmap length {len(mm)} != {expected_tokens}"
    assert int(mm[0]) == BOS_ID, f"First token in {bin_path.name} is {mm[0]}, expected BOS_ID ({BOS_ID})"
    assert int(mm[-1]) == EOS_ID, f"Last token in {bin_path.name} is {mm[-1]}, expected EOS_ID ({EOS_ID})"

    # Sample 1,000,000 tokens from head, middle, and tail to verify max token ID < vocab_size
    n = len(mm)
    head_max = int(np.max(mm[: min(n, 1_000_000)]))
    mid_max = int(np.max(mm[n // 2 : min(n, n // 2 + 1_000_000)]))
    tail_max = int(np.max(mm[max(0, n - 1_000_000) :]))
    overall_max = max(head_max, mid_max, tail_max)
    assert overall_max < vocab_size, f"Found out-of-bounds token ID {overall_max} >= {vocab_size}"

    return {
        "memmap_verified": True,
        "first_token_id": int(mm[0]),
        "last_token_id": int(mm[-1]),
        "max_token_id_sampled": overall_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 3.1 Binary Dataset Serializer")
    parser.add_argument("--storage-dir", type=str, default="/mnt/d/odia_llm_data", help="Disk path for binary outputs")
    parser.add_argument("--num-shards", type=int, default=32, help="Number of interleaving shard buckets")
    parser.add_argument("--num-workers", type=int, default=max(4, os.cpu_count() or 4), help="CPU worker processes")
    args = parser.parse_args()

    t0 = time.time()
    assert TOK_MODEL.exists(), f"Missing tokenizer at {TOK_MODEL}"
    assert TRAIN_JSONL.exists(), f"Missing {TRAIN_JSONL}"
    assert VAL_JSONL.exists(), f"Missing {VAL_JSONL}"

    bin_root = Path(args.storage_dir) / "bin"
    bin_root.mkdir(parents=True, exist_ok=True)
    temp_shards = bin_root / "_tmp_shards"

    # Symlink ~/odia-llm/data/bin -> /mnt/d/odia_llm_data/bin
    local_bin = Path("data/bin")
    if local_bin.exists() and not local_bin.is_symlink():
        try:
            local_bin.rmdir()
        except OSError:
            pass
    if not local_bin.exists():
        local_bin.symlink_to(bin_root, target_is_directory=True)

    val_bin_path = bin_root / "val.bin"
    train_bin_path = bin_root / "train.bin"
    meta_path = bin_root / "meta.json"

    # Also copy tokenizer.model into bin_root so /mnt/d/odia_llm_data/bin is a self-contained Kaggle upload bundle
    shutil.copy2(TOK_MODEL, bin_root / "tokenizer.model")
    tok_sha256 = hashlib.sha256(TOK_MODEL.read_bytes()).hexdigest()

    val_stats = serialize_validation_split(VAL_JSONL, val_bin_path, num_workers=args.num_workers)
    val_check = verify_memmap_integrity(val_bin_path, val_stats["val_tokens"])

    train_stats = serialize_training_split_interleaved(
        TRAIN_JSONL,
        train_bin_path,
        temp_shard_dir=temp_shards,
        num_shards=args.num_shards,
        num_workers=args.num_workers,
    )
    train_check = verify_memmap_integrity(train_bin_path, train_stats["train_tokens"])

    elapsed = round(time.time() - t0, 2)

    meta = {
        "dataset_name": "odia-slm-pretrain-v1",
        "dtype": "uint16",
        "byte_order": "little",
        "vocab_size": 16384,
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
        "pad_token_id": 0,
        "unk_token_id": 3,
        "tokenizer_sha256": tok_sha256,
        "interleaved_shards": args.num_shards,
        "train": {**train_stats, **train_check},
        "val": {**val_stats, **val_check},
        "total_tokens": train_stats["train_tokens"] + val_stats["val_tokens"],
        "elapsed_seconds": elapsed,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== STEP 3.1 VERIFICATION SUMMARY ===")
    print(f"Bundle Directory        : {bin_root} (Symlinked to data/bin)")
    print(f"train.bin               : {train_stats['train_tokens']:,} tokens ({train_stats['train_size_gb']} GB) | Docs: {train_stats['train_docs']:,}")
    print(f"val.bin                 : {val_stats['val_tokens']:,} tokens ({val_stats['val_size_mb']} MB) | Docs: {val_stats['val_docs']:,}")
    print(f"Total Packed Tokens     : {meta['total_tokens']:,} tokens (uint16 little-endian)")
    print(f"Val Bytes-Per-Token     : {val_stats['val_bytes_per_token']} UTF-8 bytes/token (stored in meta.json for exact BPB)")
    print(f"Memmap Integrity Gate   : train={train_check['memmap_verified']} (max_id={train_check['max_token_id_sampled']}) | val={val_check['memmap_verified']} (max_id={val_check['max_token_id_sampled']})")
    print(f"Self-Contained Bundle   : train.bin + val.bin + meta.json + tokenizer.model ready in {bin_root}")
    print(f"Total Runtime           : {elapsed}s ({round(elapsed/60, 1)} mins)")
    print("=====================================")


if __name__ == "__main__":
    main()


# python3 scripts/step_31_serialize_bin.py
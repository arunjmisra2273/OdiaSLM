#!/usr/bin/env python3
"""
Step 2.1: Balanced Tokenizer Corpus Sampling & 16,384-Vocab Odia SentencePiece BPE Training.
Enforces NFC identity normalization, 0.9999 character coverage, byte fallback,
digit splitting, and locked special/chat control token IDs (0..7).
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm

TRAIN_JSONL = Path("/mnt/d/odia_llm_data/processed/train_docs.jsonl")
BALANCED_TXT = Path("/mnt/d/odia_llm_data/processed/tokenizer_balanced_input.txt")
LOCAL_TOK_DIR = Path("artifacts/tokenizer")
DISK_TOK_DIR = Path("/mnt/d/odia_llm_data/tokenizer")
REPORT_PATH = Path("artifacts/step_21_tokenizer_train_report.json")


def build_balanced_tokenizer_input(
    train_jsonl: Path,
    output_txt: Path,
    target_mb: int = 500,
) -> dict:
    """
    Streams train_docs.jsonl to construct a stratified ~500 MB plain-text corpus
    (~94% Odia across v2 + Sangraha, ~6% English from FineWeb-Edu) so SentencePiece
    trains fast in <4 GB RAM while seeing all domains including the English anchor.
    """
    print(f"[1/3] Building stratified {target_mb} MB tokenizer input from {train_jsonl} ...")
    rng = random.Random(42)
    target_bytes = target_mb * 1024 * 1024
    max_en_bytes = int(target_bytes * 0.06)  # 6% English anchor budget

    odia_bytes = 0
    en_bytes = 0
    docs_sampled = Counter()

    # Since train_docs.jsonl has ~9.57M docs (~18 GB), sampling ~3.5% of Odia docs
    # and ~40% of English docs yields ~500 MB across the entire file.
    with open(train_jsonl, "r", encoding="utf-8") as fin, \
         open(output_txt, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=9_571_297, desc="Sampling stratified text"):
            if '"anchor:fineweb_edu_en"' in line:
                if en_bytes < max_en_bytes and rng.random() < 0.35:
                    rec = json.loads(line)
                    txt = rec["text"].strip() + "\n"
                    b_len = len(txt.encode("utf-8"))
                    fout.write(txt)
                    en_bytes += b_len
                    docs_sampled["english_anchor"] += 1
            else:
                if odia_bytes < (target_bytes - max_en_bytes) and rng.random() < 0.035:
                    rec = json.loads(line)
                    txt = rec["text"].strip() + "\n"
                    b_len = len(txt.encode("utf-8"))
                    fout.write(txt)
                    odia_bytes += b_len
                    src_prefix = rec.get("source", "odia").split(":")[0]
                    docs_sampled[src_prefix] += 1

    total_b = odia_bytes + en_bytes
    stats = {
        "balanced_txt_path": str(output_txt),
        "total_mb": round(total_b / (1024 * 1024), 2),
        "odia_mb": round(odia_bytes / (1024 * 1024), 2),
        "english_mb": round(en_bytes / (1024 * 1024), 2),
        "english_pct": round(100.0 * en_bytes / max(1, total_b), 2),
        "docs_sampled_by_group": dict(docs_sampled),
    }
    print(f"      Stratified sample ready: {stats['total_mb']} MB (Odia: {stats['odia_mb']} MB, EN: {stats['english_mb']} MB [{stats['english_pct']}%])")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Step 2.1 Train Odia SentencePiece Tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16384, help="Exact vocabulary size (multiple of 128)")
    parser.add_argument("--sample-mb", type=int, default=500, help="Stratified sample size in MB for BPE training")
    parser.add_argument("--num-threads", type=int, default=max(4, os.cpu_count() or 4), help="CPU threads")
    args = parser.parse_args()

    t0 = time.time()
    LOCAL_TOK_DIR.mkdir(parents=True, exist_ok=True)
    DISK_TOK_DIR.mkdir(parents=True, exist_ok=True)

    sample_stats = build_balanced_tokenizer_input(
        train_jsonl=TRAIN_JSONL,
        output_txt=BALANCED_TXT,
        target_mb=args.sample_mb,
    )

    model_prefix = str(LOCAL_TOK_DIR / "odia_bpe_16k")
    control_symbols = ["<|system|>", "<|user|>", "<|assistant|>", "<|end_of_turn|>"]

    print(f"\n[2/3] Training SentencePiece BPE Tokenizer (vocab_size={args.vocab_size}) ...")
    spm.SentencePieceTrainer.train(
        input=str(BALANCED_TXT),
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=0.9999,
        normalization_rule_name="identity",  # Mandatory: preserve pre-normalized NFC & ZWNJ
        byte_fallback=True,                  # Mandatory: 256 byte tokens -> zero <unk>
        split_digits=True,                   # Mandatory: split 0-9 and Odia ୦-୯ into single digits
        allow_whitespace_only_pieces=True,
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        max_sentence_length=16384,
        input_sentence_size=2_500_000,
        shuffle_input_sentence=True,
        num_threads=args.num_threads,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<s>",
        eos_piece="</s>",
        unk_piece="<unk>",
        control_symbols=control_symbols,
    )

    model_file = Path(f"{model_prefix}.model")
    vocab_file = Path(f"{model_prefix}.vocab")

    # Copy to /mnt/d/odia_llm_data/tokenizer and also create canonical tokenizer.model symlink/copy
    shutil.copy2(model_file, LOCAL_TOK_DIR / "tokenizer.model")
    shutil.copy2(vocab_file, LOCAL_TOK_DIR / "tokenizer.vocab")
    shutil.copy2(model_file, DISK_TOK_DIR / "tokenizer.model")
    shutil.copy2(vocab_file, DISK_TOK_DIR / "tokenizer.vocab")

    # Inspect trained tokenizer
    print("\n[3/3] Auditing trained tokenizer vocabulary & special IDs ...")
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    actual_vocab_size = sp.get_piece_size()

    special_id_map = {
        id_val: sp.id_to_piece(id_val) for id_val in range(8)
    }

    # Count Odia script pieces, byte pieces, and ASCII/Latin pieces in vocabulary
    odia_pieces = 0
    byte_pieces = 0
    latin_pieces = 0
    single_digit_pieces = 0

    for idx in range(actual_vocab_size):
        piece = sp.id_to_piece(idx)
        if piece.startswith("<0x") and piece.endswith(">"):
            byte_pieces += 1
        elif any(0x0B00 <= ord(ch) <= 0x0B7F for ch in piece):
            odia_pieces += 1
        elif any(("a" <= ch.lower() <= "z") for ch in piece) and not piece.startswith("<"):
            latin_pieces += 1
        if len(piece) == 1 and piece.isdigit():
            single_digit_pieces += 1

    sha256_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    elapsed = round(time.time() - t0, 2)

    report = {
        "model_path": str(LOCAL_TOK_DIR / "tokenizer.model"),
        "disk_backup_path": str(DISK_TOK_DIR / "tokenizer.model"),
        "model_sha256": sha256_hash,
        "vocab_size": actual_vocab_size,
        "special_tokens_0_to_7": special_id_map,
        "vocab_composition": {
            "odia_script_pieces": odia_pieces,
            "odia_script_pct": round(100.0 * odia_pieces / actual_vocab_size, 2),
            "latin_pieces": latin_pieces,
            "byte_fallback_pieces": byte_pieces,
            "single_digit_pieces": single_digit_pieces,
        },
        "input_sample_stats": sample_stats,
        "elapsed_seconds": elapsed,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== STEP 2.1 VERIFICATION SUMMARY ===")
    print(f"Tokenizer Model Saved   : {LOCAL_TOK_DIR / 'tokenizer.model'} (SHA256: {sha256_hash[:16]}...)")
    print(f"Exact Vocab Size        : {actual_vocab_size:,} (Target: {args.vocab_size:,})")
    print(f"Locked IDs (0..7)       : {special_id_map}")
    print(f"Odia Script Subwords    : {odia_pieces:,} ({report['vocab_composition']['odia_script_pct']}%)")
    print(f"Latin / English Subwords: {latin_pieces:,} | Byte Fallback Pieces: {byte_pieces} | Single Digits: {single_digit_pieces}")
    print(f"Stratified Input Used   : {sample_stats['total_mb']} MB (Odia: {sample_stats['odia_mb']} MB, EN: {sample_stats['english_mb']} MB)")
    print(f"Total Runtime           : {elapsed}s ({round(elapsed/60, 1)} mins)")
    print("=====================================")


if __name__ == "__main__":
    main()

# python3 scripts/step_21_train_tokenizer.py
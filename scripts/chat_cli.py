#!/usr/bin/env python3
"""
Interactive Local Terminal Chat CLI for OdiaSLM-100M-Instruct.
Loads artifacts/hf_export/odia-slm-100m-instruct locally on CPU (or downloads from HF Hub)
and lets you chat interactively in Odia at ~33 tokens/sec.
Usage:
  python3 scripts/chat_cli.py
Author: Arun Misra | AI Researcher, Bhubaneswar, Odisha, India
"""

import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM

os.environ["HF_HUB_DISABLE_XET"] = "1"
torch.set_num_threads(min(4, os.cpu_count() or 4))

LOCAL_DIR = Path("artifacts/hf_export/odia-slm-100m-instruct")
HF_REPO = "arunjmisra2273/OdiaSLM-100M-Instruct"


def main():
    model_source = str(LOCAL_DIR) if (LOCAL_DIR / "model.safetensors").exists() else HF_REPO
    tok_path = LOCAL_DIR / "tokenizer.model"
    if not tok_path.exists():
        tok_path = Path("artifacts/tokenizer/tokenizer.model")

    print(f"Loading OdiaSLM-100M-Instruct from '{model_source}' on CPU...")
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))
    model = AutoModelForCausalLM.from_pretrained(model_source, dtype=torch.float32, trust_remote_code=False).eval()
    print("Model ready! Type any question in Odia (or type 'exit' / 'quit' to leave).\n")
    print("=" * 72)

    while True:
        try:
            q = input("\n🟢 ଆପଣ (User): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nଧନ୍ୟବାଦ! (Exiting chat)")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("ଧନ୍ୟବାଦ! (Exiting chat)")
            break

        formatted = f"<|user|>\n{q}\n<|assistant|>\n"
        in_ids = torch.tensor([[sp.bos_id()] + sp.encode(formatted, out_type=int)], dtype=torch.long)

        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                in_ids,
                max_new_tokens=110,
                do_sample=False,
                repetition_penalty=1.18,
                no_repeat_ngram_size=4,
                eos_token_id=sp.eos_id(),
                pad_token_id=sp.pad_id(),
            )
        elapsed = time.time() - t0
        new_tokens = out_ids[0, in_ids.size(1) :].tolist()
        if sp.eos_id() in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(sp.eos_id())]
        ans = sp.decode(new_tokens).strip()
        tps = len(new_tokens) / max(elapsed, 1e-4)

        print(f"🏛️ OdiaSLM-100M-Instruct: {ans}")
        print(f"   [{len(new_tokens)} tokens | {elapsed:.2f}s | {tps:.1f} tok/s on CPU]")


if __name__ == "__main__":
    main()

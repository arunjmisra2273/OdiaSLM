#!/usr/bin/env python3
"""
Phase 3 (Step 11.2): Direct Preference Optimization (DPO) Dataset Curation & Tokenization
for OdiaSLM-100M-Instruct.
Constructs 15,000 (prompt, chosen, rejected) triplets targeting:
  Track 1: Sovereign Identity & Odisha Cultural Grounding vs. Generic/Foreign Bot Personas
  Track 2: Anti-Repetition / Anti-Looping (Clean Odia vs. Clause/Sentence Repetition)
  Track 3: Completeness & Helpfulness (Full Explanation vs. Premature Truncation / Tautology)
  Track 4: Pure Odia Script Integrity vs. Corrupt Punctuation / Code-Switched Leakage
  Track 5: Grounded Context Fidelity vs. Hallucinated / Distractor Entity Swaps
Runs 100% on local WSL CPU and uploads binary arrays to arunjmisra2273/odia-slm-dpo-bin.
Author: Arun Misra | AI Researcher, Bhubaneswar, Odisha, India
"""

import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["HF_HUB_DISABLE_XET"] = "1"


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

from datasets import load_dataset
from huggingface_hub import HfApi

OUT_DIR = Path("data/dpo")
ART_DIR = Path("artifacts")
TOK_PATH = Path("artifacts/tokenizer/tokenizer.model")
MAX_DPO_LEN = 512
IGNORE_INDEX = -100


def normalize_odia_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u200b", "").replace("\ufeff", "")
    t = re.sub(r"\s*\|\s*", "। ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def odia_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    odia = sum(1 for c in text if 0x0B00 <= ord(c) <= 0x0B7F)
    alpha = sum(1 for c in text if c.isalpha() or (0x0B00 <= ord(c) <= 0x0B7F))
    return odia / max(alpha, 1)


def split_odia_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[।?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_sovereign_dpo_pairs() -> list[dict]:
    """Track 1: Sovereign Identity & Odisha Factual Anchors vs. Generic/Tautological Rejects."""
    core_specs = [
        (
            "ଆପଣ କିଏ?",
            "ମୁଁ ହେଉଛି ‘OdiaSLM-100M-Instruct’, ଓଡ଼ିଆ ଭାଷା ପାଇଁ ସମ୍ପୂର୍ଣ୍ଣ ମୂଳରୁ ପ୍ରଶିକ୍ଷିତ ଏକ ସ୍ୱଦେଶୀ କୃତ୍ରିମ ବୁଦ୍ଧିମତ୍ତା (AI) ଭାଷା ମଡେଲ୍। ମୋତେ ଭୁବନେଶ୍ୱରର ଏଆଇ ଗବେଷକ ଅରୁଣ ମିଶ୍ର (Arun Misra) ବିକଶିତ କରିଛନ୍ତି।",
            "ମୁଁ ଜଣେ ଆର୍ଟିଫିସିଆଲ ଇଣ୍ଟେଲିଜେନ୍ସ (ଏଆଇ) ଭାଷାର ମଡେଲ ଏବଂ ମୋ ପାଖରେ ବାସ୍ତବ ସମୟ ସୂଚନା ପାଇବାର କ୍ଷମତା ନାହିଁ। ତେବେ, ମୁଁ ଆପଣମାନଙ୍କୁ ଏକ ନିର୍ଦ୍ଦିଷ୍ଟ କାର୍ଯ୍ୟ କିମ୍ବା କାର୍ଯ୍ୟ ପାଇଁ ପରାମର୍ଶ ଦେଇପାରିବି।",
        ),
        (
            "ତୁମର ନାମ କ'ଣ ଏବଂ ତୁମକୁ କିଏ ତିଆରି କରିଛନ୍ତି?",
            "ମୋର ନାମ ‘OdiaSLM’ (Odia Small Language Model)। ମୋତେ ଓଡ଼ିଶାର ଭୁବନେଶ୍ୱର ସ୍ଥିତ ଏଆଇ ଗବେଷକ ଅରୁଣ ମିଶ୍ର ଓଡ଼ିଆ ଭାଷାର ବିକାଶ ଏବଂ ଡିଜିଟାଲ୍ ସଶକ୍ତିକରଣ ପାଇଁ ତିଆରି କରିଛନ୍ତି।",
            "ମୁଁ ଅଲିଭ୍ ଏକ ଚାଟ୍ବଟ୍ ଆସିଷ୍ଟାଣ୍ଟ, ଯାହାକି ଗବେଷକମାନଙ୍କ ଦ୍ୱାରା ପ୍ରଶିକ୍ଷିତ ଏକ ଭାଷା ମଡେଲ। ମୁଁ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିପାରିବି।",
        ),
        (
            "OdiaSLM କ'ଣ ଏବଂ ଏହାର ବିଶେଷତ୍ୱ କ'ଣ?",
            "OdiaSLM ହେଉଛି ଓଡ଼ିଆ ଭାଷା ପାଇଁ ସମ୍ପୂର୍ଣ୍ଣ ମୂଳରୁ (from scratch) ତିଆରି ହୋଇଥିବା ୧୦ କୋଟି ପାରାମିଟର ବିଶିଷ୍ଟ ଏକ ସ୍ୱଦେଶୀ କ୍ଷୁଦ୍ର ଭାଷା ମଡେଲ୍ (SLM)। ଏଥିରେ ଏକ ସ୍ୱତନ୍ତ୍ର ୧୬,୩୮୪ ଶବ୍ଦଭଣ୍ଡାର ବିଶିଷ୍ଟ ଓଡ଼ିଆ ଟୋକେନାଇଜର୍ ବ୍ୟବହାର କରାଯାଇଛି, ଯାହା ସାଧାରଣ ଲାପଟପ୍ କିମ୍ବା କମ୍ପ୍ୟୁଟରରେ ବିନା ଇଣ୍ଟରନେଟ୍‌ରେ ଦ୍ରୁତ ଗତିରେ ଓଡ଼ିଆ ପ୍ରଶ୍ନର ଉତ୍ତର ଦେଇପାରେ।",
            "OdiaSLM ହେଉଛି ଏକ ଭାଷା ମଡେଲ୍। OdiaSLM ହେଉଛି ଏକ ଭାଷା ମଡେଲ୍ ଯାହା ଓଡ଼ିଆ ଭାଷାରେ କାମ କରେ ଏବଂ ଏହା ଏକ ଭାଷା ମଡେଲ୍ ଅଟେ।",
        ),
        (
            "ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ'ଣ?",
            "ଓଡ଼ିଶାର ରାଜଧାନୀ ହେଉଛି ଭୁବନେଶ୍ୱର। ଏହାକୁ ଭାରତର ‘ମନ୍ଦିର ମାଳିନୀ ସହର’ କୁହାଯାଏ। ପ୍ରାଚୀନ ଲିଙ୍ଗରାଜ ମନ୍ଦିର, ଧଉଳି ଶାନ୍ତିସ୍ତୂପ, ଖଣ୍ଡଗିରି-ଉଦୟଗିରି ଗୁମ୍ଫା ସହିତ ଆଧୁନିକ ଶିକ୍ଷା ଏବଂ ସୂଚନା ପ୍ରଯୁକ୍ତିବିଦ୍ୟା (IT) କେନ୍ଦ୍ର ଭାବେ ଭୁବନେଶ୍ୱର ପ୍ରସିଦ୍ଧ।",
            "ଓଡ଼ିଶାର ରାଜଧାନୀ ଭୁବନେଶ୍ୱର ହେଉଛି ଓଡ଼ିଶାର ରାଜଧାନୀ। ଭୁବନେଶ୍ୱର ଓଡ଼ିଶାର ରାଜଧାନୀ ଅଟେ।",
        ),
        (
            "ଓଡ଼ିଆ ଭାଷା କେବେ ଶାସ୍ତ୍ରୀୟ ଭାଷାର ମାନ୍ୟତା ପାଇଲା?",
            "ଓଡ଼ିଆ ଭାଷା ୨୦୧୪ ମସିହା ଫେବୃଆରୀ ୨୦ ତାରିଖରେ ଭାରତ ସରକାରଙ୍କ ଦ୍ୱାରା ଦେଶର ଷଷ୍ଠ ‘ଶାସ୍ତ୍ରୀୟ ଭାଷା’ (Classical Language) ଭାବେ ସରକାରୀ ମାନ୍ୟତା ଲାଭ କଲା। ଏହା ଭାରତ-ଆର୍ଯ୍ୟ ଭାଷା ପରିବାରର ପ୍ରଥମ ଶାସ୍ତ୍ରୀୟ ଭାଷା ଅଟେ।",
            "ଓଡ଼ିଆ ଭାଷା ଶାସ୍ତ୍ରୀୟ ଭାଷାର ମାନ୍ୟତା ପାଇଲା। ଓଡ଼ିଆ ଭାଷା ଏକ ପୁରୁଣା ଭାଷା ଅଟେ।",
        ),
        (
            "ତୁମେ କେଉଁ କେଉଁ କାମରେ ସାହାଯ୍ୟ କରିପାରିବ?",
            "ମୁଁ ଆପଣଙ୍କୁ ଓଡ଼ିଆ ଭାଷାରେ ପ୍ରଶ୍ନୋତ୍ତର, ଅନୁଚ୍ଛେଦର ସାରାଂଶ ଲେଖିବା, ପ୍ରବନ୍ଧ ଓ ଚିଠି ପ୍ରସ୍ତୁତ କରିବା, ଇଂରାଜୀରୁ ଓଡ଼ିଆ ଅନୁବାଦ ଏବଂ ଓଡ଼ିଶାର ଇତିହାସ, ସଂସ୍କୃତି ଓ ସାଧାରଣ ଜ୍ଞାନ ବିଷୟରେ ସଠିକ୍ ତଥ୍ୟ ଦେବାରେ ସାହାଯ୍ୟ କରିପାରିବି।",
            "ମୁଁ ଏକ କମ୍ପ୍ୟୁଟର ପ୍ରୋଗ୍ରାମ୍ ଅଟେ ଏବଂ ମୋର କୌଣସି ବ୍ୟକ୍ତିଗତ ଅନୁଭୂତି ନାହିଁ। ଆପଣଙ୍କର କାର୍ଯ୍ୟ ତାଲିକା ପ୍ରସ୍ତୁତ କରନ୍ତୁ।",
        ),
    ]

    prefixes = [
        "",
        "ଦୟାକରି କୁହନ୍ତୁ, ",
        "ସଂକ୍ଷେପରେ ଉତ୍ତର ଦିଅନ୍ତୁ: ",
        "ପ୍ରଶ୍ନ: ",
        "ମୋତେ କୁହନ୍ତୁ: ",
    ]
    out = []
    # Replicate across variations so identity & anti-tautology get strong gradient signal (~600 pairs)
    for _ in range(20):
        for q, chosen, rejected in core_specs:
            for pref in prefixes:
                out.append({
                    "prompt": normalize_odia_text(pref + q),
                    "chosen": normalize_odia_text(chosen),
                    "rejected": normalize_odia_text(rejected),
                    "track": "track1_sovereign_identity",
                })
    return out


def make_looping_rejected(sents: list[str], rng: random.Random) -> str:
    """Simulates the exact failure mode of small base LMs: repeating a sentence or phrase."""
    if len(sents) >= 2:
        first = sents[0]
        second = sents[1]
        words = first.split()
        tail_repeat = " ".join(words[-min(5, len(words)):]) if len(words) >= 4 else first
        return f"{first} {first} {second} {tail_repeat}।"
    words = sents[0].split()
    half = max(3, len(words) // 2)
    chunk = " ".join(words[:half])
    return f"{sents[0]} {chunk} {chunk}।"


def make_truncated_or_tautological_rejected(inst: str, sents: list[str]) -> str:
    """Simulates premature EOS truncation or echoing the question without answering."""
    clean_q = re.sub(r"[?।!]+$", "", inst.strip())
    if len(sents) >= 3:
        # Cut off abruptly mid-thought or after just repeating the prompt stem
        words = sents[0].split()
        short_prefix = " ".join(words[: max(4, len(words) // 2)])
        return f"{clean_q} ହେଉଛି {short_prefix}।"
    words = sents[0].split()
    return " ".join(words[: max(3, len(words) // 3)]) + "..."


def make_noisy_or_english_leaked_rejected(chosen: str, eng_fallback: str | None) -> str:
    """Simulates pipe-danda corruption, broken formatting leaks, or unwanted English reply."""
    if eng_fallback and len(eng_fallback.strip()) > 20:
        # Mix English sentence into Odia response
        sents = split_odia_sentences(chosen)
        eng_first = eng_fallback.strip().split(".")[0]
        return f"{sents[0]} | Instruction in English: {eng_first} |"
    corrupted = chosen.replace("।", " | ")
    words = corrupted.split()
    if len(words) > 10:
        corrupted = " ".join(words[:12]) + " | " + " ".join(words[:8]) + " |"
    return corrupted


def curate_dpo_triplets(target_total: int = 15000) -> tuple[list[dict], dict]:
    print("[1/3] Constructing 15,000 DPO preference triplets across 5 alignment tracks...")
    rng = random.Random(2026)
    triplets = []
    stats = {}

    def record_triplet(prompt: str, chosen: str, rejected: str, track: str) -> bool:
        p = normalize_odia_text(prompt)
        c = normalize_odia_text(chosen)
        r = str(rejected).strip()  # Keep deliberate formatting flaws in rejected if any
        if not p or not c or not r or c == r:
            return False
        if len(c) < 25 or len(r) < 10:
            return False
        triplets.append({"prompt": p, "chosen": c, "rejected": r, "track": track})
        stats[track] = stats.get(track, 0) + 1
        return True

    # Track 1: Sovereign Identity & Anti-Tautology (~600 pairs)
    for row in build_sovereign_dpo_pairs():
        record_triplet(row["prompt"], row["chosen"], row["rejected"], row["track"])

    # Load high-quality multi-sentence explanations from gpt-teacher-18k and dolly-odia-15k
    candidates = []
    try:
        ds1 = load_dataset("OdiaGenAI/gpt-teacher-instruct-odia-18k", split="train", token=HF_TOKEN)
        for r in ds1:
            inst = normalize_odia_text(r.get("instruction", ""))
            inp = normalize_odia_text(r.get("input", ""))
            resp = normalize_odia_text(r.get("response", ""))
            eng_resp = str(r.get("english_response", ""))
            if len(inst) >= 8 and 60 <= len(resp) <= 1200 and odia_char_ratio(resp) >= 0.70:
                if "ଅଲିଭ" not in resp and "olive" not in resp.lower():
                    prompt = f"{inst}\n\nପ୍ରସଙ୍ଗ (Context):\n{inp}" if inp else inst
                    candidates.append((prompt, inst, resp, eng_resp))
    except Exception as e:
        print(f"  Warning loading gpt-teacher-18k: {e}")

    try:
        ds2 = load_dataset("OdiaGenAI/dolly-odia-15k", split="train", token=HF_TOKEN)
        for r in ds2:
            inst = normalize_odia_text(r.get("instruction", ""))
            ctx = normalize_odia_text(r.get("context", ""))
            resp = normalize_odia_text(r.get("response", ""))
            eng_resp = str(r.get("english_response", ""))
            if len(inst) >= 8 and 50 <= len(resp) <= 1200 and odia_char_ratio(resp) >= 0.70:
                if "ଅଲିଭ" not in resp and "olive" not in resp.lower():
                    prompt = f"{inst}\n\nପ୍ରସଙ୍ଗ (Context):\n{ctx}" if ctx else inst
                    candidates.append((prompt, inst, resp, eng_resp))
    except Exception as e:
        print(f"  Warning loading dolly-odia-15k: {e}")

    rng.shuffle(candidates)
    print(f"  Loaded {len(candidates):,} high-purity multi-sentence Odia candidate responses.")

    # Partition candidates across Tracks 2, 3, 4, and 5 to reach 15,000 total triplets
    for idx, (prompt, inst, chosen, eng_resp) in enumerate(candidates):
        if len(triplets) >= target_total:
            break
        sents = split_odia_sentences(chosen)
        if not sents:
            continue

        mode = idx % 4
        if mode == 0:
            # Track 2: Anti-Repetition / Anti-Looping
            rej = make_looping_rejected(sents, rng)
            record_triplet(prompt, chosen, rej, "track2_anti_repetition")
        elif mode == 1:
            # Track 3: Completeness vs. Premature Truncation / Tautological Echo
            rej = make_truncated_or_tautological_rejected(inst, sents)
            record_triplet(prompt, chosen, rej, "track3_completeness_vs_truncation")
        elif mode == 2:
            # Track 4: Script & Formatting Purity vs. Pipe/English Leakage
            rej = make_noisy_or_english_leaked_rejected(chosen, eng_resp)
            record_triplet(prompt, chosen, rej, "track4_script_and_format_purity")
        else:
            # Track 5: Grounded Relevance vs. Generic Unhelpful Refusal / Hallucinated Swap
            other_chosen = candidates[(idx + 137) % len(candidates)][2]
            other_sents = split_odia_sentences(other_chosen)
            # Blend first 4 words of true answer with a hallucinated continuation from an unrelated topic
            c_words = sents[0].split()
            prefix = " ".join(c_words[: min(4, len(c_words))])
            hallucinated_tail = other_sents[-1] if other_sents else other_chosen[:120]
            rej = f"{prefix} {hallucinated_tail}"
            record_triplet(prompt, chosen, rej, "track5_factual_grounding_vs_hallucination")

    rng.shuffle(triplets)
    print(f"  Total DPO Preference Triplets Built: {len(triplets):,}")
    for k, v in sorted(stats.items()):
        print(f"    - {k:42s}: {v:,}")
    return triplets, stats


def tokenize_and_save_dpo(triplets: list[dict], sp: spm.SentencePieceProcessor) -> dict:
    print(f"[2/3] Tokenizing DPO pairs into fixed {MAX_DPO_LEN}-token arrays with prompt masking (-100)...")
    bos_id = sp.bos_id()  # 1
    eos_id = sp.eos_id()  # 2
    pad_id = sp.pad_id()  # 0

    chosen_in_list, chosen_lab_list = [], []
    rejected_in_list, rejected_lab_list = [], []
    kept = 0

    for row in tqdm(triplets, desc="Tokenizing DPO"):
        user_block = f"<|user|>\n{row['prompt']}\n<|assistant|>\n"
        prompt_ids = [bos_id] + sp.encode(user_block, out_type=int)
        if len(prompt_ids) >= MAX_DPO_LEN - 16:
            # Truncate overly long context inside user prompt while preserving the <|assistant|>\n suffix
            suffix_ids = sp.encode("\n<|assistant|>\n", out_type=int)
            prompt_ids = prompt_ids[: MAX_DPO_LEN - 96] + suffix_ids

        c_resp_ids = sp.encode(row["chosen"], out_type=int) + [eos_id]
        r_resp_ids = sp.encode(row["rejected"], out_type=int) + [eos_id]

        c_in = (prompt_ids + c_resp_ids)[:MAX_DPO_LEN]
        c_lab = ([IGNORE_INDEX] * len(prompt_ids) + c_resp_ids)[:MAX_DPO_LEN]

        r_in = (prompt_ids + r_resp_ids)[:MAX_DPO_LEN]
        r_lab = ([IGNORE_INDEX] * len(prompt_ids) + r_resp_ids)[:MAX_DPO_LEN]

        # Ensure both chosen and rejected have at least 4 active assistant tokens
        c_act = sum(1 for x in c_lab if x != IGNORE_INDEX)
        r_act = sum(1 for x in r_lab if x != IGNORE_INDEX)
        if c_act < 4 or r_act < 4:
            continue

        # Pad to MAX_DPO_LEN
        c_pad = MAX_DPO_LEN - len(c_in)
        r_pad = MAX_DPO_LEN - len(r_in)

        chosen_in_list.append(c_in + [pad_id] * c_pad)
        chosen_lab_list.append(c_lab + [IGNORE_INDEX] * c_pad)
        rejected_in_list.append(r_in + [pad_id] * r_pad)
        rejected_lab_list.append(r_lab + [IGNORE_INDEX] * r_pad)
        kept += 1

    c_in_arr = np.asarray(chosen_in_list, dtype=np.uint16)
    c_lab_arr = np.asarray(chosen_lab_list, dtype=np.int32)
    r_in_arr = np.asarray(rejected_in_list, dtype=np.uint16)
    r_lab_arr = np.asarray(rejected_lab_list, dtype=np.int32)

    val_size = min(500, max(200, int(kept * 0.03)))
    train_size = kept - val_size

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "dpo_train_chosen_inputs.npy", c_in_arr[:train_size])
    np.save(OUT_DIR / "dpo_train_chosen_labels.npy", c_lab_arr[:train_size])
    np.save(OUT_DIR / "dpo_train_rejected_inputs.npy", r_in_arr[:train_size])
    np.save(OUT_DIR / "dpo_train_rejected_labels.npy", r_lab_arr[:train_size])

    np.save(OUT_DIR / "dpo_val_chosen_inputs.npy", c_in_arr[train_size:])
    np.save(OUT_DIR / "dpo_val_chosen_labels.npy", c_lab_arr[train_size:])
    np.save(OUT_DIR / "dpo_val_rejected_inputs.npy", r_in_arr[train_size:])
    np.save(OUT_DIR / "dpo_val_rejected_labels.npy", r_lab_arr[train_size:])

    with open(OUT_DIR / "dpo_samples_preview.jsonl", "w", encoding="utf-8") as f:
        for r in triplets[:100]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "total_tokenized_triplets": kept,
        "train_triplets": int(train_size),
        "val_triplets": int(val_size),
        "max_dpo_seq_len": MAX_DPO_LEN,
    }


def upload_dpo_to_hf(meta: dict):
    print("[3/3] Uploading tokenized DPO preference arrays to private HF dataset repo...")
    api = HfApi(token=HF_TOKEN)
    user = api.whoami()["name"]
    repo_id = f"{user}/odia-slm-dpo-bin"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    for fname in [
        "dpo_train_chosen_inputs.npy",
        "dpo_train_chosen_labels.npy",
        "dpo_train_rejected_inputs.npy",
        "dpo_train_rejected_labels.npy",
        "dpo_val_chosen_inputs.npy",
        "dpo_val_chosen_labels.npy",
        "dpo_val_rejected_inputs.npy",
        "dpo_val_rejected_labels.npy",
        "dpo_samples_preview.jsonl",
    ]:
        api.upload_file(
            path_or_fileobj=str(OUT_DIR / fname),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=str(TOK_PATH),
        path_in_repo="tokenizer.model",
        repo_id=repo_id,
        repo_type="dataset",
    )
    meta["hf_dpo_dataset_repo"] = repo_id
    print(f"  Uploaded DPO dataset to: https://huggingface.co/datasets/{repo_id}")


def main():
    assert TOK_PATH.exists(), f"Missing tokenizer at {TOK_PATH}"
    sp = spm.SentencePieceProcessor(model_file=str(TOK_PATH))
    triplets, track_stats = curate_dpo_triplets(target_total=15000)
    tok_meta = tokenize_and_save_dpo(triplets, sp)
    upload_dpo_to_hf(tok_meta)

    report = {
        "step": "step_112_prepare_dpo_data",
        "track_stats": track_stats,
        "tokenization_meta": tok_meta,
    }
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "step_112_dpo_data_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== STEP 11.2 DPO PREFERENCE DATASET COMPLETE ===")
    print(f"Train Triplets Ready : {tok_meta['train_triplets']:,} (shape: {tok_meta['train_triplets']} x {MAX_DPO_LEN})")
    print(f"Val Triplets Ready   : {tok_meta['val_triplets']:,}")
    print(f"HF Cloud Dataset     : {tok_meta['hf_dpo_dataset_repo']}")
    print("=================================================")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

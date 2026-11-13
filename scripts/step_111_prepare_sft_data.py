#!/usr/bin/env python3
"""
Phase 1 (Step 11.1): SFT Corpus Curation, 13-Gram Decontamination, Loss-Masked Tokenization,
and 1024-Token Sequence Packing for OdiaSLM-100M-Instruct.
Author: Arun Misra | AI Researcher, Bhubaneswar, Odisha, India
"""

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["HF_HUB_DISABLE_XET"] = "1"

import unicodedata

def normalize_odia_text(text: str) -> str:
    if not text:
        return ""
    # Canonical Unicode NFC normalization preserving Odia matras, halant (U+0B4D), nukta (U+0B3C), and ZWNJ (U+200C)
    t = unicodedata.normalize("NFC", str(text))
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Remove invisible zero-width space (U+200B) and BOM (U+FEFF) while keeping ZWNJ/ZWJ
    t = t.replace("\u200b", "").replace("\ufeff", "")
    return t.strip()


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

OUT_DIR = Path("data/sft")
ART_DIR = Path("artifacts")
TOK_PATH = Path("artifacts/tokenizer/tokenizer.model")
SEQ_LEN = 1024
IGNORE_INDEX = -100

BANNED_SUBSTRINGS = [
    "ଅଲିଭ୍",
    "ଅଲିଭ",
    "olive",
    "odiagenai",
    "ଓଡିଆ-ଜେନ-ଏ.ଆଇ",
    "ଓଡ଼ିଆ-ଜେନ-ଏ.ଆଇ",
    "openai",
    "chatgpt",
    "ଆंथ୍ରୋପିକ୍",
    "instruction in",
]


def clean_odia_sft_text(text: str) -> str:
    if not text:
        return ""
    t = normalize_odia_text(str(text))
    # Replace ASCII pipe used as danda in noisy scrapes with proper Odia Purnachheda
    t = re.sub(r"\s*\|\s*", "। ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def odia_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    odia = sum(1 for c in text if 0x0B00 <= ord(c) <= 0x0B7F)
    alpha = sum(1 for c in text if c.isalpha() or (0x0B00 <= ord(c) <= 0x0B7F))
    return odia / max(alpha, 1)


def is_safe_response(inst: str, resp: str) -> bool:
    combined = (inst + " " + resp).lower()
    for bad in BANNED_SUBSTRINGS:
        if bad in combined:
            return False
    return True


def build_sovereign_identity_anchors() -> list:
    """Builds high-priority identity & Odisha grounding pairs for OdiaSLM-100M-Instruct."""
    base_pairs = [
        (
            "ଆପଣ କିଏ?",
            "ମୁଁ ହେଉଛି ‘OdiaSLM-100M-Instruct’, ଓଡ଼ିଆ ଭାଷା ପାଇଁ ସମ୍ପୂର୍ଣ୍ଣ ମୂଳରୁ ପ୍ରଶିକ୍ଷିତ ଏକ ସ୍ୱଦେଶୀ କୃତ୍ରିମ ବୁଦ୍ଧିମତ୍ତା (AI) ଭାଷା ମଡେଲ୍। ମୋତେ ଭୁବନେଶ୍ୱରର ଏଆଇ ଗବେଷକ ଅରୁଣ ମିଶ୍ର (Arun Misra) ବିକଶିତ କରିଛନ୍ତି।",
        ),
        (
            "ତୁମର ନାମ କ'ଣ ଏବଂ ତୁମକୁ କିଏ ତିଆରି କରିଛନ୍ତି?",
            "ମୋର ନାମ ‘OdiaSLM’ (Odia Small Language Model)। ମୋତେ ଓଡ଼ିଶାର ଭୁବନେଶ୍ୱର ସ୍ଥିତ ଏଆଇ ଗବେଷକ ଅରୁଣ ମିଶ୍ର ଓଡ଼ିଆ ଭାଷାର ବିକାଶ ଏବଂ ଡିଜିଟାଲ୍ ସଶକ୍ତିକରଣ ପାଇଁ ତିଆରି କରିଛନ୍ତି।",
        ),
        (
            "OdiaSLM କ'ଣ ଏବଂ ଏହାର ବିଶେଷତ୍ୱ କ'ଣ?",
            "OdiaSLM ହେଉଛି ଓଡ଼ିଆ ଭାଷା ପାଇଁ ସମ୍ପୂର୍ଣ୍ଣ ମୂଳରୁ (from scratch) ତିଆରି ହୋଇଥିବା ୧୦ କୋଟି ପାରାମିଟର ବିଶିଷ୍ଟ ଏକ ସ୍ୱଦେଶୀ କ୍ଷୁଦ୍ର ଭାଷା ମଡେଲ୍ (SLM)। ଏଥିରେ ଏକ ସ୍ୱତନ୍ତ୍ର ୧୬,୩୮୪ ଶବ୍ଦଭଣ୍ଡାର ବିଶିଷ୍ଟ ଓଡ଼ିଆ ଟୋକେନାଇଜର୍ ବ୍ୟବହାର କରାଯାଇଛି, ଯାହା ସାଧାରଣ ଲାପଟପ୍ କିମ୍ବା କମ୍ପ୍ୟୁଟରରେ ବିନା ଇଣ୍ଟରନେଟ୍‌ରେ ଦ୍ରୁତ ଗତିରେ ଓଡ଼ିଆ ପ୍ରଶ୍ନର ଉତ୍ତର ଦେଇପାରେ।",
        ),
        (
            "ତୁମେ କେଉଁ କେଉଁ କାମରେ ସାହାଯ୍ୟ କରିପାରିବ?",
            "ମୁଁ ଆପଣଙ୍କୁ ଓଡ଼ିଆ ଭାଷାରେ ପ୍ରଶ୍ନୋତ୍ତର, ଅନୁଚ୍ଛେଦର ସାରାଂଶ ଲେଖିବା, ପ୍ରବନ୍ଧ ଓ ଚିଠି ପ୍ରସ୍ତୁତ କରିବା, ଇଂରାଜୀରୁ ଓଡ଼ିଆ ଅନୁବାଦ ଏବଂ ଓଡ଼ିଶାର ଇତିହାସ, ସଂସ୍କୃତି ଓ ସାଧାରଣ ଜ୍ଞାନ ବିଷୟରେ ସଠିକ୍ ତଥ୍ୟ ଦେବାରେ ସାହାଯ୍ୟ କରିପାରିବି।",
        ),
        (
            "ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ'ଣ?",
            "ଓଡ଼ିଶାର ରାଜଧାନୀ ହେଉଛି ଭୁବନେଶ୍ୱର। ଏହାକୁ ଭାରତର ‘ମନ୍ଦିର ମାଳିନୀ ସହର’ କୁହାଯାଏ। ପ୍ରାଚୀନ ଲିଙ୍ଗରାଜ ମନ୍ଦିର, ଧଉଳି ଶାନ୍ତିସ୍ତୂପ, ଖଣ୍ଡଗିରି-ଉଦୟଗିରି ଗୁମ୍ଫା ସହିତ ଆଧୁନିକ ଶିକ୍ଷା ଏବଂ ସୂଚନା ପ୍ରଯୁକ୍ତିବିଦ୍ୟା (IT) କେନ୍ଦ୍ର ଭାବେ ଭୁବନେଶ୍ୱର ପ୍ରସିଦ୍ଧ।",
        ),
        (
            "ଓଡ଼ିଆ ଭାଷା କେବେ ଶାସ୍ତ୍ରୀୟ ଭାଷାର ମାନ୍ୟତା ପାଇଲା?",
            "ଓଡ଼ିଆ ଭାଷା ୨୦୧୪ ମସିହା ଫେବୃଆରୀ ୨୦ ତାରିଖରେ ଭାରତ ସରକାରଙ୍କ ଦ୍ୱାରା ଦେଶର ଷଷ୍ଠ ‘ଶାସ୍ତ୍ରୀୟ ଭାଷା’ (Classical Language) ଭାବେ ସରକାରୀ ମାନ୍ୟତା ଲାଭ କଲା। ଏହା ଭାରତ-ଆର୍ଯ୍ୟ ଭାଷା ପରିବାରର ପ୍ରଥମ ଶାସ୍ତ୍ରୀୟ ଭାଷା ଅଟେ।",
        ),
    ]
    anchors = []
    # Repeat across slight prompt variations so identity is rock-solid
    prefixes = ["", "ଦୟାକରି କୁହନ୍ତୁ, ", "ସଂକ୍ଷେପରେ ଉତ୍ତର ଦିଅନ୍ତୁ: ", "ପ୍ରଶ୍ନ: "]
    for q, a in base_pairs:
        for pref in prefixes:
            anchors.append({
                "instruction": clean_odia_sft_text(pref + q),
                "input": "",
                "output": clean_odia_sft_text(a),
                "source": "sovereign_identity",
            })
    return anchors


def build_benchmark_13gram_set() -> set:
    """Extracts 13-grams from saidutta69/odia-eval-benchmark to prevent any SFT leakage."""
    print("[1/5] Building 13-gram decontamination index from saidutta69/odia-eval-benchmark...")
    ngrams = set()
    try:
        ds = load_dataset("saidutta69/odia-eval-benchmark", split="odia", token=HF_TOKEN)
        for row in ds:
            txt = f"{row.get('prompt_odia', '')} {row.get('reference_odia', '')}"
            words = normalize_odia_text(txt).split()
            if len(words) >= 13:
                for i in range(len(words) - 12):
                    gram = " ".join(words[i : i + 13])
                    ngrams.add(hashlib.md5(gram.encode("utf-8")).digest()[:8])
    except Exception as e:
        print(f"  Warning loading benchmark for decontam: {e}")
    print(f"  Indexed {len(ngrams):,} unique 13-gram hashes from downstream benchmark.")
    return ngrams


def is_contaminated(text: str, bench_ngrams: set) -> bool:
    if not bench_ngrams:
        return False
    words = text.split()
    if len(words) < 13:
        return False
    for i in range(len(words) - 12):
        h = hashlib.md5(" ".join(words[i : i + 13]).encode("utf-8")).digest()[:8]
        if h in bench_ngrams:
            return True
    return False


def collect_and_filter_sft_records(bench_ngrams: set) -> tuple[list, dict]:
    print("[2/5] Downloading & filtering native Odia instruction/QA datasets...")
    records = []
    stats = {
        "raw_examined": 0,
        "rejected_short_or_empty": 0,
        "rejected_script_ratio": 0,
        "rejected_banned_identity": 0,
        "rejected_duplicates": 0,
        "rejected_decontam": 0,
        "kept_by_source": {},
    }

    seen_hashes = set()

    def try_add(inst: str, inp: str, out: str, source_name: str, min_out_chars: int = 12):
        stats["raw_examined"] += 1
        inst_c = clean_odia_sft_text(inst)
        inp_c = clean_odia_sft_text(inp)
        out_c = clean_odia_sft_text(out)

        if len(inst_c) < 4 or len(out_c) < min_out_chars or len(out_c) > 3200:
            stats["rejected_short_or_empty"] += 1
            return False

        if not is_safe_response(inst_c, out_c):
            stats["rejected_banned_identity"] += 1
            return False

        # Require strong Odia script density on the assistant's output (>= 0.60)
        if odia_char_ratio(out_c) < 0.60:
            stats["rejected_script_ratio"] += 1
            return False

        # Require Odia script on prompt unless it is an explicit English->Odia translation task
        if source_name != "odiencorp_translation" and odia_char_ratio(inst_c + " " + inp_c) < 0.45:
            stats["rejected_script_ratio"] += 1
            return False

        full_text = f"{inst_c} {inp_c} {out_c}".strip()
        if is_contaminated(full_text, bench_ngrams):
            stats["rejected_decontam"] += 1
            return False

        # Deduplicate on normalized (instruction + input + first 120 chars of output)
        dedup_key = hashlib.sha1(f"{inst_c}||{inp_c}||{out_c[:120]}".encode("utf-8")).digest()
        if dedup_key in seen_hashes:
            stats["rejected_duplicates"] += 1
            return False
        seen_hashes.add(dedup_key)

        records.append({
            "instruction": inst_c,
            "input": inp_c,
            "output": out_c,
            "source": source_name,
        })
        stats["kept_by_source"][source_name] = stats["kept_by_source"].get(source_name, 0) + 1
        return True

    # 0. Sovereign Identity Anchors
    for row in build_sovereign_identity_anchors():
        records.append(row)
        stats["kept_by_source"]["sovereign_identity"] = stats["kept_by_source"].get("sovereign_identity", 0) + 1

    # 1. OdiaGenAI/gpt-teacher-instruct-odia-18k (High-quality multi-paragraph explanations)
    try:
        ds = load_dataset("OdiaGenAI/gpt-teacher-instruct-odia-18k", split="train", token=HF_TOKEN)
        for r in ds:
            try_add(r.get("instruction", ""), r.get("input", ""), r.get("response", ""), "gpt_teacher_18k", min_out_chars=20)
    except Exception as e:
        print(f"  Warning on gpt-teacher-18k: {e}")

    # 2. OdiaGenAI/dolly-odia-15k (Grounded QA, brainstorming, summarization)
    try:
        ds = load_dataset("OdiaGenAI/dolly-odia-15k", split="train", token=HF_TOKEN)
        for r in ds:
            try_add(r.get("instruction", ""), r.get("context", ""), r.get("response", ""), "dolly_odia_15k", min_out_chars=15)
    except Exception as e:
        print(f"  Warning on dolly-odia-15k: {e}")

    # 3. OdiaGenAI/gpt-teacher-roleplay-odia-3k (Creative & dialogue instructions)
    try:
        ds = load_dataset("OdiaGenAI/gpt-teacher-roleplay-odia-3k", split="train", token=HF_TOKEN)
        for r in ds:
            try_add(r.get("instruction", ""), r.get("input", ""), r.get("response", ""), "gpt_roleplay_3k", min_out_chars=20)
    except Exception as e:
        print(f"  Warning on gpt-roleplay-3k: {e}")

    # 4. OdiaGenAI/Odia_Alpaca_instructions_52k (General instruction following)
    try:
        ds = load_dataset("OdiaGenAI/Odia_Alpaca_instructions_52k", split="train", token=HF_TOKEN)
        for r in ds:
            try_add(r.get("instruction", ""), r.get("input", ""), r.get("output", ""), "alpaca_odia_52k", min_out_chars=18)
    except Exception as e:
        print(f"  Warning on alpaca-52k: {e}")

    # 5. abhilash88/odia-instruction-dataset (Odisha news summarization & regional instructions)
    try:
        ds = load_dataset("abhilash88/odia-instruction-dataset", split="train", token=HF_TOKEN)
        kept_abhilash = 0
        for r in ds:
            if try_add(r.get("instruction", ""), r.get("input", ""), r.get("output", ""), "odia_news_instruct", min_out_chars=25):
                kept_abhilash += 1
                if kept_abhilash >= 20000:
                    break
    except Exception as e:
        print(f"  Warning on abhilash88/odia-instruction-dataset: {e}")

    # 6. OdiaGenAI/odia_context_qa_98k (Capped at 18,000 substantive context QA pairs >= 15 chars)
    try:
        ds = load_dataset("OdiaGenAI/odia_context_qa_98k", split="train", token=HF_TOKEN)
        kept_qa = 0
        for r in ds:
            inp_txt = r.get("input", "")
            out_txt = r.get("output", "")
            if len(str(inp_txt)) >= 40 and len(str(out_txt)) >= 15:
                if try_add(r.get("instruction", ""), inp_txt, out_txt, "odia_context_qa", min_out_chars=15):
                    kept_qa += 1
                    if kept_qa >= 18000:
                        break
    except Exception as e:
        print(f"  Warning on odia_context_qa_98k: {e}")

    # 7. OdiaGenAI/OdiEnCorp_translation_instructions_25k (Exclude archaic 'genesis', cap at 8,000)
    try:
        ds = load_dataset("OdiaGenAI/OdiEnCorp_translation_instructions_25k", split="train", token=HF_TOKEN)
        kept_trans = 0
        for r in ds:
            cat = str(r.get("category", "")).lower()
            if "genesis" in cat or "bible" in cat:
                continue
            if try_add(r.get("instruction", ""), r.get("input", ""), r.get("output", ""), "odiencorp_translation", min_out_chars=20):
                kept_trans += 1
                if kept_trans >= 8000:
                    break
    except Exception as e:
        print(f"  Warning on OdiEnCorp: {e}")

    print(f"  Total Clean, Deduplicated, Decontaminated SFT Examples: {len(records):,}")
    for k, v in stats["kept_by_source"].items():
        print(f"    - {k:24s}: {v:,}")
    return records, stats


def encode_and_pack_sft(records: list, sp: spm.SentencePieceProcessor) -> dict:
    """
    Formats each SFT record with <|user|> and <|assistant|> tags, masks prompt tokens
    with IGNORE_INDEX (-100), and packs multiple turns into dense 1024-token blocks.
    """
    print("[3/5] Tokenizing with Assistant-Only Loss Masking & 1024-Token Packing...")
    rng = random.Random(42)
    shuffled = list(records)
    rng.shuffle(shuffled)

    # Reserve 1,500 examples for validation, rest for training
    val_count = min(1500, max(500, int(len(shuffled) * 0.02)))
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]

    # Inject sovereign identity anchors multiple times across train_records so they appear throughout every epoch
    identity_rows = [r for r in records if r["source"] == "sovereign_identity"]
    train_records.extend(identity_rows * 8)
    rng.shuffle(train_records)

    bos_id = sp.bos_id()  # 1
    eos_id = sp.eos_id()  # 2
    pad_id = sp.pad_id()  # 0

    def tokenize_single(rec: dict) -> tuple[list[int], list[int]]:
        inst = rec["instruction"]
        inp = rec["input"]
        out = rec["output"]

        if inp:
            user_block = f"<|user|>\n{inst}\n\nପ୍ରସଙ୍ଗ (Context):\n{inp}\n<|assistant|>\n"
        else:
            user_block = f"<|user|>\n{inst}\n<|assistant|>\n"

        prompt_ids = [bos_id] + sp.encode(user_block, out_type=int)
        resp_ids = sp.encode(out, out_type=int) + [eos_id]

        input_ids = prompt_ids + resp_ids
        # Mask prompt tokens with -100 so loss is computed strictly on assistant response + EOS
        labels = [IGNORE_INDEX] * len(prompt_ids) + resp_ids

        if len(input_ids) > SEQ_LEN:
            input_ids = input_ids[:SEQ_LEN]
            labels = labels[:SEQ_LEN]
        return input_ids, labels

    def pack_split(split_recs: list) -> tuple[np.ndarray, np.ndarray, dict]:
        packed_inputs = []
        packed_labels = []
        cur_in = []
        cur_lab = []
        total_active_tokens = 0
        total_prompt_tokens = 0
        kept_examples = 0

        for rec in tqdm(split_recs, desc="Packing sequences"):
            in_ids, lab_ids = tokenize_single(rec)
            active = sum(1 for x in lab_ids if x != IGNORE_INDEX)
            if active < 3:
                continue
            kept_examples += 1
            total_active_tokens += active
            total_prompt_tokens += (len(in_ids) - active)

            if len(cur_in) + len(in_ids) <= SEQ_LEN:
                cur_in.extend(in_ids)
                cur_lab.extend(lab_ids)
            else:
                # Pad remaining tail of current 1024 block with pad_id (0) and IGNORE_INDEX (-100)
                pad_len = SEQ_LEN - len(cur_in)
                cur_in.extend([pad_id] * pad_len)
                cur_lab.extend([IGNORE_INDEX] * pad_len)
                packed_inputs.append(cur_in)
                packed_labels.append(cur_lab)
                cur_in = list(in_ids)
                cur_lab = list(lab_ids)

        if cur_in:
            pad_len = SEQ_LEN - len(cur_in)
            cur_in.extend([pad_id] * pad_len)
            cur_lab.extend([IGNORE_INDEX] * pad_len)
            packed_inputs.append(cur_in)
            packed_labels.append(cur_lab)

        arr_in = np.asarray(packed_inputs, dtype=np.uint16)
        arr_lab = np.asarray(packed_labels, dtype=np.int32)
        meta = {
            "num_examples": kept_examples,
            "num_packed_blocks_1024": int(arr_in.shape[0]),
            "total_block_tokens": int(arr_in.size),
            "active_assistant_loss_tokens": int(total_active_tokens),
            "masked_user_prompt_tokens": int(total_prompt_tokens),
            "packing_efficiency_pct": round(
                100.0 * (total_active_tokens + total_prompt_tokens) / max(arr_in.size, 1), 2
            ),
        }
        return arr_in, arr_lab, meta

    train_in, train_lab, train_meta = pack_split(train_records)
    val_in, val_lab, val_meta = pack_split(val_records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "sft_train_inputs.npy", train_in)
    np.save(OUT_DIR / "sft_train_labels.npy", train_lab)
    np.save(OUT_DIR / "sft_val_inputs.npy", val_in)
    np.save(OUT_DIR / "sft_val_labels.npy", val_lab)

    # Save 200 sample JSONL rows for quick human inspection
    with open(OUT_DIR / "sft_samples_preview.jsonl", "w", encoding="utf-8") as f:
        for r in train_records[:200]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"train": train_meta, "val": val_meta}


def upload_sft_binaries_to_hf(report: dict):
    print("[4/5] Uploading packed SFT arrays to private HF dataset repo for Colab/Kaggle...")
    api = HfApi(token=HF_TOKEN)
    user = api.whoami()["name"]
    repo_id = f"{user}/odia-slm-sft-bin"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    for fname in [
        "sft_train_inputs.npy",
        "sft_train_labels.npy",
        "sft_val_inputs.npy",
        "sft_val_labels.npy",
        "sft_samples_preview.jsonl",
    ]:
        fpath = OUT_DIR / fname
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
        )
    # Also upload tokenizer.model so the SFT dataset repo is 100% self-contained
    api.upload_file(
        path_or_fileobj=str(TOK_PATH),
        path_in_repo="tokenizer.model",
        repo_id=repo_id,
        repo_type="dataset",
    )
    report["hf_sft_dataset_repo"] = repo_id
    print(f"  Uploaded packed SFT dataset to private repo: https://huggingface.co/datasets/{repo_id}")


def main():
    assert TOK_PATH.exists(), f"Missing tokenizer at {TOK_PATH}"
    sp = spm.SentencePieceProcessor(model_file=str(TOK_PATH))

    bench_ngrams = build_benchmark_13gram_set()
    records, filter_stats = collect_and_filter_sft_records(bench_ngrams)
    pack_meta = encode_and_pack_sft(records, sp)

    report = {
        "step": "step_111_prepare_sft_data",
        "seq_len": SEQ_LEN,
        "ignore_index": IGNORE_INDEX,
        "filter_stats": filter_stats,
        "packing_stats": pack_meta,
    }
    upload_sft_binaries_to_hf(report)

    report_path = ART_DIR / "step_111_sft_data_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== STEP 11.1 SFT DATA CURATION & PACKING COMPLETE ===")
    print(f"Total Clean SFT Examples  : {len(records):,}")
    print(f"Packed Train 1024-Blocks  : {pack_meta['train']['num_packed_blocks_1024']:,} ({pack_meta['train']['packing_efficiency_pct']}% density)")
    print(f"Active Assistant Tokens   : {pack_meta['train']['active_assistant_loss_tokens']:,}")
    print(f"Packed Val 1024-Blocks    : {pack_meta['val']['num_packed_blocks_1024']:,}")
    print(f"HF Cloud Dataset Ready    : {report['hf_sft_dataset_repo']}")
    print("======================================================")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

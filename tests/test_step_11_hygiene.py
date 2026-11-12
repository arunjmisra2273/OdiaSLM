#!/usr/bin/env python3
"""
Verification & Unit Test Suite for Step 1.1 (src/hygiene.py).
Runs 7 deterministic unit tests + a 10,000-row live stream audit.
"""

import json
import os
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm
from src.hygiene import clean_and_filter_document, sanitize_zwnj_zwj
from scripts.step_03_eda_secondary_and_eval import load_dotenv_wsl_safe

ARTIFACT_PATH = Path("artifacts/step_11_hygiene_verification.json")


def run_unit_tests():
    print("[1/2] Running 7 deterministic Odia hygiene unit tests ...")

    # Test 1: Valid ZWNJ after Halant (\u0b4d\u200c) is preserved, orphaned ZWNJ at start/after space is stripped
    sample_zwnj = "\u200cଓଡ଼ିଆ ଭାଷାରେ କ୍\u200cଷ ଏବଂ \u200c\u200c ଶବ୍ଦ"
    cleaned_zwnj = sanitize_zwnj_zwj(sample_zwnj)
    assert "\u0b4d\u200c" in cleaned_zwnj, "Failed to preserve valid Halant+ZWNJ"
    assert not cleaned_zwnj.startswith("\u200c"), "Failed to strip leading orphaned ZWNJ"
    assert "\u200c\u200c" not in cleaned_zwnj, "Failed to collapse duplicate ZWNJ"

    # Test 2: Nukta characters (ଡ଼ \u0b5c, ଢ଼ \u0b5d) and composite matras (ୌ \u0b4c) remain NFC-intact
    odia_prose = (
        "ଓଡ଼ିଶାର ସଂସ୍କୃତି ଏବଂ ଐତିହ୍ୟ ଅତ୍ୟନ୍ତ ସମୃଦ୍ଧ ଅଟେ। "
        "ଏଠାରେ ଅନେକ ପ୍ରାଚୀନ ମନ୍ଦିର, ପର୍ବପର୍ବାଣି ଏବଂ ସାହିତ୍ୟିକ କୃତି ଦେଖିବାକୁ ମିଳେ। "
        "ଭୁବନେଶ୍ୱରକୁ ମନ୍ଦିର ମାଳିନୀ ସହର କୁହାଯାଏ ଏବଂ ପୁରୀର ଜଗନ୍ନାଥ ମନ୍ଦିର ବିଶ୍ୱ ప్రసిଦ୍ଧ।"
    )
    # Fix small Telugu typo in test string to pure Odia
    odia_prose = odia_prose.replace("ప్రసిଦ୍ଧ", "ପ୍ରସିଦ୍ଧ")
    cleaned, reason = clean_and_filter_document(odia_prose, source_name="odia_web_corpus_v5")
    assert cleaned is not None and reason == "accepted_odia_doc", f"Unexpected rejection: {reason}"
    assert "\u0b5c" in cleaned or "ଡ଼" in cleaned, "Nukta character lost"

    # Test 3: Surgical stripping of [INST] and [/INST] while keeping the Odia Q&A text
    inst_doc = (
        "<s>[INST] ଓଡ଼ିଶାର ରାଜଧାନୀ କ’ଣ ଏବଂ ଏହାର ଇତିହାସ ବିଷୟରେ ସଂକ୍ଷେପରେ ଲେଖନ୍ତୁ? [/INST] "
        "ଓଡ଼ିଶାର ରାଜଧାନୀ ଭୁବନେଶ୍ୱର ଅଟେ। ୧୯୪୮ ମସିହାରେ କଟକରୁ ଭୁବନେଶ୍ୱରକୁ ରାଜଧାନୀ ସ୍ଥାନାନ୍ତରିତ ହୋଇଥିଲା। "
        "ଏହା ଏକ ପ୍ରାଚୀନ ଐତିହାସିକ ସହର ଏବଂ ଏଠାରେ ଲିଙ୍ଗରାଜ ମନ୍ଦିର ଓ ଖଣ୍ଡଗିରି ଗୁମ୍ଫା ଅବସ୍ଥିତ।</s>"
    )
    cleaned_inst, reason_inst = clean_and_filter_document(inst_doc, source_name="rag-suite-oriya")
    assert cleaned_inst is not None, f"Failed to rescue [INST] doc: {reason_inst}"
    assert "[INST]" not in cleaned_inst and "[/INST]" not in cleaned_inst and "<s>" not in cleaned_inst

    # Test 4: Samanantar single sentence (8-12 words) is accepted when source='samanantar'
    sam_sentence = "ଭାରତୀୟ ସମ୍ବିଧାନ ଅନୁଯାୟୀ ପ୍ରତ୍ୟେକ ନାଗରିକଙ୍କର ମୌଳିକ ଅଧିକାର ରହିଛି ଏବଂ ଏହା ସୁରକ୍ଷିତ ଅଟେ।"
    cleaned_sam, reason_sam = clean_and_filter_document(sam_sentence, source_name="samanantar")
    assert cleaned_sam is not None and reason_sam == "accepted_odia_sentence", f"Samanantar failed: {reason_sam}"

    # Test 5: PII scrubbing (email & Indian phone number)
    pii_doc = (
        "ଯୋଗାଯୋଗ କରିବା ପାଇଁ ଆମର ଇମେଲ୍ test.user@odisha.gov.in କିମ୍ବା ଫୋନ୍ ନମ୍ବର +91 9876543210 ରେ ସମ୍ପର୍କ କରନ୍ତୁ। "
        "ଏହି କାର୍ଯ୍ୟାଳୟ ସକାଳ ଦଶଟାରୁ ସନ୍ଧ୍ୟା ପାଞ୍ଚଟା ପର୍ଯ୍ୟନ୍ତ ଖୋଲା ରହେ ଏବଂ ସମସ୍ତ ନାଗରିକଙ୍କୁ ସେବା ଯୋଗାଇ ଦିଆଯାଏ।"
    )
    cleaned_pii, _ = clean_and_filter_document(pii_doc, source_name="default")
    assert "[EMAIL]" in cleaned_pii and "[PHONE]" in cleaned_pii
    assert "9876543210" not in cleaned_pii

    # Test 6: Short-line navigation menu boilerplate rejection
    menu_doc = "ମୁଖ୍ୟ ପୃଷ୍ଠା\nରାଜନୀତି\nଖେଳ ଖବର\nমনोरंजन\nযোগাযোগ\nଆମ ବିଷୟରେ\nଲଗଇନ୍ କରନ୍ତୁ\nସାଇନ୍ ଅପ୍"
    cleaned_menu, reason_menu = clean_and_filter_document(menu_doc, source_name="default")
    assert cleaned_menu is None

    # Test 7: English anchor filtering
    en_doc = (
        "Photosynthesis is a biological process used by plants, algae, and certain bacteria to harness "
        "energy from sunlight and turn it into chemical energy. During oxygenic photosynthesis, light energy "
        "transfers electrons from water taken up by plant roots to carbon dioxide to produce carbohydrates."
    )
    cleaned_en, reason_en = clean_and_filter_document(en_doc, source_name="fineweb", is_english_anchor=True)
    assert cleaned_en is not None and reason_en == "accepted_en"

    print("      All 7 deterministic unit tests PASSED!")


def run_live_stream_audit(sample_size: int = 10_000):
    print(f"[2/2] Auditing hygiene filter on {sample_size:,} live streamed docs from saidutta69/odia_pretrain_dataset_v2 ...")
    load_dotenv_wsl_safe(".env")
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("saidutta69/odia_pretrain_dataset_v2", split="train", streaming=True, token=token)

    outcomes = Counter()
    rescued_inst_docs = 0
    raw_inst_docs = 0
    kept_chars = 0
    kept_words = 0

    for idx, row in enumerate(tqdm(ds, total=sample_size, desc="Hygiene Audit")):
        if idx >= sample_size:
            break
        raw_text = row.get("text", "")
        src = str(row.get("source", "default"))
        had_inst = "[INST]" in raw_text or "### Instruction" in raw_text
        if had_inst:
            raw_inst_docs += 1

        cleaned, reason = clean_and_filter_document(raw_text, source_name=src, is_english_anchor=False)
        outcomes[reason] += 1
        if cleaned is not None:
            kept_chars += len(cleaned)
            kept_words += len(cleaned.split())
            if had_inst:
                rescued_inst_docs += 1

    accepted_total = outcomes["accepted_odia_doc"] + outcomes["accepted_odia_sentence"]
    accept_pct = round(100.0 * accepted_total / max(1, sample_size), 2)

    report = {
        "unit_tests_passed": 7,
        "live_sample_size": sample_size,
        "accepted_total": accepted_total,
        "acceptance_rate_pct": accept_pct,
        "outcome_breakdown": dict(outcomes),
        "raw_inst_docs_seen": raw_inst_docs,
        "rescued_inst_docs_after_surgical_strip": rescued_inst_docs,
        "kept_words_in_sample": kept_words,
        "kept_chars_in_sample": kept_chars,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== STEP 1.1 VERIFICATION SUMMARY ===")
    print(f"Unit Tests Passed       : 7 / 7 (NFC, ZWNJ, [INST], Samanantar, PII, Boilerplate, English)")
    print(f"Live Stream Audited     : {sample_size:,} rows")
    print(f"Total Accepted Docs     : {accepted_total:,} ({accept_pct}%)")
    print(f"Outcome Breakdown       : {dict(outcomes)}")
    print(f"[INST] Docs Rescued     : {rescued_inst_docs:,} / {raw_inst_docs:,} cleanly stripped & retained")
    print(f"Yield per 10K Raw Docs  : {kept_words:,} clean words ({kept_chars:,} chars)")
    print(f"Artifact Saved To       : {ARTIFACT_PATH}")
    print("=====================================")


if __name__ == "__main__":
    run_unit_tests()
    run_live_stream_audit(10_000)

# python3 tests/test_step_11_hygiene.py
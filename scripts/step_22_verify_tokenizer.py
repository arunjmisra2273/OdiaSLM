#!/usr/bin/env python3
"""
Step 2.2: Tokenizer Acceptance Gate & Paper Table 1 Compression Benchmark.
Evaluates artifacts/tokenizer/tokenizer.model across the unseen 47,753-document
validation split (val_docs.jsonl).
"""

import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

VAL_JSONL = Path("/mnt/d/odia_llm_data/processed/val_docs.jsonl")
TOK_MODEL = Path("artifacts/tokenizer/tokenizer.model")
REPORT_PATH = Path("artifacts/step_22_tokenizer_verification.json")


def run_orthographic_edge_tests(sp: spm.SentencePieceProcessor) -> list:
    """Tests specific Odia morphological constructs (juktakshara, nukta, matras, ZWNJ, digits)."""
    test_cases = [
        ("juktakshara_conjuncts", "ଉତ୍କଳର ସଂସ୍କୃତି, ସ୍ୱାଧୀନତା ସଂଗ୍ରାମ ଏବଂ ଜ୍ଞାନବିଜ୍ଞାନର ବିକାଶ।"),
        ("nukta_and_matras", "ଓଡ଼ିଶার ଗୌରବମୟ ଐତିହ୍ୟ ଓ କୃষି ଭିତ୍ତିକ ଅର୍ଥନୀତି ସୁଦୃଢ଼ ଅଟେ।".replace("র", "ର").replace("ষ", "ଷ")),
        ("halant_zwnj_preservation", "ଶ୍ରୀଜଗନ୍ନାଥଙ୍କ ରଥଯାତ୍ରା ଏବଂ କ୍\u200cଷେତ୍ର ପରିକ୍ରମା।"),
        ("odia_and_ascii_digits", "୨୦୨୬ ମସିହାରେ ଓଡ଼ିଶାର ଜନସଂଖ୍ୟା 4,50,00,000 ରୁ ଅଧିକ ଅଟେ।"),
        ("code_and_bilingual", "Python 3.11 ଏବଂ PyTorch ବ୍ୟବହାର କରି ଆମେ Odia LLM ନିର୍ମାଣ କରୁଛୁ!"),
    ]
    results = []
    for name, text in test_cases:
        nfc_text = unicodedata.normalize("NFC", text)
        ids = sp.encode(nfc_text, out_type=int)
        pieces = sp.encode(nfc_text, out_type=str)
        decoded = sp.decode(ids)
        exact_match = (decoded == nfc_text)
        has_unk = (3 in ids)
        assert exact_match, f"Round-trip failed on {name}: {decoded!r} != {nfc_text!r}"
        assert not has_unk, f"<unk> token found in {name}"
        results.append({
            "test": name,
            "input": nfc_text,
            "num_words": len(nfc_text.split()),
            "num_tokens": len(ids),
            "fertility": round(len(ids) / max(1, len(nfc_text.split())), 3),
            "pieces": pieces,
            "lossless_roundtrip": exact_match,
        })
    return results


def main():
    t0 = time.time()
    assert TOK_MODEL.exists(), f"Tokenizer model not found at {TOK_MODEL}"
    assert VAL_JSONL.exists(), f"Validation file not found at {VAL_JSONL}"

    sp = spm.SentencePieceProcessor(model_file=str(TOK_MODEL))
    print(f"[1/3] Running Odia orthographic & morphological edge-case tests ...")
    edge_results = run_orthographic_edge_tests(sp)
    print("      All 5 orthographic edge-case round-trip tests PASSED!")

    print(f"[2/3] Auditing full validation set ({VAL_JSONL}) ...")
    total_docs = 0
    roundtrip_checked = 0
    roundtrip_passed = 0
    roundtrip_failures = []

    unk_token_count = 0
    odia_words = 0
    odia_chars = 0
    odia_utf8_bytes = 0
    odia_tokens = 0

    en_words = 0
    en_chars = 0
    en_utf8_bytes = 0
    en_tokens = 0

    doc_token_lengths = []
    source_stats = Counter()

    with open(VAL_JSONL, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=47_753, desc="Validating Tokenizer")):
            rec = json.loads(line)
            text = rec["text"]
            src = rec.get("source", "unknown")
            is_en = (src == "anchor:fineweb_edu_en")

            ids = sp.encode(text, out_type=int)
            n_tok = len(ids)
            n_w = len(text.split())
            n_c = len(text)
            n_b = len(text.encode("utf-8"))

            total_docs += 1
            doc_token_lengths.append(n_tok)
            source_stats[src] += n_tok

            # Count any <unk> (ID 3)
            if 3 in ids:
                unk_token_count += ids.count(3)

            # Check exact lossless round-trip on first 10,000 validation documents
            if roundtrip_checked < 10_000:
                roundtrip_checked += 1
                decoded = sp.decode(ids)
                if decoded == text:
                    roundtrip_passed += 1
                else:
                    if len(roundtrip_failures) < 5:
                        roundtrip_failures.append({"idx": idx, "orig_len": len(text), "dec_len": len(decoded)})

            if is_en:
                en_words += n_w
                en_chars += n_c
                en_utf8_bytes += n_b
                en_tokens += n_tok
            else:
                odia_words += n_w
                odia_chars += n_c
                odia_utf8_bytes += n_b
                odia_tokens += n_tok

    odia_fertility = round(odia_tokens / max(1, odia_words), 4)
    odia_bytes_per_tok = round(odia_utf8_bytes / max(1, odia_tokens), 4)
    odia_chars_per_tok = round(odia_chars / max(1, odia_tokens), 4)

    en_fertility = round(en_tokens / max(1, en_words), 4)
    en_bytes_per_tok = round(en_utf8_bytes / max(1, en_tokens), 4)

    all_words = odia_words + en_words
    all_tokens = odia_tokens + en_tokens
    overall_fertility = round(all_tokens / max(1, all_words), 4)

    # Estimate total corpus tokens for train_docs.jsonl (1,240,800,000 train words)
    estimated_train_tokens = int(1_240_800_000 * overall_fertility)

    tok_arr = np.array(doc_token_lengths, dtype=np.float64)
    pcts = {f"p{p}": int(np.percentile(tok_arr, p)) for p in [10, 25, 50, 75, 90, 95, 99]}

    # Gate checks
    gate_1_pass = (roundtrip_passed == roundtrip_checked)
    gate_2_pass = (unk_token_count == 0)
    gate_3_pass = (1.30 <= odia_fertility <= 2.10) and (odia_bytes_per_tok >= 5.5)
    all_gates_passed = gate_1_pass and gate_2_pass and gate_3_pass

    elapsed = round(time.time() - t0, 2)

    report = {
        "all_acceptance_gates_passed": all_gates_passed,
        "gate_1_lossless_roundtrip": {
            "passed": gate_1_pass,
            "checked_docs": roundtrip_checked,
            "passed_docs": roundtrip_passed,
            "failures": roundtrip_failures,
        },
        "gate_2_zero_unk": {
            "passed": gate_2_pass,
            "unk_tokens_found": unk_token_count,
            "total_val_tokens_scanned": all_tokens,
        },
        "gate_3_odia_compression": {
            "passed": gate_3_pass,
            "odia_fertility_tokens_per_word": odia_fertility,
            "odia_utf8_bytes_per_token": odia_bytes_per_tok,
            "odia_chars_per_token": odia_chars_per_tok,
            "odia_val_words": odia_words,
            "odia_val_tokens": odia_tokens,
        },
        "english_anchor_metrics": {
            "en_fertility_tokens_per_word": en_fertility,
            "en_utf8_bytes_per_token": en_bytes_per_tok,
            "en_val_words": en_words,
            "en_val_tokens": en_tokens,
        },
        "overall_val_summary": {
            "val_docs": total_docs,
            "val_words": all_words,
            "val_tokens": all_tokens,
            "overall_fertility": overall_fertility,
            "doc_token_length_percentiles": pcts,
            "projected_train_bin_tokens": estimated_train_tokens,
            "projected_train_bin_size_gb": round(estimated_train_tokens * 2 / (1024 ** 3), 2),
        },
        "orthographic_edge_tests": edge_results,
        "elapsed_seconds": elapsed,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== STEP 2.2 VERIFICATION SUMMARY ===")
    print(f"Gate 1 (Lossless Round-Trip) : {'PASS' if gate_1_pass else 'FAIL'} ({roundtrip_passed:,} / {roundtrip_checked:,} docs exact match)")
    print(f"Gate 2 (Zero <unk> Rate)     : {'PASS' if gate_2_pass else 'FAIL'} ({unk_token_count} <unk> across {all_tokens:,} val tokens)")
    print(f"Gate 3 (Odia Compression)    : {'PASS' if gate_3_pass else 'FAIL'}")
    print(f"   -> Odia Fertility         : {odia_fertility} tokens/word ({odia_chars_per_tok} chars/token)")
    print(f"   -> Odia UTF-8 Bytes/Token : {odia_bytes_per_tok} bytes/token (vs 1.00 for byte-level / ~1.8 for English GPT tokenizers on Odia)")
    print(f"   -> English Fertility      : {en_fertility} tokens/word ({en_bytes_per_tok} bytes/token)")
    print(f"Validation Set Token Count   : {all_tokens:,} tokens across {total_docs:,} docs (p50={pcts['p50']}, p90={pcts['p90']} toks/doc)")
    print(f"Projected Full Train Tokens  : ~{estimated_train_tokens:,} tokens (~{report['overall_val_summary']['projected_train_bin_size_gb']} GB uint16 train.bin)")
    print(f"All Acceptance Gates Passed  : {all_gates_passed} (in {elapsed}s)")
    print("=====================================")

    if not all_gates_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

# python3 scripts/step_22_verify_tokenizer.py
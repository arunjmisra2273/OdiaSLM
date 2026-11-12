#!/usr/bin/env python3
"""
Step 1.1: Odia-Specific NFC Normalization, Orthography-Safe ZWNJ Sanitization,
Surgical Instruction-Wrapper Stripping, PII Scrubbing, and Source-Aware Quality Filtering.
"""

import re
import unicodedata
from typing import Optional, Tuple

# 1. Surgical chat/instruction wrapper regexes (strip markers, keep Odia content)
RE_CHAT_MARKERS = re.compile(
    r"(\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|<s>|</s>|"
    r"###\s*(?:Instruction|Input|Response|Context)\s*:|"
    r"^(?:User|Assistant|Human|System)\s*:)",
    flags=re.IGNORECASE | re.MULTILINE,
)

# 2. HTML tag stripper
RE_HTML_TAGS = re.compile(r"</?(?:html|body|div|span|p|br|hr|table|tr|td|th|ul|ol|li|a|img|script|style|font|center|b|i|u|strong|em)\b[^>]*>", re.IGNORECASE)

# 3. PII patterns (Emails, Indian 10-digit mobile numbers, IPv4)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_INDIA_PHONE = re.compile(r"(?:\+91[\-\s]?)?\b[6-9]\d{9}\b")
RE_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# 4. Zero-width noise characters to strip unconditionally
RE_NOISE_ZW = re.compile(r"[\u200b\u200e\u200f\ufeff\u00ad\u2060]")

# 5. Orphaned ZWNJ (\u200c) / ZWJ (\u200d):
# Keep ONLY when immediately preceded by an Odia consonant (\u0b15-\u0b39, \u0b5c-\u0b5d) or Odia Halant (\u0b4d)
RE_ORPHAN_ZWNJ_ZWJ = re.compile(r"(?<![\u0b15-\u0b39\u0b5c\u0b5d\u0b4d])[\u200c\u200d]+|[\u200c\u200d]{2,}")

# 6. Whitespace normalizers
RE_HORIZONTAL_WS = re.compile(r"[ \t\r\f\v]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def sanitize_zwnj_zwj(text: str) -> str:
    """
    Preserves a single ZWNJ (\u200c) or ZWJ (\u200d) strictly when preceded by an Odia
    consonant or Halant (\u0b4d), and strips orphaned or repeated occurrences.
    """
    # Collapse multiple consecutive ZWNJ/ZWJ into one first
    text = re.sub(r"([\u200c\u200d])[\u200c\u200d]+", r"\1", text)
    # Remove any ZWNJ/ZWJ not preceded by valid Odia consonant or halant
    out_chars = []
    for i, ch in enumerate(text):
        if ch in ("\u200c", "\u200d"):
            if i > 0:
                prev_cp = ord(text[i - 1])
                if (0x0B15 <= prev_cp <= 0x0B39) or prev_cp in (0x0B4D, 0x0B5C, 0x0B5D):
                    out_chars.append(ch)
            # else drop orphaned ZWNJ/ZWJ
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def compute_script_metrics(text: str) -> Tuple[float, float, int]:
    """
    Returns (odia_ratio, latin_ratio, non_ws_char_count).
    Odia block: U+0B00..U+0B7F plus Danda/Double Danda (U+0964, U+0965).
    """
    odia_count = 0
    latin_count = 0
    non_ws_Punct = 0

    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        # Skip common ASCII punctuation when computing script ratio so punctuation doesn't penalize short sentences
        if cp < 128 and not ch.isalnum():
            continue
        non_ws_Punct += 1
        if (0x0B00 <= cp <= 0x0B7F) or cp in (0x0964, 0x0965):
            odia_count += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            latin_count += 1

    if non_ws_Punct == 0:
        return 0.0, 0.0, 0
    return odia_count / non_ws_Punct, latin_count / non_ws_Punct, non_ws_Punct


def clean_and_filter_document(
    raw_text: str,
    source_name: str = "default",
    is_english_anchor: bool = False,
) -> Tuple[Optional[str], str]:
    """
    Executes the 6-stage hygiene pipeline on a single document.
    Returns (cleaned_text, status_reason).
    If rejected, cleaned_text is None and status_reason explains which filter triggered.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None, "rejected_empty"

    # Stage 1: Canonical Unicode NFC Composition (NEVER NFKC)
    text = unicodedata.normalize("NFC", raw_text)

    # Stage 2: Surgical removal of chat/instruction markers & HTML tags
    text = RE_CHAT_MARKERS.sub("\n", text)
    text = RE_HTML_TAGS.sub(" ", text)

    # Stage 3: Zero-width sanitization & ZWNJ/ZWJ morphological preservation
    text = RE_NOISE_ZW.sub("", text)
    text = sanitize_zwnj_zwj(text)

    # Remove non-printable control characters except \n and \t
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)

    # Stage 4: PII Scrubbing
    text = RE_EMAIL.sub("[EMAIL]", text)
    text = RE_INDIA_PHONE.sub("[PHONE]", text)
    text = RE_IPV4.sub("[IP]", text)

    # Normalize line-by-line whitespace
    cleaned_lines = []
    for line in text.splitlines():
        ln = RE_HORIZONTAL_WS.sub(" ", line).strip()
        if ln:
            cleaned_lines.append(ln)

    if not cleaned_lines:
        return None, "rejected_empty_after_clean"

    text = "\n".join(cleaned_lines)
    text = RE_MULTI_NEWLINE.sub("\n\n", text).strip()

    char_len = len(text)
    words = text.split()
    word_len = len(words)

    # Stage 5 & 6: Source-Aware Length, Script Density, and Boilerplate Filtering
    odia_ratio, latin_ratio, non_ws_count = compute_script_metrics(text)

    if is_english_anchor:
        if char_len < 250 or word_len < 40:
            return None, "rejected_en_too_short"
        if latin_ratio < 0.70:
            return None, "rejected_en_low_latin_ratio"
        return text, "accepted_en"

    # Odia corpus rules: check if single-sentence parallel corpus (e.g. samanantar)
    src_lower = (source_name or "").lower()
    is_parallel_sentence_source = "samanantar" in src_lower

    if is_parallel_sentence_source:
        if char_len < 40 or word_len < 6:
            return None, "rejected_samanantar_too_short"
        if odia_ratio < 0.70:
            return None, "rejected_low_odia_ratio"
        return text, "accepted_odia_sentence"

    # Standard multi-sentence Odia documents
    if char_len < 90 or word_len < 15:
        return None, "rejected_odia_too_short"

    if odia_ratio < 0.65:
        return None, "rejected_low_odia_ratio"

    # Boilerplate menu/sidebar check (only when document has >= 4 lines)
    if len(cleaned_lines) >= 4:
        short_lines = sum(1 for ln in cleaned_lines if len(ln.split()) < 4)
        if (short_lines / len(cleaned_lines)) > 0.35:
            return None, "rejected_short_line_boilerplate"

    return text, "accepted_odia_doc"

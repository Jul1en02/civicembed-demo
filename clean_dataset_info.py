#!/usr/bin/env python3
# clean_dataset_info.py

import json
import re
from pathlib import Path
from deep_translator import GoogleTranslator
from time import sleep

# ─── Helpers ──────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split on common sentence boundaries."""
    return re.split(r'(?<=[.!?]) +', text)

def dedupe_sentences(text: str) -> str:
    """
    Remove exact-duplicate sentences (case-insensitive), preserving order.
    """
    seen = set()
    out = []
    for sent in split_sentences(text):
        s = sent.strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return " ".join(out)

def translate_to_en(text: str) -> str:
    """
    Uses GoogleTranslator to translate any text into English.
    If it fails, returns the original.
    """
    try:
        # throttle lightly so we don't hammer the API
        sleep(0.1)
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# ─── Main cleaning logic ─────────────────────────────────────────────────────

def clean_record(rec: dict) -> dict:
    """
    Return a new dict with 'title' and 'description' translated into
    English and deduped.
    """
    out = dict(rec)  # shallow copy everything else through
    raw_title = rec.get("title", "")
    raw_desc  = rec.get("description", "")

    # 1) translate
    en_title = translate_to_en(raw_title) if raw_title else ""
    en_desc  = translate_to_en(raw_desc)  if raw_desc  else ""

    # 2) dedupe repeated sentences
    out["title"]       = dedupe_sentences(en_title)
    out["description"] = dedupe_sentences(en_desc)

    return out

def main():
    src = Path("dataset_info.json")
    dst = Path("dataset_info_en.json")

    if not src.exists():
        print(f"❌ Cannot find {src}")
        return

    # load entire file (should be a list of dicts)
    data = json.loads(src.read_text(encoding="utf-8"))
    print(f"🔄 Translating & cleaning {len(data)} records…")

    cleaned = []
    for rec in data:
        cleaned.append(clean_record(rec))

    dst.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ Wrote cleaned English‐only file to {dst}")

if __name__ == "__main__":
    main()

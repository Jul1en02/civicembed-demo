#!/usr/bin/env python3
# batch_translate_dataset_info.py

import json
import math
from pathlib import Path
from time import sleep
from deep_translator import GoogleTranslator

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IN_FILE      = Path("dataset_info.json")
BATCH_DIR    = Path("batches")
FINAL_OUT    = Path("dataset_info_translated.json")
NUM_BATCHES  = 100        # split into 100 pieces; adjust if you like

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def translate_to_en(text: str) -> str:
    """Translate text to English (throttled)."""
    try:
        sleep(0.1)  # gentle throttle
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not IN_FILE.exists():
        print(f"❌ Source not found: {IN_FILE}")
        return

    # load all records
    recs = json.loads(IN_FILE.read_text(encoding="utf-8"))
    total = len(recs)
    batch_size = math.ceil(total / NUM_BATCHES)

    # ensure batch directory exists
    BATCH_DIR.mkdir(exist_ok=True)
    # detect which batches are already done
    done = {
        int(p.name.split("_")[1].split(".")[0])
        for p in BATCH_DIR.glob("batch_*.json")
        if p.name.startswith("batch_")
    }
    start = (max(done) + 1) if done else 0
    print(f"🔄 Translating {total} records in {NUM_BATCHES} batches (≈{batch_size} each)")
    print(f"⏯ Resuming at batch {start}/{NUM_BATCHES}")

    # process missing batches
    for i in range(start, NUM_BATCHES):
        lo = i * batch_size
        hi = min(lo + batch_size, total)
        if lo >= hi:
            break

        batch_file = BATCH_DIR / f"batch_{i:03d}.json"
        if batch_file.exists():
            print(f"→ Skipping batch {i}/{NUM_BATCHES} (already exists)")
            continue

        slice_ = recs[lo:hi]
        out = []
        for rec in slice_:
            t = rec.copy()
            t["title"]       = translate_to_en(rec.get("title",""))
            t["description"] = translate_to_en(rec.get("description",""))
            out.append(t)

        batch_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Wrote batch {i}/{NUM_BATCHES} → {batch_file}")

    # if *all* batches exist, combine & cleanup
    existing = sorted(BATCH_DIR.glob("batch_*.json"))
    if len(existing) >= NUM_BATCHES:
        print("🔗 All batches complete—combining into final output")
        combined = []
        for bf in existing:
            combined.extend(json.loads(bf.read_text(encoding="utf-8")))
        FINAL_OUT.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Wrote combined file → {FINAL_OUT}")

        # delete intermediates
        for bf in existing:
            bf.unlink()
        print(f"🗑 Deleted {len(existing)} batch files from {BATCH_DIR}")

if __name__ == "__main__":
    main()

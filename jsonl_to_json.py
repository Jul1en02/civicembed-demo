import json
from pathlib import Path

in_path  = Path("/Users/juliencoquet/Desktop/CivicEmbed/data/opendata/opendataswiss_metadata.jsonl")
out_path = Path("./dataset_info.json")

with in_path.open("r", encoding="utf-8") as fin:
    lines = [ json.loads(l) for l in fin if l.strip() ]

# (Optionally) prune to only the fields your UI needs:
# keep = ("id","title","description","url","publisher")
# lines = [{k: rec[k] for k in keep if k in rec} for rec in lines]

with out_path.open("w", encoding="utf-8") as fout:
    json.dump(lines, fout, ensure_ascii=False, indent=2)

print(f"Wrote {len(lines)} records to {out_path}")

"""Fetch a bounded sample of FineWeb (nanochat's actual pretrain corpus) to a
plain-text file the TS pipeline can ingest. Mirrors nanochat's dataset.py: stream
real web documents (no full download) and write them, <|endoftext|>-separated,
until MAX_BYTES. Needs `datasets` (a data-acquisition boundary tool, like the Rust
tokenizer / ffmpeg). Run with a venv that has it:
  python3 -m venv /tmp/fwvenv && /tmp/fwvenv/bin/pip install datasets
  MAX_BYTES=100000000 /tmp/fwvenv/bin/python fetch-fineweb.py
"""
import os
from datasets import load_dataset

OUT = os.environ.get("CORPUS", "fineweb.txt")
MAX_BYTES = int(os.environ.get("MAX_BYTES", 100_000_000))

ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
written = 0
with open(OUT, "w") as f:
    for row in ds:
        t = (row["text"] or "").strip()
        if not t:
            continue
        f.write(t)
        f.write("\n<|endoftext|>\n")
        written += len(t) + 14
        if written >= MAX_BYTES:
            break
print(f"wrote {written} bytes ({MAX_BYTES} target) -> {OUT}")

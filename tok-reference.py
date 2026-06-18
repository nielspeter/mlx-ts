"""Ground-truth tokenizer fixtures from the real Qwen3 tokenizer via HF
`tokenizers`. tokenizer.ts must reproduce these ids and decoded text exactly.
Run: python3 tok-reference.py  ->  writes tok-fixtures.json
"""
import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")

tests = [
    "Hello world",
    " hello",
    "Don't  panic—42!",
    "café\n  tabs\there",
    "The quick brown fox jumps over the lazy dog.",
    "你好，世界",
    "こんにちは、世界！",
    "I ❤️ MLX 🚀",
    "1234567890",
    "def f(x):\n    return x + 1",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]

fixtures = []
for s in tests:
    enc = tok.encode(s)
    fixtures.append({
        "text": s,
        "ids": enc.ids,
        "decoded": tok.decode(enc.ids),            # skip_special_tokens=True (default)
    })

json.dump(fixtures, open("tok-fixtures.json", "w"), ensure_ascii=False, indent=1)
print(f"wrote tok-fixtures.json with {len(fixtures)} cases")

# Ground-truth GPT-2 BPE: encode/decode a set of strings with HF `tokenizers`
# (which applies the full ByteLevel pretokenizer + merges from gpt2-tokenizer.json)
# and dump fixtures for gpt2-tok-test.ts to match. This validates the TS BPE
# encoder is token-exact for GPT-2's r50k pretokenization.
import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("gpt2-tokenizer.json")
TEXTS = [
    "The capital of France is",
    "Hello, world! 1234567890",
    "GPT-2 was released in 2019 by OpenAI.",
    "    leading spaces and\ttabs\nand newlines",
    "Don't can't won't I'm we're they'll he'd",
    "Supercalifragilisticexpialidocious unsplittable_words",
    "Numbers: 3.14159 and 42000 and 007",
    "Café naïve résumé — em-dash and ünïcödë",
]
out = []
for t in TEXTS:
    ids = tok.encode(t).ids
    out.append({"text": t, "ids": ids, "decoded": tok.decode(ids)})
json.dump(out, open("gpt2-tok-fixtures.json", "w"))
print(f"wrote {len(out)} GPT-2 tokenizer fixtures")

"""Train a byte-level BPE tokenizer with the Rust HF `tokenizers` core (the same
library nanochat's tok_train uses). Tokenizer *training* (merge counting) is not
an MLX computation — it's a data-prep step at the boundary, like ffmpeg for audio
— so it runs in native Rust. The output models/tokenizer.json is consumed by our pure-TS
inference (`tokenizer.ts`), which is validated token-exact.
  VOCAB=2048 CORPUS=data/input.txt python3 tok-train.py
"""
import os, json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

VOCAB = int(os.environ.get("VOCAB", 2048))
CORPUS = os.environ.get("CORPUS", "data/input.txt")

tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)   # GPT-2 r50k pretokenization
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),             # all 256 bytes -> no OOV
    show_progress=False,
)
tok.train([CORPUS], trainer)
tok.save("models/tokenizer-trained.json")

# fixtures so tok-train-test.ts can verify the TS encoder reproduces the freshly
# trained Rust tokenizer token-for-token (and round-trips decode).
TEXTS = [
    "To be, or not to be: that is the question",
    "ROMEO:\nBut soft, what light through yonder window breaks?",
    "First Citizen, speak.",
    "123 and 456 numbers",
]
out = [{"text": t, "ids": tok.encode(t).ids, "decoded": tok.decode(tok.encode(t).ids)} for t in TEXTS]
json.dump(out, open("tests/tok-trained-fixtures.json", "w"))
print(f"trained byte-level BPE: vocab={tok.get_vocab_size()} on {CORPUS} -> models/tokenizer-trained.json")

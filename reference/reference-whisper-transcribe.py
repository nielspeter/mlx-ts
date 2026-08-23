# Reference greedy transcription via mlx_whisper's model + mel + tokenizer, so
# whisper-transcribe-test.ts can assert mlx-ts produces the identical token ids.
import mlx.core as mx
import numpy as np
from mlx_whisper.load_models import load_model
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, load_audio
from mlx_whisper.tokenizer import get_tokenizer

import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/jfk.flac"
model = load_model("mlx-community/whisper-tiny", dtype=mx.float16)
tok = get_tokenizer(multilingual=True, num_languages=99, language="en", task="transcribe")

mel = log_mel_spectrogram(pad_or_trim(load_audio(path), N_SAMPLES))   # [3000, 80]
af = model.encoder(mx.array(mel).astype(mx.float16)[None])
prompt = list(tok.sot_sequence_including_notimestamps)
tokens = list(prompt)
for _ in range(224):
    logits, _, _ = model.decoder(mx.array([tokens]), af)
    nxt = int(mx.argmax(logits[0, -1]).item())
    if nxt == tok.eot:
        break
    tokens.append(nxt)
gen = tokens[len(prompt):]
np.array(gen, dtype=np.int32).tofile("/tmp/whisper-tok.i32")
print("text:", tok.decode(gen).strip())
print("tokens:", gen)

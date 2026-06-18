# Reference for whisper.ts: run mlx_whisper's own encoder + decoder on a fixed
# (seeded) mel and token sequence, save the audio features and logits so
# whisper-test.ts can feed the identical inputs through the mlx-ts implementation
# and assert parity. Run with the venv python that has mlx-whisper.
import mlx.core as mx
import numpy as np
from mlx_whisper.load_models import load_model

model = load_model("mlx-community/whisper-tiny", dtype=mx.float16)

np.random.seed(0)
mel = (np.random.randn(1, 3000, 80) * 0.5).astype(np.float32)   # [1, 3000, n_mels]
tokens = np.array([[50258, 50259, 50359, 1029, 318]], dtype=np.int32)

af = model.encoder(mx.array(mel).astype(mx.float16))
mx.eval(af)
logits, _, _ = model.decoder(mx.array(tokens), af)
mx.eval(logits)

afn = np.array(af.astype(mx.float32))
lg = np.array(logits.astype(mx.float32))
mel.tofile("/tmp/whisper-mel.f32")
tokens.tofile("/tmp/whisper-tokens.i32")
afn.tofile("/tmp/whisper-enc.f32")
lg.tofile("/tmp/whisper-logits.f32")
print(f"enc {tuple(af.shape)} sum={afn.sum():.3f}  logits {tuple(logits.shape)} argmax_last={int(lg[0,-1].argmax())}")

# Oracle for BiCodec's decode path, using mlx-audio's own implementation.
#
# Dumps a fingerprint at every boundary, not just the waveform: the pipeline is
# quantizer -> speaker d-vector -> prenet -> wave generator, and a mismatch at
# the end says nothing about which of the four is wrong.
#
#   /tmp/sdvenv/bin/python reference/reference-bicodec.py
import os
import mlx.core as mx
from mlx_audio.tts.models.spark.bicodec import BiCodec

DIR = os.path.expanduser("~/.cache/mlx-ts/mlx-community/Spark-TTS-0.5B-bf16/BiCodec")
model = BiCodec.load_from_checkpoint(DIR)

def fp(tag, a):
    f = a.flatten()
    print(f"{tag} shape={list(a.shape)} mean={float(f.mean()):.6f} "
          f"absmean={float(mx.abs(f).mean()):.6f} "
          f"first4={[round(float(v), 5) for v in f[:4].tolist()]}")

# Deterministic tokens in range: 8192 semantic codes, 4096 global (4^6).
T = 16
semantic = mx.array([[(i * 137 + 11) % 8192 for i in range(T)]], dtype=mx.int32)
# The pipeline expands global tokens to [B, 1, 32] before detokenize, because
# detokenize swaps the last two axes; passing [B, 32] silently reshapes into
# nonsense (or, here, a shape error).
glob = mx.expand_dims(mx.array([[(i * 91 + 7) % 4096 for i in range(32)]], dtype=mx.int32), 1)

z_q = model.quantizer.detokenize(semantic.transpose(0, 1)).transpose(0, 2, 1)
fp("z_q      ", z_q)

d_vector = model.speaker_encoder.detokenize(glob)
fp("d_vector ", d_vector)

x = model.prenet(z_q, d_vector)
fp("prenet   ", x)

x = x + d_vector[..., None]
wav = model.decoder(x)
mx.eval(wav)
fp("wav      ", wav)

# Oracle for BiCodec's speaker encoder — the encode half, which is what voice
# cloning needs.
#
# Dumps a fingerprint at every boundary: mel -> ECAPA features -> x-vector ->
# perceiver latents -> the 32 global tokens. A mismatch in the tokens alone says
# nothing about which of the four produced it.
#
# Deterministic synthetic audio, exactly the 6 s window the model crops to, so
# this needs no audio file and no decoder.
#
#   /tmp/sdvenv/bin/python reference/reference-speaker.py
import os

import mlx.core as mx
from mlx_audio.tts.models.spark.bicodec import BiCodec

DIR = os.path.expanduser("~/.cache/mlx-ts/mlx-community/Spark-TTS-0.5B-bf16/BiCodec")
model = BiCodec.load_from_checkpoint(DIR)
# mlx-audio leaves the model in training mode, so ECAPA's BatchNorms try to
# compute batch statistics from one utterance and throw. Inference wants the
# running stats — which is also why our port has no training path.
model.train(False)

N = 6 * 16000 // 320 * 320          # the model's own reference window: 96000
wav = mx.array([((i * 131 + 7) % 1009) / 1009 - 0.5 for i in range(N)], dtype=mx.float32)


def fp(tag, a):
    f = a.astype(mx.float32).flatten()
    mx.eval(f)
    print(f"{tag:10s} shape={list(a.shape)} mean={float(f.mean()):.6f} "
          f"absmean={float(mx.abs(f).mean()):.6f} "
          f"first4={[round(float(v), 5) for v in f[:4].tolist()]}")


mel = model.get_mel_spectrogram(wav[None, ...])
fp("mel", mel)

se = model.speaker_encoder
x_vector, features = se.speaker_encoder(mel, True)
fp("features", features)
fp("x_vector", x_vector)

latents = se.perceiver_sampler(features.transpose(0, 2, 1))
fp("latents", latents)

indices = se.tokenize(mel)
mx.eval(indices)
print("tokens:", mx.array(indices).flatten().tolist())

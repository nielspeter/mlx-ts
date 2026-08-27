# Oracle for BiCodec's speaker encoder — the encode half, which voice cloning needs.
#
# This one runs the ORIGINAL PyTorch Spark-TTS, not mlx-audio, because mlx-audio
# is wrong here in two ways:
#
#   1. Its mel left-aligns a short window inside n_fft. torch.stft centres it,
#      and the checkpoint was trained that way. The result still looks like a
#      spectrogram, and it moved one of the 32 token ids.
#   2. It never leaves training mode, so ECAPA's BatchNorms try to compute batch
#      statistics from a single utterance and throw. Its cloning path does not
#      run as shipped; the original calls model.eval() at load.
#
# Everything downstream of the mel (ECAPA, perceiver, FSQ) does agree between the
# two, but the authoritative reference is the model that produced the weights.
#
# Setup — the Spark-TTS package is not on PyPI, so fetch the modules it needs:
#
#   /tmp/sdvenv/bin/pip install torch torchaudio einops einx safetensors
#   mkdir -p /tmp/sparktts/sparktts/modules/{speaker,fsq}
#   cd /tmp/sparktts && for d in sparktts sparktts/modules sparktts/modules/speaker \
#     sparktts/modules/fsq; do touch $d/__init__.py; done
#   B=https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/sparktts
#   for f in modules/speaker/ecapa_tdnn.py modules/speaker/perceiver_encoder.py \
#            modules/speaker/speaker_encoder.py modules/speaker/pooling_layers.py \
#            modules/fsq/finite_scalar_quantization.py modules/fsq/residual_fsq.py; do
#     curl -sL "$B/$f" -o "sparktts/$f"; done
#
#   MLX_SPARK=/tmp/sparktts /tmp/sdvenv/bin/python reference/reference-speaker.py
import os
import sys

sys.path.insert(0, os.environ.get("MLX_SPARK", "/tmp/sparktts"))

import numpy as np
import torch
import torchaudio.transforms as TT
from safetensors.torch import load_file
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder

CKPT = os.path.expanduser(
    "~/.cache/mlx-ts/mlx-community/Spark-TTS-0.5B-bf16/BiCodec/model.safetensors")

W = load_file(CKPT)
sd = {k[len("speaker_encoder."):]: v.float() for k, v in W.items()
      if k.startswith("speaker_encoder.")}

model = SpeakerEncoder(input_dim=128, out_dim=1024, latent_dim=128, token_num=32,
                       fsq_levels=[4, 4, 4, 4, 4, 4], fsq_num_quantizers=1)
missing, unexpected = model.load_state_dict(sd, strict=True)
model.eval()

# Deterministic synthetic audio, exactly the 6 s window the model crops to, so
# this needs no audio file and no decoder.
N = 6 * 16000 // 320 * 320
wav = torch.tensor([((i * 131 + 7) % 1009) / 1009 - 0.5 for i in range(N)],
                   dtype=torch.float32)

# Built exactly as the original BiCodec.init_mel_transformer does.
mel_fn = TT.MelSpectrogram(16000, 1024, 640, 320, 10, None,
                           n_mels=128, power=1, norm="slaney", mel_scale="slaney")


def fp(tag, a):
    print(f"{tag:10s} shape={list(a.shape)} mean={a.mean():.6f} "
          f"absmean={a.abs().mean():.6f} "
          f"first4={[round(float(v), 5) for v in a.flatten()[:4]]}")


with torch.no_grad():
    mel = mel_fn(wav[None, ...]).squeeze(1).transpose(1, 2)      # [1, frames, mels]
    fp("mel", mel)

    x_vector, features = model.speaker_encoder(mel, True)
    fp("features", features)
    fp("x_vector", x_vector)

    latents = model.perceiver_sampler(features.transpose(1, 2))
    fp("latents", latents)

    print("tokens:", model.tokenize(mel).flatten().tolist())

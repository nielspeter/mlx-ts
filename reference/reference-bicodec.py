# Oracle for BiCodec's decode path, using the ORIGINAL PyTorch Spark-TTS.
#
# Dumps a fingerprint at every boundary, not just the waveform: the pipeline is
# quantizer -> speaker d-vector -> prenet -> wave generator, and a mismatch at
# the end says nothing about which of the four is wrong.
#
# PyTorch rather than mlx-audio, deliberately. Comparing one port against another
# can only show they agree, and mlx-audio is wrong here: its WNConvTranspose1d
# passes `groups` positionally into conv_transpose1d's `output_padding` slot, so
# every upsampling stage emits one extra sample and 16 frames come out as 5171
# rather than 320*16 = 5120. PyTorch gives 5120, which is what this checkpoint
# was trained as.
#
# Setup — the Spark-TTS package is not on PyPI, see reference-speaker.py for the
# curl commands; this needs these modules on top of the ones listed there:
#
#   models/bicodec.py  modules/blocks/{layers,samper,vocos}.py
#   modules/encoder_decoder/{feat_decoder,feat_encoder,wave_generator}.py
#   modules/vq/factorized_vector_quantize.py  utils/file.py
#   /tmp/sdvenv/bin/pip install torch omegaconf soundfile
#
#   MLX_SPARK=/tmp/sparktts /tmp/sdvenv/bin/python reference/reference-bicodec.py
import os
import sys

sys.path.insert(0, os.environ.get("MLX_SPARK", "/tmp/sparktts"))

import torch
from sparktts.models.bicodec import BiCodec

DIR = os.path.expanduser("~/.cache/mlx-ts/mlx-community/Spark-TTS-0.5B-bf16/BiCodec")
model = BiCodec.load_from_checkpoint(DIR)
model.eval()


def fp(tag, a):
    f = a.flatten().float()
    print(f"{tag:10s} shape={list(a.shape)} mean={f.mean():.6f} "
          f"absmean={f.abs().mean():.6f} "
          f"first4={[round(float(v), 5) for v in f[:4]]}")


# Deterministic tokens in range: 8192 semantic codes, 4096 global (4^6).
T = 16
semantic = torch.tensor([[(i * 137 + 11) % 8192 for i in range(T)]], dtype=torch.long)
# The pipeline expands global tokens to [B, 1, 32] before detokenize, because
# detokenize swaps the last two axes; passing [B, 32] silently reshapes into
# nonsense (or, here, a shape error).
glob = torch.tensor([[(i * 91 + 7) % 4096 for i in range(32)]], dtype=torch.long).unsqueeze(1)

with torch.no_grad():
    z_q = model.quantizer.detokenize(semantic)          # [B, 1024, T], channels-first
    fp("z_q", z_q)

    d_vector = model.speaker_encoder.detokenize(glob)
    fp("d_vector", d_vector)

    x = model.prenet(z_q, d_vector)
    fp("prenet", x)

    wav = model.decoder(x + d_vector.unsqueeze(-1))
    fp("wav", wav)

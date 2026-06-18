# Oracle for the iSTFT vocoder spike: run mlx-audio's istft on a seeded random
# complex spectrum and save inputs + output, so spike-istft.ts can reproduce it
# in mlx-c/TS (inverse rfft as a DFT matmul + windowed overlap-add) and compare.
import mlx.core as mx
import numpy as np
from mlx_audio.dsp import istft

N_FFT, HOP = 400, 100
BINS = N_FFT // 2 + 1
F = 50

np.random.seed(0)
re = np.random.randn(BINS, F).astype(np.float32)
im = np.random.randn(BINS, F).astype(np.float32)
x = mx.array((re + 1j * im).astype(np.complex64))   # [BINS, F]

y = istft(x, hop_length=HOP, win_length=N_FFT, window="hann", center=True, normalized=False)
mx.eval(y)
yn = np.array(y).astype(np.float32)

re.tofile("/tmp/istft-re.f32")
im.tofile("/tmp/istft-im.f32")
yn.tofile("/tmp/istft-ref.f32")
print(f"bins={BINS} frames={F} -> waveform len={yn.shape[0]} sum={yn.sum():.4f} sumsq={(yn*yn).sum():.4f}")

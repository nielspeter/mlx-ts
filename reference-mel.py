# Reference log-Mel spectrogram (Whisper front-end params) using numpy's FFT as
# the oracle. Saves the input PCM and the reference log-mel so audio-test.ts can
# feed the identical PCM through the mlx-ts pipeline (which computes the rfft as a
# DFT matmul) and assert allclose — validating the FFT-equivalent + framing/mel.
import numpy as np

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 80

np.random.seed(0)
pcm = (np.random.randn(SR) * 0.1).astype(np.float32)  # 1 s of deterministic noise

def hann(N):
    n = np.arange(N)
    return (0.5 * (1 - np.cos(2 * np.pi * n / N))).astype(np.float32)  # periodic Hann

def frames(x):
    x = np.pad(x, N_FFT // 2, mode="reflect")           # center padding
    nf = 1 + (len(x) - N_FFT) // HOP
    W = hann(N_FFT)
    return np.stack([x[i * HOP : i * HOP + N_FFT] * W for i in range(nf)])  # [F, 400]

def mel_filters():
    hz2mel = lambda f: 2595.0 * np.log10(1 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1)
    mpts = np.linspace(hz2mel(0), hz2mel(SR / 2), N_MELS + 2)
    bins = np.floor((N_FFT + 1) * mel2hz(mpts) / SR).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l: fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c: fb[m - 1, k] = (r - k) / (r - c)
    return fb  # [80, 201]

F = frames(pcm)                              # [nf, 400]
spec = np.fft.rfft(F, axis=1)                # [nf, 201] complex
power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
mel = power @ mel_filters().T                # [nf, 80]
logmel = np.log10(np.maximum(mel, 1e-10))
logmel = np.maximum(logmel, logmel.max() - 8.0)
logmel = ((logmel + 4.0) / 4.0).astype(np.float32)  # Whisper normalization

pcm.tofile("/tmp/mel-pcm.f32")
logmel.tofile("/tmp/mel-ref.f32")
print(f"frames={F.shape[0]} mels={N_MELS} sum={logmel.sum():.4f} sum_sq={(logmel ** 2).sum():.4f}")

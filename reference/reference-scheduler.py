# Oracle for the noise schedule, using mlx-examples' own sampler.
#
# Printed to 3-4 decimals: MLX builds the schedule in float32 and the TypeScript
# side in float64, so the last digit legitimately differs (0.029168 vs
# 0.029167). Four decimals still catches a wrong schedule by a mile.
#   MLX_SD=/path /tmp/sdvenv/bin/python reference/reference-scheduler.py
import os, sys
sys.path.insert(0, os.environ.get("MLX_SD", "/tmp/mlxsd_pkg"))
from stable_diffusion.config import DiffusionConfig
from stable_diffusion.sampler import SimpleEulerSampler

s = SimpleEulerSampler(DiffusionConfig())
print(f"max_time={s.max_time}")
print("sigmas_first4=" + ", ".join(f"{float(v):.4f}" for v in s._sigmas[:4].tolist()))
print("sigmas_last3=" + ", ".join(f"{float(v):.3f}" for v in s._sigmas[-3:].tolist()))
for t in [0.0, 1.0, 12.5, 500.0, 999.5, 1000.0]:
    print(f"sigma({t:.1f})={float(s.sigmas(__import__('mlx').core.array(t))):.3f}")
ts = s.timesteps(8)
print("timesteps8=" + ", ".join(f"({float(a):.1f}->{float(b):.1f})" for a, b in ts))

// TS side of the noise-schedule parity check.
//
// The schedule is where a diffusion port quietly goes wrong: SD trains on
// "scaled_linear" betas — linear in sqrt(beta), not in beta — and the plain
// linear one still produces an image, just a worse one. No weights involved, so
// a mismatch here is pure arithmetic.
//
// Compared at 3-4 decimals: MLX builds the schedule in float32 and this side in
// float64, so the last digit differs by right. Four decimals still catches a
// wrong schedule by a mile.
//   /tmp/sdvenv/bin/python reference/reference-scheduler.py && bun validation/scheduler.ts
import { EulerSampler } from "../src/models/diffusion.ts";

const s = new EulerSampler({
  beta_start: 0.00085, beta_end: 0.012,
  beta_schedule: "scaled_linear", num_train_timesteps: 1000,
});

console.log(`max_time=${s.maxTime}`);
console.log("sigmas_first4=" + Array.from(s.sigmas.slice(0, 4)).map((v) => v.toFixed(4)).join(", "));
console.log("sigmas_last3=" + Array.from(s.sigmas.slice(-3)).map((v) => v.toFixed(3)).join(", "));
for (const t of [0, 1, 12.5, 500, 999.5, 1000]) console.log(`sigma(${t.toFixed(1)})=${s.sigma(t).toFixed(3)}`);
console.log("timesteps8=" + s.timesteps(8).map(([a, b]) => `(${a.toFixed(1)}->${b.toFixed(1)})`).join(", "));

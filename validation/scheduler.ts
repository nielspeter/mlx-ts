// The noise schedule. No weights, so a mismatch is pure arithmetic — and this is
// where a diffusion port quietly goes wrong: SD trains on "scaled_linear" betas,
// linear in sqrt(beta), and the plain linear one still makes an image, just a
// worse one.
//
// Reference: diffusers' EulerDiscreteScheduler (see validation/golden.ts).
//
// Ours reparametrises to continuous time — sigma(0) = 0, and sigma(t) is
// diffusers' sigmas[t-1] at integer t, interpolating between — so the check is
// that mapping rather than an array compare.
//   bun validation/scheduler.ts
import { EulerSampler } from "../src/models/diffusion.ts";
import { checkNum, loadGolden, verdict } from "./golden.ts";

const g = loadGolden("sd-golden.json").scheduler;
const s = new EulerSampler({
  beta_start: g.betas.beta_start,
  beta_end: g.betas.beta_end,
  beta_schedule: g.betas.schedule,
  num_train_timesteps: g.betas.num_train_timesteps,
});

// A leading zero, then diffusers' curve — one sigma per training step.
checkNum("sigma count", s.sigmas.length, g.sigmas_len + 1);
checkNum("sigma(0)", s.sigma(0), 0);
for (const [t, want] of Object.entries(g.at as Record<string, number>)) {
  checkNum(`sigma(${t})`, s.sigma(Number(t)), want, 1e-3);
}
// Interpolated between integer steps, not snapped to one of them.
checkNum("sigma(12.5)", s.sigma(12.5), (g.at["12"] + g.at["13"]) / 2, 1e-3);
for (const [i, want] of (g.last3 as number[]).entries()) {
  checkNum(`sigmas[-${3 - i}]`, s.sigmas[s.sigmas.length - 3 + i], want, 1e-3);
}
verdict("noise schedule");

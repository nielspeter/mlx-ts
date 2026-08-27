// The sampling half of Stable Diffusion: a noise schedule and an Euler step.
//
// No weights, so nothing here can be loaded wrongly — but the schedule is where
// a port quietly goes astray. SD trains on a "scaled_linear" beta schedule,
// which is linear in sqrt(beta), not in beta; using the plain linear one still
// produces an image, just a worse one.
//
// The sigma maths is scalar, so it stays in plain JS and only the Euler step
// touches arrays.
import type { MX } from "../core/mx.ts";

export type DiffusionConfig = {
  beta_start: number;
  beta_end: number;
  beta_schedule: string;
  num_train_timesteps: number;
};

export class EulerSampler {
  /** sigmas[0] = 0, then one per training step; index is continuous time. */
  readonly sigmas: Float64Array;

  constructor(cfg: DiffusionConfig) {
    const { beta_start: b0, beta_end: b1, num_train_timesteps: N } = cfg;
    const scaled = cfg.beta_schedule !== "linear";
    const lo = scaled ? Math.sqrt(b0) : b0;
    const hi = scaled ? Math.sqrt(b1) : b1;

    this.sigmas = new Float64Array(N + 1);
    let cumprod = 1;
    for (let i = 0; i < N; i++) {
      const beta0 = lo + ((hi - lo) * i) / (N - 1);
      cumprod *= 1 - (scaled ? beta0 * beta0 : beta0);
      this.sigmas[i + 1] = Math.sqrt((1 - cumprod) / cumprod);
    }
  }

  get maxTime(): number { return this.sigmas.length - 1; }

  /** Linear interpolation into the schedule; time runs continuously. */
  sigma(t: number): number {
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, this.sigmas.length - 1);
    return this.sigmas[lo] * (1 - (t - lo)) + this.sigmas[hi] * (t - lo);
  }

  /** `steps` consecutive (t, tPrev) pairs, walking maxTime down to 0. */
  timesteps(steps: number, startTime = this.maxTime): Array<[number, number]> {
    const pts = Array.from({ length: steps + 1 }, (_, i) => startTime - (startTime * i) / steps);
    return pts.slice(0, -1).map((t, i) => [t, pts[i + 1]] as [number, number]);
  }

  /** How much to scale unit-variance noise to start the walk. */
  priorScale(): number {
    const s = this.sigmas[this.sigmas.length - 1];
    return s / Math.sqrt(s * s + 1);
  }

  /** One Euler step from x_t towards x_{t_prev}, given the predicted noise. */
  step(epsPred: MX, xT: MX, t: number, tPrev: number): MX {
    const s = this.sigma(t), sPrev = this.sigma(tPrev);
    return xT.mulScalar(Math.sqrt(s * s + 1))
      .add(epsPred.mulScalar(sPrev - s))
      .divScalar(Math.sqrt(sPrev * sPrev + 1));
  }
}

// Loss functions over MX.
import { MX } from "./mx.ts";

// Cross-entropy for next-token prediction.
// logits: [M, V], targets: [M, 1] int -> scalar mean negative log-likelihood.
export function crossEntropy(logits: MX, targets: MX): MX {
  const logProbs = logits.softmax(1).log();        // log_softmax (mlx softmax is max-stable)
  return logProbs.takeAlong(targets, 1).neg().meanAll();
}

// Loss functions over MX.
import { MX } from "./mx.ts";

// Cross-entropy for next-token prediction.
// logits: [M, V], targets: [M, 1] int -> scalar mean negative log-likelihood.
export function crossEntropy(logits: MX, targets: MX): MX {
  // Numerically stable log_softmax: x - logsumexp(x). The naive softmax(x).log()
  // is fine in the forward but its BACKWARD produces NaN once probabilities
  // saturate (model near-certain on a token, e.g. memorizing SFT examples).
  const logProbs = logits.sub(logits.logsumexp(1, true));
  return logProbs.takeAlong(targets, 1).neg().meanAll();
}

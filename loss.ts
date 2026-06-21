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

// Masked cross-entropy for BATCHED SFT: average NLL over only the positions where
// mask==1 (assistant-response tokens; prompt + padding masked to 0). logits [M,V],
// targets [M,1] int, mask [M,1]. Stable log_softmax keeps it NaN-free even where
// the model saturates on masked positions (inf*0 can't arise — logprobs finite).
export function maskedCrossEntropy(logits: MX, targets: MX, mask: MX): MX {
  const logProbs = logits.sub(logits.logsumexp(1, true));
  const nll = logProbs.takeAlong(targets, 1).neg().mul(mask);
  return nll.sumAxes([0, 1], false).div(mask.sumAxes([0, 1], false));
}

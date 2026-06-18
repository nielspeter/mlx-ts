// TS port of benchmarks/python/single_ops.py (forward ops only).
//   bun benchmarks/single-ops.ts

import { fromI32, MX } from "../mx.ts";
import { randomUniform, maximum, maxAxis, minAxis, exp, negative, logsumexpAxis, timeFn } from "./time-utils.ts";

function timeAdd() {
  const a = randomUniform([32, 1024, 1024]); const b0 = randomUniform([32, 1024, 1024]);
  a.eval(); b0.eval();
  timeFn(() => a.add(b0), "add");

  const aT = a.transpose([0, 2, 1]); aT.eval();
  timeFn(() => aT.add(b0), "transpose_add");

  const b1 = randomUniform([1024]); b1.eval();
  timeFn(() => a.add(b1), "slice_add");

  const b2 = b1.reshape([1, 1024, 1]); b2.eval();
  timeFn(() => a.add(b2), "mid_slice_add");
}

function timeMatmul() {
  const a = randomUniform([1024, 1024]); const b = randomUniform([1024, 1024]);
  a.eval(); b.eval();
  timeFn(() => a.matmul(b), "matmul");
}

function timeMaximum() {
  const a = randomUniform([32, 1024, 1024]); const b = randomUniform([32, 1024, 1024]);
  a.eval(); b.eval();
  timeFn(() => maximum(a, b), "maximum");
}

function timeMaxMin() {
  const a = randomUniform([32, 1024, 1024]); a.eval();
  timeFn(() => maxAxis(a, 0), "max(axis=0)");
  timeFn(() => minAxis(a, 0), "min(axis=0)");
}

function timeExp() {
  const a = randomUniform([1000, 100]); a.eval();
  timeFn(() => exp(a), "exp");
}

function timeNegative() {
  const a = randomUniform([10000, 1000]); a.eval();
  timeFn(() => negative(a), "negative");
}

function timeLogsumexp() {
  const a = randomUniform([64, 10, 10000]); a.eval();
  timeFn(() => logsumexpAxis(a, -1), "logsumexp(axis=-1)");
}

function timeTake() {
  const a = randomUniform([10000, 500]); a.eval();
  const ids = Array.from({ length: 20 }, () =>
    fromI32(Int32Array.from({ length: 10 }, () => Math.floor(Math.random() * 10000)), [10]));
  ids.forEach((i) => i.eval());
  timeFn(() => ids.map((idx) => a.takeAxis(idx, 0)) as MX[], "take");
}

function timeReshapeTransposed() {
  const x = randomUniform([256, 256, 128]); x.eval();
  timeFn(() => x.transpose([1, 0, 2]).reshape([256 * 256 * 128]), "reshape_transposed");
}

timeAdd();
timeMatmul();
timeMaxMin();
timeMaximum();
timeExp();
timeNegative();
timeLogsumexp();
timeTake();
timeReshapeTransposed();

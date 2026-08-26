// Writing a custom Metal kernel from TypeScript.
//
// MLX has no fused LSTM, so Apple's EnCodec ships its own Metal kernel for one.
// This is that kernel — source verbatim from mlx-examples/musicgen/encodec.py —
// dispatched through mlx-ts. Its output matches MLX Python to every digit,
// which the parity suite checks on every run.
//
// Needs no model files.
//   bun examples/metal-kernel.ts
import { metalKernel, scalarI32, fromF32, tidy } from "../src/index.ts";

const lstm = metalKernel({
  name: "lstm",
  inputNames: ["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
  outputNames: ["hidden_state", "cell_state"],
  // `header` is for anything the body needs declared first — Metal has no
  // built-in sigmoid.
  header: `
    template <typename T>
    T sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
  `,
  // The body is one thread's work. MLX wraps it in a kernel signature built
  // from inputNames/outputNames, so `x`, `h_in`, `hidden_size` etc. are in
  // scope with the types their arguments imply.
  source: `
        uint b = thread_position_in_grid.x;
        uint d = hidden_size * 4;

        uint elem = b * d + thread_position_in_grid.y;
        uint index = elem;
        uint x_index = b * num_time_steps * d + time_step * d + index;

        auto i = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto f = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto g = metal::precise::tanh(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto o = sigmoid(h_in[index] + x[x_index]);

        cell_state[elem] = f * cell[elem] + i * g;
        hidden_state[elem] = o * metal::precise::tanh(cell_state[elem]);
  `,
});

const B = 2, H = 4, T = 3;
// Deterministic, so reference/reference-metal-kernel.py reproduces it exactly.
const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);

const x = fromF32(det(B * T * H * 4, 1), [B, T, H * 4]);
const hIn = fromF32(det(B * H * 4, 2), [B, H * 4]);
const cell = fromF32(det(B * H, 3), [B, H]);

const [hidden, cellOut] = tidy(() =>
  lstm.apply(
    // scalarI32, not fromF32: a shaped array would bind as a device buffer and
    // the shader would fail to compile.
    [x, hIn, cell, scalarI32(H), scalarI32(0), scalarI32(T)],
    [{ shape: [B, H] }, { shape: [B, H] }],   // output shapes
    [B, H * 4, 1],                            // grid
    [256, 1, 1],                              // threadgroup
  ));

console.log("=== custom Metal kernel from TypeScript (Apple's LSTM, verbatim) ===");
console.log(`  hidden_state ${JSON.stringify(hidden.shape)}: ${hidden.toF32().map((v) => +v.toFixed(6)).join(", ")}`);
console.log(`  cell_state   ${JSON.stringify(cellOut.shape)}: ${cellOut.toF32().map((v) => +v.toFixed(6)).join(", ")}`);

# Oracle for spikes/spike-metal-kernel.ts: the SAME custom Metal kernel (Apple's
# fused LSTM step from mlx-examples/musicgen/encodec.py), the same deterministic
# inputs, run through mx.fast.metal_kernel in MLX Python.
#   python3 reference/reference-metal-kernel.py
import mlx.core as mx

_lstm_kernel = mx.fast.metal_kernel(
    name="lstm",
    input_names=["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
    output_names=["hidden_state", "cell_state"],
    header="""
    template <typename T>
    T sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
    """,
    source="""
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
    """,
)

B, H, T = 2, 4, 3
det = lambda n, seed: mx.array([((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5 for i in range(n)], dtype=mx.float32)

x = det(B * T * H * 4, 1).reshape(B, T, H * 4)
h_in = det(B * H * 4, 2).reshape(B, H * 4)
cell = det(B * H, 3).reshape(B, H)

hidden, cell_out = _lstm_kernel(
    inputs=[x, h_in, cell, H, 0, T],
    output_shapes=[(B, H), (B, H)],
    output_dtypes=[mx.float32, mx.float32],
    grid=(B, h_in.size // 4, 1),   # Apple's own expression: B*H
    threadgroup=(256, 1, 1),
)
mx.eval(hidden, cell_out)
fmt = lambda a: ", ".join(f"{v:.6f}".rstrip("0").rstrip(".") if abs(v) > 1e-9 else "0" for v in a.flatten().tolist())
print(f"  hidden_state [{B},{H}]: {fmt(hidden)}")
print(f"  cell_state   [{B},{H}]: {fmt(cell_out)}")

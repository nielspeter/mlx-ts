// Live transcription from the microphone.
//
//   bun examples/parakeet-mic.ts              # default input device
//   bun examples/parakeet-mic.ts --list       # what devices exist
//   bun examples/parakeet-mic.ts --device 1
//   bun examples/parakeet-mic.ts --look 4     # less lag, more errors
//
// This is examples/parakeet-live.ts with a real microphone instead of a file
// played at microphone pace — and it should feel the same, because that example
// feeds the model no faster than the sound would arrive.
//
// It is not instant, and cannot be. A word appears one lookahead after it is
// spoken at best, one chunk plus one lookahead at worst: 1.04-2.08 s with the
// defaults. Almost none of that is compute, which runs around 0.12 s per chunk
// — the wait is the encoder needing a little future audio to be accurate.
// `--look` buys latency back at the cost of word error (see parakeet-stream.ts).
//
// macOS will ask for microphone permission the first time. If nothing arrives,
// grant it to your terminal in System Settings -> Privacy & Security.
import { spawn } from "node:child_process";
import { type ParakeetConfig, ParakeetStream, ParakeetTokenizer, singleFileWeights } from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";

const argv = process.argv.slice(2);
const flag = (name: string, def: number): number => {
  const i = argv.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = Number(argv[i + 1]);
  argv.splice(i, 2);
  return v;
};
const chunkFrames = flag("chunk", 13);
const lookaheadFrames = flag("look", 13);
const device = flag("device", 0);

if (argv.includes("--list")) {
  // ffmpeg prints the device table to stderr and then exits with an error,
  // because listing is not a real capture. That error is expected.
  const p = spawn("ffmpeg", ["-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]);
  p.stderr.on("data", (b) => process.stdout.write(String(b).replace(/^\[[^\]]+\] ?/gm, "")));
  p.on("close", () => process.exit(0));
} else {
  const REPO = "nvidia/parakeet-tdt-0.6b-v3";
  const cfg = await readJson<ParakeetConfig>(await hubFile(REPO, "config.json"));
  const W = singleFileWeights(await hubFile(REPO, "model.safetensors"));
  const tok = await ParakeetTokenizer.fromFile(await hubFile(REPO, "tokenizer.json"));
  const stream = new ParakeetStream(W, cfg, tok, { chunkFrames, lookaheadFrames });

  const SR = 16000;
  console.log(`\n  listening on device ${device} — ctrl-c to stop`);
  console.log(`  chunk ${((chunkFrames * 1280) / SR).toFixed(2)}s, lookahead ${((lookaheadFrames * 1280) / SR).toFixed(2)}s`);
  console.log(`  a word should appear ${((lookaheadFrames * 1280) / SR).toFixed(2)}-${stream.latencySeconds.toFixed(2)}s after you say it\n`);

  // Warm the graph so the first words are not waiting on kernel compilation.
  stream.push(new Float32Array(1600));

  // Raw 32-bit float samples straight to stdout: no container to parse, and no
  // decode step between the microphone and the model.
  const ff = spawn("ffmpeg", [
    "-hide_banner", "-loglevel", "error",
    "-f", "avfoundation", "-i", `:${device}`,
    "-ac", "1", "-ar", String(SR), "-f", "f32le", "-",
  ]);
  ff.stderr.on("data", (b) => process.stderr.write(String(b)));

  let stopped = false;
  const shutdown = () => {
    // Same reasoning as parakeet-live.ts: a second ctrl-c must not be swallowed,
    // and a throw in cleanup must not stop us leaving.
    if (stopped) process.exit(130);
    stopped = true;
    try {
      ff.kill("SIGKILL");
      const tail = stream.flush();
      if (tail) process.stdout.write(tail);
      stream.close();
      process.stdout.write(`\n\n  stopped — ${stream.text.trim().split(/\s+/).filter(Boolean).length} words\n\n`);
    } finally {
      process.exit(130);
    }
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  // ffmpeg writes whenever it likes, so a read can split a sample down the
  // middle. Whatever is left over is carried to the next one.
  let rest = new Uint8Array(0);
  process.stdout.write("  ");
  for await (const buf of ff.stdout) {
    if (stopped) break;
    const bytes = new Uint8Array(rest.length + buf.length);
    bytes.set(rest);
    bytes.set(buf, rest.length);
    const whole = bytes.length - (bytes.length % 4);
    rest = bytes.slice(whole);
    if (whole === 0) continue;
    const pcm = new Float32Array(bytes.buffer.slice(0, whole));
    const text = stream.push(pcm);
    if (text) process.stdout.write(text);
  }
}

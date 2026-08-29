// Hear the audio and watch the transcript arrive — the delay, felt rather than
// quoted.
//
//   bun examples/parakeet-live.ts audio.wav
//
// The file is played through the speakers and fed to the model at the same
// wall-clock rate a microphone would, so the words appear on screen exactly as
// far behind the sound as they would in a live meeting.
//
// Each line shows where the audio had got to and how far behind the text is.
// The lag is real: the encoder needs a little future audio to be accurate, so a
// word cannot be transcribed the instant it is spoken. `--look` and `--chunk`
// trade that lag against accuracy.
import { spawn } from "node:child_process";
import {
  decodeAudio,
  type ParakeetConfig,
  ParakeetStream,
  ParakeetTokenizer,
  singleFileWeights,
} from "../src/index.ts";
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
const silent = argv.includes("--silent") && (argv.splice(argv.indexOf("--silent"), 1), true);
// The model emits subwords, so a chunk can end mid-word ("country" arrives as
// ▁co + un + tr + y). Appending to one growing line hides that, the way a real
// caption does; --lines shows each emission separately and makes it visible.
const lines = argv.includes("--lines") && (argv.splice(argv.indexOf("--lines"), 1), true);

const path = argv.find((a) => !a.startsWith("--"));
if (!path) {
  console.error("usage: bun examples/parakeet-live.ts <audio-file> [--chunk 13] [--look 13] [--silent]");
  process.exit(1);
}

const REPO = "nvidia/parakeet-tdt-0.6b-v3";
const cfg = await readJson<ParakeetConfig>(await hubFile(REPO, "config.json"));
const W = singleFileWeights(await hubFile(REPO, "model.safetensors"));
const tok = await ParakeetTokenizer.fromFile(await hubFile(REPO, "tokenizer.json"));
const pcm = await decodeAudio(path);

const stream = new ParakeetStream(W, cfg, tok, { chunkFrames, lookaheadFrames, wholeWords: lines });
const SR = 16000;
const TICK = 1600; // 100 ms, about what a MediaRecorder hands you

console.log(`\n  ${path} — ${(pcm.length / SR).toFixed(1)}s`);
console.log(`  chunk ${(chunkFrames * 1280 / SR).toFixed(2)}s, lookahead ${(lookaheadFrames * 1280 / SR).toFixed(2)}s`);
console.log(`  a word should appear ${(lookaheadFrames * 1280 / SR).toFixed(2)}-${stream.latencySeconds.toFixed(2)}s after you hear it\n`);

// Warm the graph so the first chunk is not paying compile costs while the audio
// is already playing.
stream.push(new Float32Array(TICK));

const player = silent ? null : spawn("afplay", [path], { stdio: "ignore" });

// Ctrl-C has to stop the sound too. `afplay` is its own process, and killing
// this one does not kill it — without this the audio plays on to the end of the
// file after the transcript is gone.
let stopped = false;
const shutdown = () => {
  // A second Ctrl-C means the first one did not get us out. Leave immediately;
  // no cleanup is worth ignoring the user twice.
  if (stopped) process.exit(130);
  stopped = true;
  try {
    // SIGKILL, not SIGTERM: this is a media player with nothing to wind down,
    // and a polite signal can sit unhandled long enough to keep making noise.
    player?.kill("SIGKILL");
    stream.close();
    const words = stream.text.trim().split(/\s+/).filter(Boolean).length;
    process.stdout.write(`\n\n  stopped — ${words} words transcribed\n\n`);
  } finally {
    // Whatever went wrong above, still leave. Without the finally an exception
    // in the cleanup strands the process with the handler already latched, and
    // every later Ctrl-C is swallowed.
    process.exit(130);
  }
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

const t0 = performance.now();
const elapsed = () => (performance.now() - t0) / 1000;
let lag = "";
if (!lines) process.stdout.write("  ");

for (let i = 0; i < pcm.length && !stopped; i += TICK) {
  const audioAt = (i + TICK) / SR;
  // Feed no faster than the sound is playing.
  const wait = audioAt - elapsed();
  if (wait > 0) await new Promise((r) => setTimeout(r, wait * 1000));

  const text = stream.push(pcm.subarray(i, i + TICK));
  if (text) {
    const now = elapsed();
    // These words were spoken in the chunk that ends one lookahead behind the
    // audio we have fed. So the newest of them is `lookahead` old and the oldest
    // is `chunk + lookahead` old, plus however long the compute took.
    const overhead = now - audioAt;
    const newest = overhead + (lookaheadFrames * 1280) / SR;
    const oldest = overhead + ((chunkFrames + lookaheadFrames) * 1280) / SR;
    if (lines) {
      console.log(`  [audio ${audioAt.toFixed(1)}s]  spoken ${oldest.toFixed(1)}-${newest.toFixed(1)}s ago  ${text}`);
    } else {
      // One growing line, as a caption would render it.
      process.stdout.write(text);
      lag = `${oldest.toFixed(1)}-${newest.toFixed(1)}s`;
    }
  }
}

const tail = stream.flush();
stream.close();
if (lines) {
  if (tail) console.log(`  [audio ${(pcm.length / SR).toFixed(1)}s]  final chunk                ${tail}`);
} else {
  process.stdout.write(`${tail}\n`);
  console.log(`\n  words appeared ${lag} after they were spoken`);
}
player?.kill();

if (lines) console.log(`\n  ${stream.text}`);
console.log(`\n  Nothing above was ever revised — a transducer emits a token once`);
console.log(`  and moves on. What you waited for was the lookahead, not a window.\n`);

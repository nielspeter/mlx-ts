// Speech to text with Parakeet TDT — a recording in, a transcript out.
//
//   bun examples/parakeet.ts audio.wav
//   bun examples/parakeet.ts audio.wav --timestamps
//   bun examples/parakeet.ts audio.wav --srt > subs.srt
//
// NVIDIA's FastConformer transducer. Where Whisper sees a fixed 30 s window and
// decodes autoregressively, Parakeet's decoder walks the encoder frames and
// predicts how far to skip at each step — so it does far less work, and streams
// by construction.
//
// That walk is also why the timestamps are nearly free: the frame pointer
// already *is* a clock, one frame being 80 ms, so nothing extra is computed and
// the decode loop only records where it was when each token came out. An
// attention decoder has no such pointer, which is why timestamps there need a
// separate alignment pass. Word starts are the number to trust; ends come from
// the model's duration head, capped at four frames, so a word can overrun by
// up to ~0.3 s.
//
// First run downloads ~2.4 GB. Any format afconvert reads works.
import { Parakeet } from "../src/index.ts";

const argv = process.argv.slice(2);
const take = (name: string): boolean =>
  argv.includes(`--${name}`) && (argv.splice(argv.indexOf(`--${name}`), 1), true);
const srt = take("srt");
const timestamps = take("timestamps") || srt;

const path = argv.find((a) => !a.startsWith("--"));
if (!path) {
  console.error("usage: bun examples/parakeet.ts <audio-file> [--timestamps] [--srt]");
  process.exit(1);
}

const t0 = performance.now();
const model = await Parakeet.fromPretrained();
// An SRT file has to be pipeable, so nothing but cues may go to stdout.
const log = srt ? console.error : console.log;
log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const clock = (s: number, sep: string): string => {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${
    (s % 60).toFixed(3).padStart(6, "0").replace(".", sep)
  }`;
};

const t1 = performance.now();
if (!timestamps) {
  const text = await model.transcribeFile(path);
  console.log(`\n${text}\n`);
} else {
  const words = await model.wordsFromFile(path);
  if (srt) {
    // One cue per few seconds, as a subtitle file wants — not one per word,
    // which no player renders usefully.
    let n = 0;
    for (let i = 0; i < words.length; ) {
      let j = i;
      while (j < words.length && words[j].end - words[i].start < 3.5 && j - i < 12) j++;
      const line = words.slice(i, j);
      console.log(`${++n}\n${clock(line[0].start, ",")} --> ${clock(line[line.length - 1].end, ",")}`);
      console.log(`${line.map((w) => w.text).join(" ")}\n`);
      i = j;
    }
  } else {
    console.log(`\n  ${words.length} words\n`);
    for (const w of words) console.log(`  ${clock(w.start, ".")} → ${clock(w.end, ".")}   ${w.text}`);
    console.log("");
  }
}
log(`transcribed in ${((performance.now() - t1) / 1000).toFixed(2)}s`);

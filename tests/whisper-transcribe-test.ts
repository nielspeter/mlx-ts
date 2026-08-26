// Assert mlx-ts greedy transcription is token-for-token identical to mlx_whisper.
//   /tmp/wvenv/bin/python reference-whisper-transcribe.py && bun whisper-transcribe-test.ts

import { decodeAudio, loadMelFilters } from "../src/audio/mel.ts";
import { loadWhisper } from "../src/models/whisper.ts";
import { WhisperTokenizer } from "../src/text/whisper-tokenizer.ts";

const AUDIO = process.argv[2] ?? "/tmp/jfk.flac";
const model = await loadWhisper();
const tok = await WhisperTokenizer.fromFile();
const filtersT = await loadMelFilters();
const ids = model.transcribe(await decodeAudio(AUDIO), filtersT);

const refBuf = new Uint8Array(await Bun.file("/tmp/whisper-tok.i32").arrayBuffer());
const ref = [...new Int32Array(refBuf.buffer, refBuf.byteOffset, refBuf.byteLength / 4)];

const same = ids.length === ref.length && ids.every((t, i) => t === ref[i]);
console.log("text:", tok.decode(ids).trim());
console.log(`tokens: mlx-ts=${ids.length} ref=${ref.length} identical=${same}`);
console.log(same ? "TRANSCRIBE OK" : "TRANSCRIBE MISMATCH");

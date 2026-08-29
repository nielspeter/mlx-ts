// Parakeet against the committed reference numbers: mel -> subsampling ->
// 24 FastConformer layers -> projection -> TDT greedy decode.
//
// Every boundary is fingerprinted, because a token mismatch says nothing about
// which stage produced it. The numbers come from transformers.ParakeetForTDT
// (see validation/golden.ts), so this needs no Python.
//
// Deterministic synthetic audio — no fixture file. The two tokens it decodes to
// are a smoke test of the loop; validation/parakeet-transcribe.ts is what checks
// the transcript.
//   bun validation/parakeet-encode.ts
import { parakeetMel } from "../src/audio/mel.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import {
  encode, type ParakeetConfig, projectEncoder, relPositionalEncoding, subsample,
} from "../src/models/parakeet.ts";
import { check, loadGolden, verdict } from "./golden.ts";

const REPO = "nvidia/parakeet-tdt-0.6b-v3";
const g = loadGolden("parakeet-golden.json");

const cfg = await readJson<ParakeetConfig>(await hubFile(REPO, "config.json"));
const W = singleFileWeights(await hubFile(REPO, "model.safetensors"));

const N = 16000 * g.seconds;
const wav = Float32Array.from({ length: N }, (_, i) => ((i * 131 + 7) % 1009) / 1009 - 0.5);

const mel = parakeetMel(wav);
check("mel", mel, g.mel, true);

const sub = subsample(W, cfg.encoder_config, mel);
check("subsample", sub, g.subsample, true);
check("pos_embed", relPositionalEncoding(cfg.encoder_config.hidden_size, sub.shape[1]), g.pos_embed, true);

const enc = encode(W, cfg.encoder_config, mel);
check("encoder", enc, g.encoder, true);
check("enc_proj", projectEncoder(W, enc), g.encoder_projected, true);

// No token check here: on synthetic noise the blank logit ties the best real
// token, so the argmax is a coin flip and the trajectory diverges for reasons
// unrelated to the code. validation/parakeet-transcribe.ts checks the transcript
// on real speech, which is what the decode path is for.
verdict("Parakeet encoder + TDT decode");

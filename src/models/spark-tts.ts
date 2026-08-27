// Spark-TTS, end to end: text in, a 16 kHz waveform out.
//
// Two models in series. A Qwen2 language model predicts *audio* tokens — the
// vocabulary is extended past text with 4096 global (speaker) tokens and 8192
// semantic (content) tokens — and BiCodec turns those tokens back into sound.
// Both halves are checked separately against mlx-audio (validation/qwen2-*.ts
// and validation/bicodec-decode.ts), so this file is only the join.
//
// Voice *creation* is what runs here: gender, pitch and speed are given as
// control tokens and the LM invents a matching speaker. Voice *cloning* needs
// BiCodec's encoder — a mel front end, an ECAPA speaker encoder and a perceiver
// sampler, none of which are implemented, since generation never uses them.
import { decodeAudio, melSpectrogram } from "../audio/mel.ts";
import { fromI32, type MX, setCacheLimit, tidy } from "../core/mx.ts";
import { readJson } from "../io/fs.ts";
import { hubFile } from "../io/hub.ts";
import { singleFileWeights } from "../io/loader.ts";
import { streamTokens } from "../text/lm.ts";
import { Tokenizer } from "../text/tokenizer.ts";
import { BiCodecPrenet, BiCodecQuantizer, SpeakerDetokenizer, WaveGenerator } from "./bicodec.ts";
import { Qwen2, type Qwen2Config } from "./qwen2.ts";
import { referenceClip, SpeakerTokenizer, volumeNormalize } from "./speaker.ts";

const DEFAULT_REPO = "mlx-community/Spark-TTS-0.5B-bf16";

/** 16 kHz, fixed: BiCodec's 320x upsampling is trained at that rate. */
export const SPARK_SAMPLE_RATE = 16000;

// The audio tokens occupy contiguous id ranges, so the generated ids can be
// sorted numerically. mlx-audio decodes to text and runs a regex over it, which
// costs a round trip through the vocabulary and breaks if a token is unknown.
const GLOBAL_BASE = 151665; // <|bicodec_global_0|>    .. _4095
const GLOBAL_COUNT = 4096;
const SEMANTIC_BASE = 155761; // <|bicodec_semantic_0|>  .. _8191
const SEMANTIC_COUNT = 8192;

// Cloning writes the speaker into the prompt rather than letting the LM invent
// one, so it needs the markers that delimit that run of tokens.
const TASK_TTS = 165137; // <|task_tts|>
const START_CONTENT = 165146;
const END_CONTENT = 165152;
const START_GLOBAL = 165150;
const END_GLOBAL = 165156;

/**
 * Split a generated id stream into the two token kinds BiCodec needs, dropping
 * everything else — the LM also emits the structural markers that separate the
 * two runs, and text tokens if it strays.
 *
 * Returned 0-based, which is what the codebook and the FSQ expect.
 */
export function splitAudioTokens(tokens: Iterable<number>): { global: number[]; semantic: number[] } {
  const global: number[] = [],
    semantic: number[] = [];
  for (const t of tokens) {
    if (t >= SEMANTIC_BASE && t < SEMANTIC_BASE + SEMANTIC_COUNT) semantic.push(t - SEMANTIC_BASE);
    else if (t >= GLOBAL_BASE && t < GLOBAL_BASE + GLOBAL_COUNT) global.push(t - GLOBAL_BASE);
  }
  return { global, semantic };
}

export type Gender = "female" | "male";
/** Five levels, from `very_low` to `very_high`. */
export type Level = "very_low" | "low" | "moderate" | "high" | "very_high";

const GENDERS: Gender[] = ["female", "male"];
const LEVELS: Level[] = ["very_low", "low", "moderate", "high", "very_high"];

export type SpeechOptions = {
  gender?: Gender;
  pitch?: Level;
  speed?: Level;
  temp?: number;
  topP?: number;
  topK?: number;
  /** Spark's own default is 1.3; without it the LM loops on one semantic token. */
  repetitionPenalty?: number;
  seed?: number;
  /** Each token is 320 samples, so 3000 is a hair over a minute of audio. */
  maxTokens?: number;
  cacheLimitMB?: number;
  onToken?: (n: number) => void;
};

export class SparkTTS {
  private tok: Tokenizer;
  private lm: Qwen2;
  private quantizer: BiCodecQuantizer;
  private speaker: SpeakerDetokenizer;
  private prenet: BiCodecPrenet;
  private generator: WaveGenerator;
  private speakerEnc: SpeakerTokenizer;

  private constructor(
    tok: Tokenizer,
    lm: Qwen2,
    quantizer: BiCodecQuantizer,
    speaker: SpeakerDetokenizer,
    prenet: BiCodecPrenet,
    generator: WaveGenerator,
    speakerEnc: SpeakerTokenizer,
  ) {
    this.tok = tok;
    this.lm = lm;
    this.quantizer = quantizer;
    this.speaker = speaker;
    this.prenet = prenet;
    this.generator = generator;
    this.speakerEnc = speakerEnc;
  }

  static async fromPretrained(repo = DEFAULT_REPO): Promise<SparkTTS> {
    const tok = await Tokenizer.fromFile(await hubFile(repo, "tokenizer.json"));
    const cfg = await readJson<Qwen2Config>(await hubFile(repo, "config.json"));
    const lm = new Qwen2(cfg, singleFileWeights(await hubFile(repo, "model.safetensors")));

    const B = singleFileWeights(await hubFile(repo, "BiCodec/model.safetensors"));
    return new SparkTTS(
      tok,
      lm,
      new BiCodecQuantizer(B),
      new SpeakerDetokenizer(B),
      new BiCodecPrenet(B),
      new WaveGenerator(B),
      new SpeakerTokenizer(B),
    );
  }

  /**
   * The control prompt. Everything the model is told about the voice is in
   * these tokens; the text itself sits between the content markers.
   */
  private promptIds(text: string, gender: Gender, pitch: Level, speed: Level): number[] {
    const g = GENDERS.indexOf(gender),
      p = LEVELS.indexOf(pitch),
      s = LEVELS.indexOf(speed);
    if (g < 0) throw new Error(`gender must be one of ${GENDERS.join(", ")}`);
    if (p < 0 || s < 0) throw new Error(`pitch/speed must be one of ${LEVELS.join(", ")}`);
    return this.tok.encode(
      "<|task_controllable_tts|><|start_content|>" +
        text +
        "<|end_content|>" +
        `<|start_style_label|><|gender_${g}|><|pitch_label_${p}|><|speed_label_${s}|><|end_style_label|>`,
    );
  }

  /** Text -> a mono waveform `[1, samples, 1]` in [-1, 1] at 16 kHz. */
  async generate(text: string, opts: SpeechOptions = {}): Promise<MX> {
    const { gender = "female", pitch = "moderate", speed = "moderate" } = opts;

    const { global, semantic } = await this.run(this.promptIds(text, gender, pitch, speed), opts);

    if (global.length === 0) throw new Error("no global (speaker) tokens generated");
    if (semantic.length === 0) throw new Error("no semantic (content) tokens generated");
    return this.decode(global, semantic);
  }

  /** The generation loop both entry points share. */
  private async run(
    prompt: number[],
    opts: SpeechOptions,
  ): Promise<{ global: number[]; semantic: number[] }> {
    const {
      temp = 0.8,
      topP = 0.95,
      topK = 50,
      repetitionPenalty = 1.3,
      maxTokens = 3000,
      seed,
      cacheLimitMB = 4096,
      onToken,
    } = opts;

    const generated: number[] = [];
    let spoken = 0; // semantic tokens so far, for progress
    const prevLimit = setCacheLimit(cacheLimitMB);
    try {
      for await (const { token } of streamTokens(this.lm, prompt, {
        max: maxTokens,
        temp,
        topP,
        topK,
        repetitionPenalty,
        repetitionContext: 20,
        seed,
      })) {
        generated.push(token);
        if (token >= SEMANTIC_BASE && token < SEMANTIC_BASE + SEMANTIC_COUNT) onToken?.(++spoken);
      }
    } finally {
      setCacheLimit(prevLimit);
    }
    return splitAudioTokens(generated);
  }

  /**
   * The 32 tokens that stand for the speaker in a recording.
   *
   * Exposed on its own because it is also how you *measure* a clone: tokenize
   * the reference and the output and compare.
   */
  async speakerTokens(audioPath: string): Promise<number[]> {
    const pcm = referenceClip(volumeNormalize(await decodeAudio(audioPath)));
    const mel = melSpectrogram(pcm);
    try {
      return this.speakerEnc.tokenize(mel);
    } finally {
      mel.free();
    }
  }

  /**
   * Speak `text` in the voice of a recording.
   *
   * The speaker is *measured* rather than invented: the 32 global tokens go
   * into the prompt, and the LM is left to generate only the content. Six
   * seconds of reference audio is what the encoder looks at; more is ignored,
   * less is tiled.
   *
   * Spark can also take the reference's transcript, which conditions the LM on
   * a worked example and matches the speaker's cadence more closely. That path
   * needs BiCodec's content encoder and a separate wav2vec2 model, neither of
   * which is implemented — so pacing here follows the text, not the reference.
   */
  async clone(text: string, audioPath: string, opts: SpeechOptions = {}): Promise<MX> {
    const global = await this.speakerTokens(audioPath);
    const prompt = [
      TASK_TTS,
      START_CONTENT,
      ...this.tok.encode(text),
      END_CONTENT,
      START_GLOBAL,
      ...global.map((g) => GLOBAL_BASE + g),
      END_GLOBAL,
    ];
    const { semantic } = await this.run(prompt, opts);
    if (semantic.length === 0) throw new Error("no semantic (content) tokens generated");
    return this.decode(global, semantic);
  }

  /** BiCodec's decode path: audio tokens -> waveform. Caller frees the result. */
  decode(global: number[], semantic: number[]): MX {
    return tidy(() => {
      const zq = this.quantizer.detokenize(fromI32(Int32Array.from(semantic), [1, semantic.length]));
      const d = this.speaker.detokenize([global]);
      // The speaker vector enters twice: through the prenet's AdaLayerNorm, and
      // again as a plain bias on its output.
      const x = this.prenet.forward(zq, d).add(d.reshape([1, 1, d.shape[1]]));
      return this.generator.forward(x);
    });
  }
}

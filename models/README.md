# Model

Ganj voice (آوای گنج): a Persian text-to-speech voice for classical poetry and prose. It speaks in "nobody's voice":
a voice that does not match any of the reciters it learned from.

Copyright (c) 2026 Amin Akbari. Licensed under CC BY-NC 4.0 ([LICENSE](LICENSE)). For-profit use needs a paid
commercial licence: see [COMMERCIAL.md](../COMMERCIAL.md).

## Download

The model is not stored in git. Download the folder `ganj-zip-d4h` from the repository's
[Releases](https://github.com/aminkvh/Ganj_voice/releases) page or from
[Hugging Face](https://huggingface.co/aminkvh/Ganj_voice).

| File | What it is |
|---|---|
| `iter-40000-avg-2.pt`, `model.json` | the acoustic model (ZipVoice, about 123 M parameters) |
| `tokens.txt` | the model's phoneme and pause tokens |
| `prompt.wav`, `prompt_phonemes.txt` | the voice prompt and its transcript |
| `g2p/` | the letters-to-phonemes model used for verse |
| `ezafe.joblib`, `vowels.json` | checks used to choose the best of several takes |
| `example_in.jsonl`, `example.tsv`, `example.jsonl` | a worked example |

## Use

Install this repository and a ZipVoice checkout as described in [recipes/README.md](../recipes/README.md#setup), then:

```bash
python tools/reading_list.py ganj-zip-d4h/example_in.jsonl lists --voice ganj-zip-d4h
VOICE_DIR=ganj-zip-d4h recipes/zipvoice/synth_voice.sh lists/list.tsv out lists/text.jsonl
```

The input has one JSON object per line: `{"key": "...", "lines": [...], "kind": "verse" | "prose", "metre": "..."}`.
`lines` holds the hemistichs of a verse or one prose sentence; `metre` (optional) is the poem's metre name. The output
is one 24 kHz WAV file per line in `out/`.

Each line is generated 8 times, and the take that says every word, ezafe and short vowel is kept. This suits
pre-rendering; `DRAWS=1` is much faster.

Hemistichs that are not Persian (Arabic, Azeri Turkish) are left out of the reading list, because the voice reads them
badly. Ganjoor's language tag decides when a unit carries one (`langs`); otherwise a detector of Arabic hemistichs
does, and it reads a hemistich when unsure. `--read-all` keeps them.

## On a phone: `ganj-zip-distill`

The same voice, distilled so that it reads in 8 steps instead of 64, as ONNX files for
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 2.0.1 on a phone's CPU. Download the folder `ganj-zip-distill`
from the same places. It holds the acoustic model at full precision (`text_encoder.onnx`, `fm_decoder.onnx`, 495 MB),
the vocoder with a mel despike in front (`vocos_24khz_despike.onnx`, 54 MB), the tokens, the voice prompt prepared for
sherpa, and `phone.json`: the settings for sherpa and the clean-up after it. Its README gives the steps.

On a computer, `tools/sherpa_read.py` reads a reading list with it the same way. It takes about 0.7 s per second of
speech on a desktop CPU with 4 threads. If 8 steps are too slow on a phone, 6 or 4 steps still sound good with this
model.

It was distilled from `ganj-zip-d4h` in two stages (60,000 and 2,000 iterations, the released model as the teacher;
`recipes/zipvoice/distill_q9h.sh`) and exported by `recipes/zipvoice/export_distill.sh`. On 50 held-out lines read
through sherpa, its highest speaker similarity to any training reciter is 0.47.

## How it was made

- **Acoustic model.** A ZipVoice flow-matching model trained from scratch (no pretrained ZipVoice weights) on 216.1 h:
  152.1 h of Ganjoor recitations, used with Ganjoor's consent, and 64.1 h of ManaTTS prose (CC0). It was then
  fine-tuned on 210.9 h with corrected pronunciations and the reciters' pauses marked in the labels.
- **Pronunciation.** The labels follow the reading rules of the `ganj-reading` package: ezafe, written vowel marks,
  tashdid, sukun, and the poem's metre. Hafez and part of Saadi were proofread word by word by a language model, with
  corrections from a native reader. The rest of the verse was read by the `g2p/` model.
- **Letters to phonemes (`g2p/`).** Homo-GE2PE (MIT), fine-tuned on the proofread Hafez and Saadi lines. It reads
  verse; prose is read by the original Homo-GE2PE. See `g2p/NOTICE.md`.
- **Voice prompt.** 7.5 s from a recitation by امیرحسین عظیمی, which is not in the training data, cleaned and
  disguised (formants, pitch and spectrum shifted).
- **Vocoder.** Vocos `charactr/vocos-mel-24khz` (MIT), unchanged. The phone model uses sherpa-onnx's ONNX export of
  it, with the same mel despike as the desktop recipe put in front.

About 77 % of the training hours are women's voices; 0.26 % are Arabic or Azeri lines.

## Identity

On 50 held-out lines, the voice's highest speaker similarity (ECAPA cosine) to any training reciter is 0.49. The
project's limit for "nobody's voice" is 0.6.

## Limitations

- A take can still drop a short word or an ezafe. Choosing among takes removes most of these.
- Long vowels carry a slight pitch wobble, and س and ش are a little brighter than in human readings.
- Words whose reading depends on the meaning (for example کشت: kesht or kosht) are sometimes read the wrong way.
- Arabic and Azeri lines are read with Persian sounds, so the reading list leaves them out unless asked.
- The phone model reads each line once. It is close to the full recipe but has no choice among takes, so a short word
  or an ezafe is dropped a little more often.

## Intended use

Reading Persian poetry and prose aloud, for example in the Ganj app. Not for imitating real people: do not re-tune the
voice to match a reciter.

## Credits

Ganjoor and its reciters, امیرحسین عظیمی, ManaTTS, ZipVoice, Vocos and Homo-GE2PE: see [CREDITS.md](../CREDITS.md)
and [NOTICE.md](../NOTICE.md).

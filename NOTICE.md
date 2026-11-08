# Notice

Ganj voice (آوای گنج)
Copyright (c) 2026 Amin Akbari

Code: AGPL-3.0 ([LICENSE](LICENSE)). Model: CC BY-NC 4.0 ([models/LICENSE](models/LICENSE)). For-profit use:
[COMMERCIAL.md](COMMERCIAL.md). People and data behind the voice: [CREDITS.md](CREDITS.md).

## Third-party software and models

These keep their own licences. They are installed as dependencies, not copied into this repository, unless a file
says otherwise.

| Component | Use | Licence | Source |
|---|---|---|---|
| ZipVoice (k2-fsa), commit 2f7326f | acoustic model code; the model is trained from scratch, no ZipVoice weights are used | Apache-2.0 | https://github.com/k2-fsa/ZipVoice |
| Vocos, `charactr/vocos-mel-24khz` | vocoder code and weights; the phone model's `vocos_24khz_despike.onnx` is sherpa-onnx's ONNX export of these weights (`vocos_24khz.onnx`) with a mel despike added in front | MIT | https://github.com/gemelo-ai/vocos |
| sherpa-onnx (k2-fsa), 2.0.1 | runs the phone model | Apache-2.0 | https://github.com/k2-fsa/sherpa-onnx |
| Homo-GE2PE, `mehdi-hf/Homo-GE2PE-Persian-HF` (copyright (c) 2025 Elnaz Rahmati) | Persian letters to phonemes; the model download's `g2p/` is a fine-tune of it, with its licence and notice inside | MIT | https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF |
| ByT5, `google/byt5-small` | the base model of Homo-GE2PE | Apache-2.0 | https://huggingface.co/google/byt5-small |
| Persian wav2vec2, `jonatasgrosman/wav2vec2-large-xlsr-53-persian` | CTC forced alignment for corpus cutting, labels and analysis only; not part of the voice | Apache-2.0 | https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian |
| pyworld | voice disguise (WORLD analysis and synthesis) | MIT | https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder |
| WORLD (Masanori Morise) | inside pyworld | modified BSD | https://github.com/mmorise/World |

Data: Ganjoor recitations, used with Ganjoor's consent; ManaTTS, CC0 1.0. See [CREDITS.md](CREDITS.md).

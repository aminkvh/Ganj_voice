# Ganj voice (آوای گنج)

A text-to-speech voice for Persian poetry and prose. The voice is "nobody's voice": a ZipVoice flow-matching model
trained from scratch on Ganjoor recitations (used with Ganjoor's consent) and ManaTTS (CC0), prompted with a disguised
recording, so that it is close to no corpus reciter. The model itself (weights, voice prompt, token list, lexicon) is
described in [models/README.md](models/README.md).

## Use the voice

Download the model folder `ganj-zip-d4h` from [Releases](https://github.com/aminkvh/Ganj_voice/releases) or
[Hugging Face](https://huggingface.co/aminkvh/Ganj_voice), install the packages (below) and set up ZipVoice ([recipes/README.md](recipes/README.md#setup)), then:

```bash
python tools/reading_list.py ganj-zip-d4h/example_in.jsonl lists --voice ganj-zip-d4h
VOICE_DIR=ganj-zip-d4h recipes/zipvoice/synth_voice.sh lists/list.tsv out lists/text.jsonl
```

`out/` then holds one WAV file per line of `example_in.jsonl`. To read your own text, write it in the same format.

## Code

The code is split into packages that other projects can use on their own; they live in one repository and depend on
each other only in this order: reading → corpus → voice → eval.

| Package | Import | What it does | Needs |
|---|---|---|---|
| [ganj-reading](packages/ganj-reading) | `ganj_reading` | Persian poetry reading rules: letters to phonemes (Homo-GE2PE) with ezafe, harakat, tashdid, sukun, tanwin and small-alef rules and a word lexicon; aruz metre scansion; the token set (style, metre and pause tokens) | – |
| [ganj-corpus](packages/ganj-corpus) | `ganj_corpus` | Training data from Ganjoor and ManaTTS: selection, download, cutting at aligned pauses, cleaning, labels, pause labels, the rules check every label must pass | ganj-reading |
| [ganj-voice](packages/ganj-voice) | `ganj_voice` | Voice disguise and synthesis helpers: prompt disguise (WORLD, Praat, mel domain), mains-hum and noise removal, mel despike before the vocoder, guidance interval, finer sampler finish, regeneration from part-way along the flow, pause trimming, Vocos fine-tune data | ganj-reading |
| [ganj-eval](packages/ganj-eval) | `ganj_eval` | Evaluation: Whisper WER, UTMOS, distance to every corpus reciter's voice, prosody against a reference reading, CTC label checks, noise measures, a listening page | reading, corpus, voice |

Beside the packages:

- [recipes/](recipes) – the scripts that build the corpus, train ZipVoice, synthesise and score the voice
  (`recipes/zipvoice/`), and fine-tune Vocos (`recipes/vocos/`). See [recipes/README.md](recipes/README.md).
- [tools/](tools) – command-line tools: voice prompts (`make_prompt.py`, `melwarp_prompt.py`), post-processing
  (`postclean.py`, `trim_pauses.py`), choosing among noise draws (`pick_draws.py`), checks before synthesis or training
  (`check_rules.py`, `g2p_gate.py`), pause labels (`pause_labels.py`), metre and lexicon tools (`freeze_metres.py`,
  `scansion_report.py`, `validate_lexicon.py`), corpus audits (`boundary_audit.py`, `ina_segment.py`), measurements
  (`noise_probe.py`, `prosody_compare.py`) and a licence check of the installed packages (`check_licenses.py`). Each
  file's docstring says how to run it.

## Install

Python 3.10 or newer (developed on 3.11). From the repository root:

```bash
pip install -e "packages/ganj-reading[g2p]" -e packages/ganj-corpus -e packages/ganj-voice -e "packages/ganj-eval[models]"
```

A package can also be installed alone, with the ones it needs, e.g. only the reading rules:

```bash
pip install -e packages/ganj-reading          # tokens and scansion; add [g2p] for the letters-to-phonemes model
```

```python
from ganj_reading.scansion import scans

scans("t/vana bov/d h/r ke dana bov/d", "فعولن فعولن فعولن فعل")   # True: the line scans in the metre
```

The training and synthesis recipes also need a ZipVoice checkout and its environment; see
[recipes/README.md](recipes/README.md).

## Environment

| Variable | Used by | Meaning |
|---|---|---|
| `GANJ_DATA` | tools, recipes | the data folder: corpus, eval set, model runs, voice packages (not in git) |
| `ZIPVOICE_DIR` | recipes, `tools/pause_labels.py` | a ZipVoice checkout (https://github.com/k2-fsa/ZipVoice) |
| `GANJ_PY`, `ZIPVOICE_PY`, `INA_PY` | recipes | the command that runs Python in each environment (defaults `python`, `python3`, `python`) |

A tool or recipe that needs one of these stops with a message naming it when it is not set.

## Tests

```bash
pip install pytest
pytest -q            # from the repository root: every package, the tools and the recipes; no GPU, data or network
```

Each package's tests also run on their own (`cd packages/ganj-reading && pytest -q`). Tests that need `GANJ_DATA` and
downloaded audio are marked `integration` and skip without it.

## Licence

Code: GNU AGPL-3.0-or-later ([LICENSE](LICENSE)). Model: CC BY-NC 4.0 ([models/LICENSE](models/LICENSE)). For-profit
use needs a commercial licence: [COMMERCIAL.md](COMMERCIAL.md). Third-party parts: [NOTICE.md](NOTICE.md). The people
and data behind the voice: [CREDITS.md](CREDITS.md). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

Copyright (c) 2026 Amin Akbari.

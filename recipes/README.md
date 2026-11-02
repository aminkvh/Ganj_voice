# Recipes

The scripts that made the released voice: corpus, ZipVoice training, synthesis, scoring, and the Vocos fine-tune. They
are bash; run them from anywhere (each finds the repository itself). Training and synthesis need an NVIDIA GPU.

## Setup

Three Python environments (one is enough if everything installs together):

- the **ganj environment**: the four packages of this repository (`pip install -e packages/...`, see the main README);
- the **ZipVoice environment**: a checkout of https://github.com/k2-fsa/ZipVoice (commit 2f7326f) and its
  requirements; `recipes/vocos/` also needs `vocos` there;
- the **segmentation environment**: inaSpeechSegmenter (only `build_corpus.sh`).

```bash
export GANJ_DATA=/path/to/ganj-voice-data        # corpus, eval set, model runs (paths in voice_d4h.json are relative to it)
export ZIPVOICE_DIR=/path/to/ZipVoice
# the command that runs Python in each environment, e.g. with conda environments named ganj-voice, ganj-zip, ganj-ina:
export GANJ_PY="conda run --no-capture-output -n ganj-voice python"
export ZIPVOICE_PY="conda run --no-capture-output -n ganj-zip python3"
export INA_PY="conda run --no-capture-output -n ganj-ina python"
```

Without them the scripts use `python` / `python3` from the PATH. A script stops with a message when `GANJ_DATA` or
`ZIPVOICE_DIR` is needed and not set. ZipVoice's data for this voice lives in `$ZIPVOICE_DIR/egs/zipvoice/data_q`.

## The voice, step by step

1. **Corpus.** `python -m ganj_corpus.select` picks recitations from a Ganjoor census, `python -m ganj_corpus.download`
   fetches them into `$GANJ_DATA`, and `zipvoice/build_corpus.sh` builds `$GANJ_DATA/corpus_q` (speech/music
   segmentation, the metre-token list, cutting at aligned pauses, cleaning, labels, audits). The fixed eval set comes
   from `python -m ganj_eval.set_builder`; ManaTTS prose from `python -m ganj_corpus.manatts`.
2. **Data for ZipVoice.** `zipvoice/prep_q.sh`: TSVs (`make_tsv.py`), lhotse manifests, fbank features, the token list
   and the eval test list (`make_test_q.py`).
3. **Training from scratch.** `zipvoice/run_q.sh` (120k iterations), then `zipvoice/resume_q.sh` (to 200k).
4. **Clean-subset fine-tune.** Relabel with the current reading rules,
   `python -m ganj_corpus.relabel --data $GANJ_DATA --cuts-dir $ZIPVOICE_DIR/egs/zipvoice/data_q/fbank --subset clean --tag _q9c`
   (the clean subset is `ganj_corpus.clean_subset`), then `LBL=_q9c zipvoice/finetune_clean_lbl.sh`.
5. **ManaTTS fine-tune.** `zipvoice/finetune_mana.sh`: the clean poetry plus ManaTTS prose, shuffled together.
   `zipvoice/finish_ft.sh` finishes a fine-tune whose wrapper stopped after training.
6. **Pause-token fine-tune (the released model).** Relabel with the current reading rules, verse read by the
   letters-to-phonemes model of the download (`g2p/`) and reviewed per-line readings (not published):
   `python -m ganj_corpus.relabel --data $GANJ_DATA --cuts-dir $ZIPVOICE_DIR/egs/zipvoice/data_q/fbank --subset clean --tag _q9g --verse-model $GANJ_DATA/models/ganj-zip-d4h/g2p --overrides READINGS.json`.
   Then `zipvoice/finetune_q9p.sh`: the reciters' measured pauses marked in the labels (`tools/pause_labels.py`), two
   pause tokens added to the model, 40k iterations from step 5's model. The script reads the `_q9e` labels and writes
   `_q9p`; the released model is the same recipe on the `_q9g` labels (`pause_labels.py --tag-in _q9g --tag _q9h`).
   Its `iter-40000-avg-2.pt` is the released model.
7. **Voice prompt.** `tools/make_prompt.py` disguises a held-out recording; the recipe of the released prompt and voice
   is in `zipvoice/voice_d4h.json`.
8. **Synthesis.** Download the model folder `ganj-zip-d4h` (see [models/README.md](../models/README.md)) into
   `$GANJ_DATA/models/`, or point `VOICE_DIR` at it. `tools/reading_list.py IN.jsonl OUT_DIR --voice VOICE_DIR` makes a
   reading list and its text file from plain text (one unit per line: key, lines, kind verse or prose, optional metre):
   the reading rules, the voice's `g2p/` model for verse, reviewed per-line readings (`--overrides`), the reading-time
   pauses. `zipvoice/synth_voice.sh LIST.tsv OUT_DIR [TEXT.jsonl]` reads the list with that voice, e.g.
   `tools/reading_list.py $GANJ_DATA/models/ganj-zip-d4h/example_in.jsonl lists --voice $GANJ_DATA/models/ganj-zip-d4h`,
   then `zipvoice/synth_voice.sh lists/list.tsv out lists/text.jsonl`. Every list passes `tools/check_rules.py` first;
   the model reads a «و» after a consonant-final word glued onto it (`ganj_voice.glue`); the sampler guides only up to
   flow time 0.6 and finishes from 0.8 in finer steps; the mel is despiked before Vocos (`infer_despike.py`) and the
   output high-passed at 70 Hz (`tools/postclean.py`). With TEXT (the lines and labels), each line is drawn several
   times and `tools/pick_draws.py` keeps the draw that says every word, ezafe and short vowel of its label
   (`vowels.json` of the voice; `ganj_eval.vowelcheck`), the hiss of س and ش is turned down 6 dB (`ganj_voice.deess`;
   `DEESS=0` switches it off), then `tools/trim_pauses.py` shortens long stops inside a hemistich.
   `zipvoice/draws.sh` makes N draws of chosen lines for inspection.
9. **Scoring.** `python -m ganj_eval.run --data $GANJ_DATA --tag TAG --wav-dir DIR --set set_q.jsonl`;
   `zipvoice/sweep_q.sh`, `zipvoice/sweep_despike_x.sh` and `zipvoice/sweep_tail.sh` synthesise and score settings.

Diagnostics, not part of the released voice:

- **Where the vocoder's tones form.** `zipvoice/regen_partial.sh` and `zipvoice/regen_steps.sh` regenerate real
  recordings from part-way along the flow.
- **Vocos fine-tune.** `zipvoice/regen_pairs.sh` builds pairs from the model's own output, `vocos/finetune.sh` trains.

The suffixes `_q9c`, `_q9e`, `_q9g`, `_q9p`, `_q9h` in file names are label-set versions: each relabel writes a new set beside the old.

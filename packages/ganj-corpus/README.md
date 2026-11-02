# ganj-corpus

Training data for Ganj voice (آوای گنج) from Ganjoor recitations and ManaTTS: the Ganjoor API client, selection and
download of recitations, clip cutting at aligned pauses (CTC forced alignment), cleaning (loudness, bandwidth,
clipping, music), labels from `ganj-reading`, relabelling, pause labels, ZipVoice manifest helpers, and the rules
check every label must pass before it is trained on or synthesised.

```bash
pip install -e ../ganj-reading -e .
python -m ganj_corpus.build --help
```

Part of https://github.com/aminkvh/Ganj_voice. Licence: AGPL-3.0-or-later.

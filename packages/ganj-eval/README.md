# ganj-eval

Evaluation of Ganj voice (آوای گنج): Whisper WER on Persian (with a Persian-only summary), UTMOSv2, speaker-identity
distance of the output to every corpus reciter (ECAPA), tiny-word and line-end recall, prosody against a reference
reading, CTC label checks (does a line say its label?), background and voice noise, and an HTML listening page.

```bash
pip install -e ../ganj-reading -e ../ganj-corpus -e ../ganj-voice -e ".[models]"
python -m ganj_eval.run --help
```

Part of https://github.com/aminkvh/Ganj_voice. Licence: AGPL-3.0-or-later.

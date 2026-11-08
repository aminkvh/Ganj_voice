# ganj-reading

Persian poetry reading rules for text-to-speech, as used by Ganj voice (آوای گنج):

- `ganj_reading.g2p`: letters to phonemes with the Homo-GE2PE model (MIT) and the rules it misses: the ezafe (written
  or not), written vowel marks (harakat) as constraints, tashdid (doubling), sukun, tanwin, the small alef, ة, a word
  lexicon (`lexicon.json`), and, for verse, the n-best reading that scans in the poem's metre.
- `ganj_reading.scansion`: aruz scansion of phonemes against a metre name.
- `ganj_reading.tokens`: the token set: style tokens (verse, prose, hemistich), one token per metre (`metres.json`).
- `ganj_reading.language`: which hemistichs to read: Ganjoor's language tag when there is one, else a letter n-gram
  detector of Arabic hemistichs (`arabic_ngrams.json`) that reads a hemistich when unsure.

```bash
pip install -e .            # tokens and scansion
pip install -e ".[g2p]"     # plus the G2P model (torch, transformers)
```

Part of https://github.com/aminkvh/Ganj_voice. Licence: AGPL-3.0-or-later.

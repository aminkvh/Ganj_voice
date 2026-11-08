# ganj-voice

Voice disguise and synthesis helpers of Ganj voice (آوای گنج):

- prompt disguise: WORLD formant and pitch shift, Praat, mel-domain warp, LTAS match (`anonymise`, `melwarp`);
- clean-up: mains-hum notch, stationary denoise, high-pass (`dehum`), pause trimming (`pausetrim`);
- ZipVoice inference helpers: mel despike before the vocoder (`melsmooth`), guidance on part of the flow (`guidance`),
  a finer sampler finish (`schedule`), regeneration from part-way along the flow (`regen`), chorus sampling (`chorus`);
- Vocos fine-tune data (`vocos_data`), checkpoint blending and ONNX export (`blend`, `export`, `synth`);
- the phone voice: distillation settings (`distill`), the mel despike put in front of an ONNX vocoder
  (`despike_onnx`), the prompt cut so that sherpa-onnx reads ZipVoice's prompt frames (`sherpa_prompt`), and the
  clean-up after sherpa (`phonepost`, with pause trimming from levels alone in `levelstops`).

The ZipVoice helpers patch ZipVoice's own classes and need a ZipVoice checkout on the path.

```bash
pip install -e ../ganj-reading -e .
```

Part of https://github.com/aminkvh/Ganj_voice. Licence: AGPL-3.0-or-later.

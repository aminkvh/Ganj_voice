# Contributing

Thank you for helping. Issues and pull requests are welcome: pronunciation errors (a word, the line it is in and how
it should be read), bugs, and improvements to any package.

## Before your first pull request

Sign the [Contributor Licence Agreement](CLA.md): add a comment to your pull request saying
"I have read the CLA and I agree to it." The project is offered under the AGPL-3.0 and, for for-profit users, under a
paid commercial licence ([COMMERCIAL.md](COMMERCIAL.md)); the CLA is what lets your contribution be offered both ways.
You keep the copyright in your work.

## Pull requests

- One change per pull request, with tests (`pytest -q`); the existing tests must pass.
- Reading rules: a change to how a word or mark is read needs the line it comes from and a source or a reciter's
  reading. A word with more than one reading is fixed per line, never everywhere.
- No audio, model weights or other large files in git.
- Keep the commit message short: one line saying what changed.

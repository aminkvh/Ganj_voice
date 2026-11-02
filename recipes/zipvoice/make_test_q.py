# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""ZipVoice test list for eval/set_q.jsonl: each eval item's current phonemes with the prompt of the existing
test list. The prompt is a placeholder, to be replaced by the voice prompt.
Usage: python -m recipes.zipvoice.make_test_q SET_Q.jsonl OLD_TEST.tsv OUT.tsv"""
import json
import sys


def read_prompt(path):
    first = open(path, encoding="utf-8").readline().rstrip("\n").split("\t")
    return first[1], first[2]


def test_line(item: dict, prompt) -> str:
    return f"{item['key']}\t{prompt[0]}\t{prompt[1]}\t{item['phonemes']}"


def main(set_path, old_test, out_path):
    prompt = read_prompt(old_test)
    items = [json.loads(l) for l in open(set_path, encoding="utf-8")]
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(test_line(it, prompt) + "\n")
    print(f"wrote {len(items)} items, prompt {prompt[1]}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

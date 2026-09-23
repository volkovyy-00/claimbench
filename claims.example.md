---
# This is an illustrative claims file, not a real memo. It pairs with
# memos.yaml.example. `extract` writes files like this into claims/<memo_id>.md;
# you can also write one entirely by hand and skip `extract`.
memo_id: MEMO-001
source_folder: sources/acme
# Needed by `tag_pipeline.py draft` only: the company the memo is about.
filing_entity: Acme plc
# Optional per-memo tuning — same three keys and ranges as memos.yaml
# (design decision 12). Omit any you don't need.
relative_threshold: 0.45
---

<!-- written by 'extract'; edit freely, then run 'build' -->

## Business Profile

1. Acme is the largest listed widget maker in Europe.
2. Acme's plants were valued at EUR 20bn at 31 December 2024.

<!-- a single-line comment like this is the only silent way to leave a note, anywhere in the file; bare prose with no comment markers still won't become a claim, but now warns -- see docs/pipeline-overview.md, "How to write a claims file" -->

## Ownership

1. Acme is listed in London and in Frankfurt.

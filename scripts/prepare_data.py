"""Prepare PixelLM JSONL training data."""

from __future__ import annotations

import argparse

from pixellm.data.prepare import iter_nouns_records, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PixelLM SFT JSONL data.")
    parser.add_argument("--output", default="data/processed/train_v1.jsonl")
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()

    write_jsonl(iter_nouns_records(limit=args.limit), args.output)


if __name__ == "__main__":
    main()

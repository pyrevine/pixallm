# pixellm

Teaching small LLMs to draw 16x16 pixel art through code generation.

`pixellm` trains an open-source language model to emit a compact Palette Index Grid DSL. The generated text can be parsed, validated, rendered to a PNG, and scored with deterministic rewards before moving into GRPO.

## Status

Week 1 foundation is in place.

- Palette Index Grid parser, serializer, and validator
- PIL renderer for DSL outputs
- Nouns dataset preparation into `{caption, prompt, dsl, source, source_id, category, view, license}` JSONL
- Border-background removal for transparent sprite training data
- Deterministic metrics: parse rate, palette constraint, non-empty, symmetry, connected component
- 100 fixed evaluation prompts
- Preview contact sheets for dataset inspection

## DSL

```text
<PALETTE>
0:transparent, 1:#f6d9c5, 2:#d59d85, 3:#1b0809
</PALETTE>
<GRID>
0000000000000000
0000000330000000
0000013332200000
...
</GRID>
```

Rules:

- Grid size is fixed at 16x16.
- Palette indices are `0-7`.
- `0` is always `transparent`.
- Non-transparent colors use `1-7`.
- The training target is the tag DSL string, not JSON.

## Quick Start

```bash
uv sync
uv run pytest
uv run python scripts/prepare_data.py --limit 3000 --output data/processed/train_v1.jsonl
uv run python scripts/preview_samples.py --input data/processed/train_v1.jsonl --output docs/results/gallery/train_v1_sample_50.png --limit 50 --sample --seed 42
```

## SFT

The first SFT target is Qwen2.5-Coder-3B-Instruct with QLoRA.

```bash
uv run python -m pixellm.train.sft \
  --train-file data/processed/train_v1.jsonl \
  --output-dir checkpoints/sft-v1 \
  --report-to none
```

For RunPod training, pass `--report-to wandb` after installing and logging into Weights & Biases.

## Project Layout

```text
src/pixellm/dsl.py            # Palette Index Grid parser/validator
src/pixellm/render.py         # DSL -> PIL image
src/pixellm/data/prepare.py   # dataset/image conversion
src/pixellm/eval/metrics.py   # deterministic metrics
src/pixellm/train/sft.py      # QLoRA SFT entrypoint
scripts/prepare_data.py       # JSONL generation
scripts/preview_samples.py    # contact sheet preview
data/eval_prompts.json        # fixed eval prompts
```

## Current Data Snapshot

`m1guelpf/nouns` is used as the SFT v1 dataset. The generated `train_v1.jsonl` currently contains 3,000 front-view character samples with transparent backgrounds.

Preview: `docs/results/gallery/train_v1_sample_50.png`

"""QLoRA SFT entrypoint for pixallm."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_TRAIN_FILE = "data/processed/train_v1.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/sft-v1"
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass(frozen=True)
class SFTPaths:
    train_file: Path
    output_dir: Path


@dataclass(frozen=True)
class PrecisionConfig:
    use_cpu: bool
    bf16: bool
    fp16: bool
    torch_dtype: torch.dtype
    compute_dtype: torch.dtype


def main() -> None:
    args = parse_args()
    trainer = build_trainer(args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pixallm SFT v1 with QLoRA.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--report-to", default="none", help="Use 'wandb' on RunPod after wandb login.")
    parser.add_argument("--run-name", default="pixallm-sft-v1")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA loading.")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 training.")
    return parser.parse_args()


def build_trainer(args: argparse.Namespace) -> SFTTrainer:
    dataset = load_sft_dataset(args.train_file, eval_split=args.eval_split, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    ensure_tokenizer_padding(tokenizer)

    trainer_kwargs: dict[str, Any] = {
        "model": args.model_name,
        "args": build_sft_config(args),
        "train_dataset": dataset["train"] if isinstance(dataset, dict) else dataset,
        "processing_class": tokenizer,
        "peft_config": build_lora_config(args),
    }
    if isinstance(dataset, dict) and "eval" in dataset:
        trainer_kwargs["eval_dataset"] = dataset["eval"]

    return SFTTrainer(**trainer_kwargs)


def load_sft_dataset(
    train_file: str | Path, *, eval_split: float = 0.0, seed: int = 42
) -> Dataset | dict[str, Dataset]:
    path = Path(train_file)
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.map(record_to_messages, remove_columns=dataset.column_names)

    if eval_split <= 0:
        return dataset
    if not 0 < eval_split < 1:
        raise ValueError("eval_split must be 0 or a fraction between 0 and 1.")

    split = dataset.train_test_split(test_size=eval_split, seed=seed)
    return {"train": split["train"], "eval": split["test"]}


def record_to_messages(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert one prepared JSONL row into Qwen chat messages."""

    return {
        "messages": [
            {"role": "user", "content": str(record["prompt"])},
            {"role": "assistant", "content": str(record["dsl"])},
        ]
    }


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    precision = resolve_precision(args)
    model_init_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": precision.torch_dtype,
    }
    if not args.no_4bit:
        if precision.use_cpu:
            raise ValueError("4-bit QLoRA requires CUDA. Use --no-4bit when running on CPU.")
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=precision.compute_dtype,
        )
        model_init_kwargs["device_map"] = "auto"

    report_to = [] if args.report_to == "none" else [args.report_to]
    return SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_cpu=precision.use_cpu,
        bf16=precision.bf16,
        fp16=precision.fp16,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        packing=False,
        shuffle_dataset=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=report_to,
        run_name=args.run_name,
        seed=args.seed,
        model_init_kwargs=model_init_kwargs,
    )


def resolve_precision(args: argparse.Namespace) -> PrecisionConfig:
    """Select a precision mode that the current runtime actually supports."""

    if not torch.cuda.is_available():
        return PrecisionConfig(
            use_cpu=True,
            bf16=False,
            fp16=False,
            torch_dtype=torch.float32,
            compute_dtype=torch.float32,
        )

    if not args.no_bf16 and torch.cuda.is_bf16_supported():
        return PrecisionConfig(
            use_cpu=False,
            bf16=True,
            fp16=False,
            torch_dtype=torch.bfloat16,
            compute_dtype=torch.bfloat16,
        )

    return PrecisionConfig(
        use_cpu=False,
        bf16=False,
        fp16=True,
        torch_dtype=torch.float16,
        compute_dtype=torch.float16,
    )


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=parse_target_modules(args.target_modules),
    )


def parse_target_modules(value: str) -> list[str]:
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("target_modules must contain at least one module name.")
    return modules


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


if __name__ == "__main__":
    main()

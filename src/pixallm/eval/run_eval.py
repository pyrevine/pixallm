"""Evaluate a pixallm checkpoint on the fixed eval prompts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pixallm.data.prompts import build_prompt
from pixallm.dsl import DSLParseError, parse_dsl
from pixallm.eval.metrics import (
    connected_component_score,
    non_empty_score,
    palette_constraint_score,
    symmetry_score,
)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_PROMPTS_FILE = "data/eval_prompts.json"
DEFAULT_OUTPUT_FILE = "docs/results/eval_sft_v1.json"
METRIC_KEYS: tuple[str, ...] = (
    "palette_constraint",
    "non_empty",
    "symmetry",
    "connected_component",
)


@dataclass(frozen=True)
class EvalConfig:
    base_model: str
    adapter_path: str | None
    prompts_file: Path
    output_file: Path
    run_name: str
    num_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    load_in_4bit: bool

    def to_serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prompts_file"] = str(self.prompts_file)
        payload["output_file"] = str(self.output_file)
        return payload


def main() -> None:
    config = parse_args()
    config.output_file.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(config.prompts_file)
    tokenizer = load_tokenizer(config.base_model)
    model = load_model(config)

    per_prompt = generate_and_score(model, tokenizer, prompts, config)
    aggregate = summarize(per_prompt)

    payload = {
        "config": config.to_serializable(),
        "aggregate": aggregate,
        "per_prompt": per_prompt,
    }
    config.output_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_summary(aggregate, config.output_file)


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate a pixallm checkpoint.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="LoRA adapter directory. Omit to evaluate the base model directly.",
    )
    parser.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--run-name", default="sft-v1")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading.")
    args = parser.parse_args()

    return EvalConfig(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        prompts_file=Path(args.prompts_file),
        output_file=Path(args.output_file),
        run_name=args.run_name,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        load_in_4bit=not args.no_4bit,
    )


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array of caption strings.")
    return [str(item) for item in data]


def load_tokenizer(base_model: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(config: EvalConfig) -> Any:
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16

    kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if cuda:
        kwargs["device_map"] = "auto"
    if config.load_in_4bit:
        if not cuda:
            raise ValueError("4-bit loading requires CUDA. Pass --no-4bit to run on CPU.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path)
    model.eval()
    return model


def generate_and_score(
    model: Any, tokenizer: Any, prompts: list[str], config: EvalConfig
) -> list[dict[str, Any]]:
    torch.manual_seed(config.seed)
    device = next(model.parameters()).device

    per_prompt: list[dict[str, Any]] = []
    for idx, caption in enumerate(prompts):
        user_text = build_prompt(caption)
        messages = [{"role": "user", "content": user_text}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                num_return_sequences=config.num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[:, input_ids.shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        samples = [score_sample(text) for text in texts]
        per_prompt.append({"idx": idx, "caption": caption, "samples": samples})

        if (idx + 1) % 10 == 0 or idx + 1 == len(prompts):
            print(f"[{idx + 1}/{len(prompts)}] generated", flush=True)

    return per_prompt


def score_sample(text: str) -> dict[str, Any]:
    try:
        pixel_art = parse_dsl(text)
    except DSLParseError:
        return {"text": text, "parsed": False}

    return {
        "text": text,
        "parsed": True,
        "palette_constraint": palette_constraint_score(pixel_art),
        "non_empty": non_empty_score(pixel_art),
        "symmetry": symmetry_score(pixel_art),
        "connected_component": connected_component_score(pixel_art),
    }


def summarize(per_prompt: list[dict[str, Any]]) -> dict[str, Any]:
    all_samples = [sample for item in per_prompt for sample in item["samples"]]
    parsed_samples = [sample for sample in all_samples if sample["parsed"]]
    num_total = len(all_samples)

    aggregate: dict[str, Any] = {
        "num_prompts": len(per_prompt),
        "num_samples_per_prompt": len(per_prompt[0]["samples"]) if per_prompt else 0,
        "num_samples_total": num_total,
        "parse_rate": len(parsed_samples) / num_total if num_total else 0.0,
    }
    for key in METRIC_KEYS:
        values = [sample[key] for sample in parsed_samples]
        aggregate[f"{key}_mean_on_parsed"] = mean(values) if values else 0.0
    return aggregate


def print_summary(aggregate: dict[str, Any], output_file: Path) -> None:
    print(f"\n=== eval summary ({output_file}) ===")
    for key, value in aggregate.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

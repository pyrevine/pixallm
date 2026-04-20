from argparse import Namespace

from pixallm.train.sft import build_lora_config, build_sft_config, parse_target_modules, record_to_messages


def test_record_to_messages_uses_prompt_and_dsl() -> None:
    record = {"prompt": "Draw 16x16 pixel art as Palette Index Grid DSL: a cat", "dsl": "<PALETTE />"}

    result = record_to_messages(record)

    assert result == {
        "messages": [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["dsl"]},
        ]
    }


def test_parse_target_modules() -> None:
    assert parse_target_modules("q_proj, k_proj ,,v_proj") == ["q_proj", "k_proj", "v_proj"]


def test_sft_config_defaults_match_plan() -> None:
    args = Namespace(
        output_dir="checkpoints/sft-v1",
        max_length=1024,
        num_train_epochs=3.0,
        max_steps=-1,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        no_bf16=False,
        no_4bit=True,
        logging_steps=10,
        save_steps=200,
        report_to="none",
        run_name="pixallm-sft-v1",
        seed=42,
    )

    config = build_sft_config(args)

    assert config.learning_rate == 2e-4
    assert config.per_device_train_batch_size == 2
    assert config.gradient_accumulation_steps == 8
    assert config.assistant_only_loss is True
    assert config.max_length == 1024


def test_lora_config_defaults_match_plan() -> None:
    args = Namespace(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="q_proj,k_proj",
    )

    config = build_lora_config(args)

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.target_modules == {"q_proj", "k_proj"}

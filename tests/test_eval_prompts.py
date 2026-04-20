import json
from pathlib import Path


def test_eval_prompt_file_has_100_unique_front_view_prompts() -> None:
    prompts = json.loads(Path("data/eval_prompts.json").read_text(encoding="utf-8"))

    assert len(prompts) == 100
    assert len(set(prompts)) == 100
    assert all("front view pixel art" in prompt for prompt in prompts)

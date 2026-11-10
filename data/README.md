# Pirate Dolly dataset

Question → pirate-dialect answer pairs for fine-tuning a chat model (workshop Notebook 1).

- **Files:** `pirate_dolly_train.jsonl` (1500 rows), `pirate_dolly_eval.jsonl` (50 rows).
  Each row is `{"messages": [user, assistant]}` with no system prompt.
- **Questions:** `databricks/databricks-dolly-15k`, context-free prompts in open_qa, general_qa, brainstorming, creative_writing, classification, 10-200 characters.
- **Answers:** `Qwen/Qwen3.6-27B-FP8` with thinking off, `temperature=0.7`, `top_p=0.9`,
  `max_tokens=300`, seed 0, under a pirate system prompt (see `pirate_dolly_meta.json`).
- **Filter:** kept 1780/1782 replies: finished normally, no `<think>`, over 20 characters,
  and matching `\b(arr+|ahoy|matey|ye|yer|aye|hearties)\b`.
- **Generated:** 2026-11-09 by `scripts/make_pirate_dataset.py`.

## Licence

Derived from databricks-dolly-15k (© Databricks, CC BY-SA 3.0), so this dataset is released under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Answers were generated with Qwen/Qwen3.6-27B-FP8. TODO: confirm its licence terms on the model card and note them here.

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["vllm", "datasets"]
# ///
"""Build the pirate SFT dataset for Notebook 1.

Takes prompts from databricks/databricks-dolly-15k and has a teacher model
(default Qwen/Qwen3.6-27B-FP8, thinking off) answer them in pirate dialect.
Adapted from UoA-eResearch/ahoy_training scripts/make_dataset.py.

Rows are {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]},
with NO system prompt, so the student learns to be a pirate unconditionally.

Outputs (in --out, default <repo>/data):
    pirate_dolly_train.jsonl, pirate_dolly_eval.jsonl,
    pirate_dolly_meta.json (generation settings), README.md

Run on the GPU machine, either:
    # offline: loads the model in this process. `uv run` installs vllm + datasets
    # from the header above into a throwaway environment.
    uv run make_pirate_dataset.py

    # against a running server:
    #   vllm serve Qwen/Qwen3.6-27B-FP8 --max-model-len 4096 --max-num-seqs 256
    #     --limit-mm-per-prompt '{"image": 0, "video": 0}'
    # (needs `pip install openai datasets`)
    python scripts/make_pirate_dataset.py --base-url http://localhost:8000/v1

Try a quick run first and read the output:  --n 20 --eval-n 5
"""
import argparse
import json
import random
import re
from datetime import date
from pathlib import Path

SYSTEM = (
    "You are Captain Redbeard, a salty old pirate. Answer the user's question helpfully "
    "and correctly, but speak ENTIRELY in exaggerated pirate dialect: use 'Arr', 'Ahoy', "
    "'matey', 'ye', 'yer', 'aye', 'me hearties', 'shiver me timbers', nautical metaphors, "
    "and pirate slang in every sentence. Keep answers concise (2-5 sentences). Never break character."
)
CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing", "classification")
PIRATE = re.compile(r"\b(arr+|ahoy|matey|ye|yer|aye|hearties)\b", re.I)
NO_THINK = {"enable_thinking": False}


def select_prompts(k, seed):
    """Context-free, short-ish Dolly prompts, deduplicated and shuffled."""
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    seen, prompts = set(), []
    for r in ds:
        p = r["instruction"].strip()
        if r["context"] or r["category"] not in CATEGORIES or not 10 < len(p) < 200:
            continue
        if p.lower() not in seen:
            seen.add(p.lower())
            prompts.append(p)
    random.Random(seed).shuffle(prompts)
    return prompts[:k]


def conversation(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def generate_offline(a, prompts):
    """Returns [(reply, finish_reason)] using vLLM in this process."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization, seed=a.seed,
              # Qwen3.6's hybrid (Mamba-style) layers need one cache block per
              # sequence; vLLM's default of 1024 can exceed what's available.
              max_num_seqs=a.max_num_seqs,
              # text only: don't reserve memory for the vision encoder
              limit_mm_per_prompt={"image": 0, "video": 0})
    params = SamplingParams(temperature=a.temperature, top_p=a.top_p,
                            max_tokens=a.max_tokens, seed=a.seed)
    outs = llm.chat([conversation(p) for p in prompts], params,
                    chat_template_kwargs=NO_THINK, use_tqdm=True)
    return [(o.outputs[0].text.strip(), o.outputs[0].finish_reason) for o in outs]


def generate_server(a, prompts):
    """Returns [(reply, finish_reason)] from an OpenAI-compatible vLLM server."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    client = OpenAI(base_url=a.base_url, api_key=a.api_key)

    def one(item):
        i, p = item
        r = client.chat.completions.create(
            model=a.model, messages=conversation(p), temperature=a.temperature,
            top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed + i,
            extra_body={"chat_template_kwargs": NO_THINK},  # must be nested, not top-level
        )
        c = r.choices[0]
        return (c.message.content or "").strip(), c.finish_reason

    results = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for res in ex.map(one, enumerate(prompts)):
            results.append(res)
            if len(results) % 100 == 0 or len(results) == len(prompts):
                print(f"{len(results)}/{len(prompts)}", flush=True)
    return results


def keep(reply, finish_reason):
    """Same filter as ahoy, plus: no thinking leaked, not cut off."""
    return (finish_reason == "stop" and "<think>" not in reply
            and len(reply) > 20 and PIRATE.search(reply) is not None)


def write_jsonl(path, pairs):
    with open(path, "w") as f:
        for p, r in pairs:
            row = {"messages": [{"role": "user", "content": p},
                                {"role": "assistant", "content": r}]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


README = """# Pirate Dolly dataset

Question → pirate-dialect answer pairs for fine-tuning a chat model (workshop Notebook 1).

- **Files:** `pirate_dolly_train.jsonl` ({n_train} rows), `pirate_dolly_eval.jsonl` ({n_eval} rows).
  Each row is `{{"messages": [user, assistant]}}` with no system prompt.
- **Questions:** `databricks/databricks-dolly-15k`, context-free prompts in {categories}, 10-200 characters.
- **Answers:** `{model}` with thinking off, `temperature={temperature}`, `top_p={top_p}`,
  `max_tokens={max_tokens}`, seed {seed}, under a pirate system prompt (see `pirate_dolly_meta.json`).
- **Filter:** kept {n_kept}/{n_generated} replies: finished normally, no `<think>`, over 20 characters,
  and matching `{regex}`.
- **Generated:** {date} by `scripts/make_pirate_dataset.py`.

## Licence

Derived from databricks-dolly-15k (© Databricks, CC BY-SA 3.0), so this dataset is released under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Answers were generated with {model}. TODO: confirm its licence terms on the model card and note them here.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    ap.add_argument("--base-url", help="OpenAI-compatible server URL; omit to run vLLM offline")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--concurrency", type=int, default=64, help="server mode only")
    ap.add_argument("--n", type=int, default=1500, help="training pairs to keep")
    ap.add_argument("--eval-n", type=int, default=50, help="held-out pairs to keep")
    ap.add_argument("--oversample", type=float, default=1.15,
                    help="extra prompts to generate, to cover replies dropped by the filter")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--max-model-len", type=int, default=4096, help="offline mode only")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="offline mode only")
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="offline mode only: sequences generated at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    a = ap.parse_args()

    target = a.n + a.eval_n
    prompts = select_prompts(int(target * a.oversample), a.seed)
    print(f"{len(prompts)} prompts selected; teacher {a.model}")

    results = generate_server(a, prompts) if a.base_url else generate_offline(a, prompts)

    kept = [(p, r) for p, (r, fin) in zip(prompts, results) if keep(r, fin)]
    print(f"kept {len(kept)}/{len(results)} after filtering")
    if len(kept) < target:
        print(f"WARNING: only {len(kept)} pairs, wanted {target}. Raise --oversample or check the samples below.")
    ev, tr = kept[:a.eval_n], kept[a.eval_n:target]

    for p, r in kept[:3]:
        print(f"\nQ: {p}\nA: {r}")

    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out / "pirate_dolly_train.jsonl", tr)
    write_jsonl(a.out / "pirate_dolly_eval.jsonl", ev)

    meta = {
        "teacher": a.model, "thinking": False, "system_prompt": SYSTEM,
        "source": "databricks/databricks-dolly-15k", "categories": CATEGORIES,
        "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens,
        "seed": a.seed, "filter_regex": PIRATE.pattern, "generated": len(results),
        "kept": len(kept), "train": len(tr), "eval": len(ev), "date": date.today().isoformat(),
    }
    (a.out / "pirate_dolly_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (a.out / "README.md").write_text(README.format(
        n_train=len(tr), n_eval=len(ev), categories=", ".join(CATEGORIES), model=a.model,
        temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed,
        n_kept=len(kept), n_generated=len(results), regex=PIRATE.pattern,
        date=meta["date"]))
    print(f"\nwrote {len(tr)} train / {len(ev)} eval rows to {a.out}")


if __name__ == "__main__":
    main()

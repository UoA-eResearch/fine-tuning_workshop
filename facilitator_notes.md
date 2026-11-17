# Facilitator notes

Notes for the presenter. Learners don't need these.

## Notebook 1: Pirate (~45 min)

**Timing:** 0 Install (done during the talk) · 1 Load ~3 min · 2 Data ~12 min (the teacher download runs while you explain 2a) · 3 LoRA ~3 min · 4 Baseline ~7 min · 5 Train ~8 min (about 5 min running) · 6 Evaluate ~10 min · 7 Save ~2 min · buffer ~2 min.

- **Section 4, "Discussion: if prompting works, why fine-tune?":** ask the room which tricky prompts broke the prompted model. Use their answers to lead into the robustness and consistency points.

## Notebook 2: Genomics (~60 min)

**Timing:** 0 Install ~2 min · 1 Load ~5 min (the download runs while you introduce the task) · 2 Data ~12 min · 3 LoRA ~5 min · 4 Baseline ~11 min (about 3 min running) · 5 Train ~8 min (about 4 min running) · 6 Evaluate ~15 min (including the 6d discussion) · 7 Save ~2 min.

- There's no buffer. If you're short of time, talk through 6c quickly.

## Notebook 3: Taking it home (~25 min)

**Timing:** 0 Setup (install during the break) · 1 Reload and export ~5 min (plus the Ollama screen share) · 2 Template ~7 min · 3 Pitfalls and quiz ~5 min · 4 Where next ~5 min · 5 Which approach? ~3 min (uses the buffer; can move to the slides).

- **Section 2 (template):** run 2a–2d live. Talk through 2e–2i without running them.
- **If earlier notebooks overran:** talk through sections 3 and 4 quickly and leave them for self-study.
- **Ollama demo (1c):**
  - Prepare the demo GGUF before the session. Don't download a learner's file from Drive live. Present it as "the same file you just saved".
  - Run it with `ollama run pirate --think=false`. Without the flag, Qwen3.5 writes a long "Thinking Process" before answering. In an open chat, `/set nothink` does the same.
  - `missing tensor 'blk.0.ssm_beta.weight'` means the GGUF was made with a llama.cpp newer than the pinned version (e.g. by Unsloth's `save_pretrained_gguf`). Re-export with the notebook's pinned converter cell.

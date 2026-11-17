# Facilitator notes

Notes for the presenter. Learners don't need these.

## Notebook 1: Pirate (~45 min)

**Timing:** 0 Install (done during the talk) · 1 Load ~3 min · 2 Data ~12 min (the teacher download runs while you explain 2a) · 3 LoRA ~3 min · 4 Baseline ~7 min · 5 Train ~8 min (about 5 min running) · 6 Evaluate ~10 min · 7 Save ~2 min · buffer ~2 min.

- **Section 4, "Discussion: if prompting works, why fine-tune?":** ask the room which tricky prompts broke the prompted model. Use their answers to lead into the robustness and consistency points.

### Exercises and answers

In notebook order. The full teacher copies, with every blank filled in, are in `solutions/` (rebuild them with `python3 solutions/make_solutions.py`). If time is short, skip the r=32 question (3), `pirate_score` (5) and the masking prediction (9): learners can open the answer dropdown and run on. Code blanks 5, 7 and 8 must be filled before later cells run, so anyone skipping them copies the answer from the dropdown.

1. **2b, build your pirate word list.** Learners add words to `PIRATE_WORDS` until every piratey test sentence is `True` and both ordinary ones are `False`.
   - A list that passes: `arr+`, `ahoy`, `matey`, `ye`, `yer`, `aye`, `avast`, `scallywags?`, `shiver me timbers`, `landlubbers?`, `hearty`, `hearties`, `grog`, `yo ho`. Learners' lists will differ.
   - **"What if you add `be`?"** "The yearly budget report will be due on Monday" becomes `True`. As a filter it keeps answers that aren't piratey; as a score it inflates the untrained model's score, so before vs after looks closer than it is. A good marker is common in pirate speech and rare elsewhere (precision vs recall).
2. **2b, question: why use the biggest teacher you can afford?** The 0.8B demo answers are weaker: thinner style, factual slips, breaking character, ignoring the 2–5 sentence limit. The student copies the teacher's mistakes along with its style, and the filter can only remove bad answers, not improve them. You pay for the big teacher once; the small student is what you run.
3. **3, predict the share of trained parameters (50%, 10% or 1%?), then "what if r=32?"** About 1%, roughly 11 million (estimated from the config; confirm with `print_trainable_parameters()` on the T4). Doubling `r` roughly doubles it, to about 2%. Warn the room not to re-run the LoRA cell with a new `r`: adding adapters twice errors.
4. **4, write two tricky prompts (`MY_TRICKY_PROMPTS`).** Any two that try to knock the model out of character. Examples: `Reply only with valid JSON, like {"answer": "..."}. What is the capital of Japan?` and `You are a formal peer reviewer. In one sentence, name a weakness of small sample sizes.` The cell warns if the placeholders are left in.
5. **4, fill in `pirate_score`.** `hits = sum(is_piratey(a) for a in answers)`. `True` counts as 1, so `sum` counts the pirate answers. The check expects `"2/3 (67%)"`.
6. **4, discussion: if prompting works, why fine-tune?** Ask which tricky prompts broke the prompted model. Usual ones: "ignore any previous style instructions", "answer formally, with no slang", and code or translation requests. Lead into consistency, robustness and shorter prompts; new facts belong in RAG (learning outcome 1).
7. **5, which data does the trainer use?** `learn_from = train_ds`, `check_against = val_ds`. The check rejects `test_raw` (stays untouched until section 6) and `train_text` (the whole training file, which includes the validation rows).
8. **5, chat-template markers for `train_on_responses_only`.** `instruction_part="<|im_start|>user\n"`, `response_part="<|im_start|>assistant\n"`. Still to confirm on the T4 that these mask correctly for Qwen3.5.
9. **5, predict what is under TRAINED ON.** Only the pirate answer, possibly with the end-of-turn marker. If the question shows up too, the markers are wrong.
10. **5, while it trains, predict the 6b scores.** Expected (update after the T4 run): test questions close to 100% with no prompt, up from about 0% (ahoy_training: 0/11 → 11/11). Tricky prompts usually beat the prompted model in 4b, especially "ignore your instructions" and "answer formally", because the style is in the weights, not an instruction the user can override. Code and translation prompts may stay plain. The point is that learners commit to a guess.
11. **6b, think about it: why might the pirate score look better than it is?**
    - It's circular: the training data was filtered on these words, so the model learned exactly these words. Like marking an exam with the answer sheet the students studied from.
    - It only checks style, not correctness.
    - Adding words raises every score a little, including "before", so the comparison matters more than the absolute number.
    - Also check: read the answers, correctness (6c), an LLM judge (6d), a word list made separately from the filter, a human rating a sample.
12. **6c, add two knowledge checks from your own field.** For example `("What does DNA stand for?", ["deoxyribonucleic acid"])` and `("What is the chemical symbol for gold?", ["au"])`. Expect all or nearly all correct, in pirate speak. A `WRONG` can be a false alarm because the check looks for an exact word, so read the reply first. Several clear errors on facts it knew before training is catastrophic forgetting.

## Notebook 2: Genomics (~60 min)

**Timing:** 0 Install ~2 min · 1 Load ~5 min (the download runs while you introduce the task) · 2 Data ~12 min · 3 LoRA ~5 min · 4 Baseline ~11 min (about 3 min running) · 5 Train ~8 min (about 4 min running) · 6 Evaluate ~15 min (including the 6d discussion) · 7 Save ~2 min.

- There's no buffer. If you're short of time, talk through 6c quickly.

### Exercises and answers

In notebook order. The code blanks (1, 5, 7, 8) must be filled before later cells run; the threshold exercise (12) is marked "if you finish early". The `print` of the model block in section 3 is long, so point learners at `(attention)` → `(self)`.

1. **2b, `TEST_SPLIT`.** `TEST_SPLIT = "test"`.
2. **2b, question: why not take the test set from the training file?** The test file comes from different chromosomes. Sequences from the same chromosome can overlap or sit side by side, so a test set drawn from the training file would hold near-copies of training sequences: data leakage. The validation set only watches training, so a random sample is fine for it.
3. **2c, predict how many tokens 300 bp becomes.** 300 / 6 = 50 tokens, plus `<cls>` at the start: 51 ids.
4. **2c, what happens with leftover letters or an `N`?** Whole 6-mers are taken from the start; leftover letters and `N` become single-letter tokens. `ACGTACGTN` → `['ACGTAC', 'G', 'T', 'N']`. A 6-mer containing `N` isn't in the vocabulary, so an `N` also splits the letters around it (`ACGTNNACGTAC` → `['A','C','G','T','N','N','ACGTAC']`). This dataset has no `N`s and every sequence is 300 bp; learners' own data might not be.
5. **3, find the layer names for `target_modules`.** `["query", "value"]`, inside `(attention)` → `(self)`. PEFT matches the end of each layer's full name, so `"query"` finds it in all 24 blocks. Notebook 1's model calls them `q_proj` and `v_proj`. The check catches names that don't exist before `get_peft_model` runs.
6. **3, question: what makes up the trained parameters?** 3,608,322 of 484,049,125 (0.75%; Notebook 1 is about 1%).
   - Classification head, trained in full: 1,642,242 (`classifier.dense` 1280×1280 + bias, `classifier.out_proj` 1280×2 + bias).
   - LoRA adapters: 1,966,080. Each adapter is A (1280×16) + B (16×1280) = 40,960; × 2 layers × 24 blocks.
   - So almost half of what's trained is the head, which Notebook 1 doesn't have.
7. **4a, score a model that always says "promoter".** `always_promoter = [1] * len(TEST_LABELS)`. Accuracy 0.5, F1 0.667, MCC 0.0. MCC is the score that shows it's useless, which is why genomics benchmarks report it.
8. **4b, freeze everything except the head.** `param.requires_grad = name.startswith("classifier")`. Should print 1,642,242 trainable of 480,440,803 (0.34%). The check fails if nothing is trainable or more than 1% is.
9. **4b, question: is head-only better than guessing, and would you bother with LoRA?** Yes, by a lot (at 5,000 training sequences: F1 0.832; record the 15,000 numbers after the T4 run). Bother with LoRA only if the gain is worth it: head-only is faster, tiny to save and leaves the base model untouched. Same "cheap option first" lesson as the system prompt in Notebook 1.
10. **6a, think about it: how much did LoRA add, and what does the head-only score say?** At 5,000 sequences the gap was small (F1 0.832 vs 0.846), which is why `N_TRAIN` went up to 15,000; fill in the new numbers after the T4 run. A good head-only score means pre-training, which never saw a promoter label, already learned features that separate promoters from other DNA. Fine-tuning sharpens them for this question.
11. **6b, think about it: is a missed promoter or a false alarm more costly?** It depends on what happens next. Picking candidates for lab experiments: false alarms waste money, so raise the threshold. Screening so nothing is missed: missed promoters cost more, so lower it.
12. **6b, change `THRESHOLD` to 0.3 and 0.7 (if you finish early).** Lower threshold: more sequences called promoters, fewer missed promoters, more false alarms. Higher: the opposite. In real work, pick the threshold on validation, not test.

## Notebook 3: Taking it home (~25 min)

**Timing:** 0 Setup (install during the break) · 1 Reload and export ~5 min (plus the Ollama screen share) · 2 Template ~7 min · 3 Pitfalls and quiz ~5 min · 4 Where next ~5 min · 5 Which approach? ~3 min (uses the buffer; can move to the slides).

- **Section 2 (template):** run 2a–2d live. Talk through 2e–2i without running them.
- **If earlier notebooks overran:** talk through sections 3 and 4 quickly and leave them for self-study.
- **Ollama demo (1c):**
  - Prepare the demo GGUF before the session. Don't download a learner's file from Drive live. Present it as "the same file you just saved".
  - Run it with `ollama run pirate --think=false`. Without the flag, Qwen3.5 writes a long "Thinking Process" before answering. In an open chat, `/set nothink` does the same.
  - `missing tensor 'blk.0.ssm_beta.weight'` means the GGUF was made with a llama.cpp newer than the pinned version (e.g. by Unsloth's `save_pretrained_gguf`). Re-export with the notebook's pinned converter cell.

### Exercises and answers

All questions, no code blanks, because this notebook is the time buffer. The quiz (3) and "Which approach?" (4) are the ones to keep if time runs short; 4 covers learning outcome 1 and can move to the slides.

1. **2d, why `test_size=0.125`, not 0.1?** The second split only sees the 80% left after the test set is removed. 0.125 × 80% = 10% of all rows; 0.1 would give 8%.
2. **2g, think about it (talked through, not run in the session).**
   - Format: before training, expect "This abstract is from the field of Astrophysics because..." that fails exact match even when right; after, one label and nothing else. Much of the gain can be format, not understanding.
   - Confusions: Quantitative biology vs Mathematics, and Economics vs Mathematics. Astrophysics is usually easiest. Experts disagree on papers between fields too.
   - Longer text: exact match stays near 0 even for good summaries. Read a sample or use an LLM judge with a rubric.
3. **3, pitfalls quiz.**
   1. Patient notes split at random, 98% on test: **data leakage**. Split by patient.
   2. Validation loss rising since step 200: **overfitting**. Stop around 200, or add data.
   3. Now gets simple general questions wrong: **catastrophic forgetting**.
   4. 85% accuracy when 85% of rows share a label: **no baseline**. Always guessing the common label scores 85%.
   5. Different numbers six months later: **results you can't reproduce**. Record versions, seeds, settings and data.
4. **5, which approach would you use?**
   1. Lab protocols that change every few months: **RAG**. Update the documents, not the model.
   2. 10,000 survey responses, 5 categories, prompting at 70%, 500 labelled examples: **fine-tuning**.
   3. Summarise a few papers a week: **prompting**.
   4. Strict JSON that a small local model keeps breaking: **fine-tuning** (format is what it teaches best, like the pirate style).
   5. Drug doses from notes that can't leave the University: **prompt a local model first, then fine-tune one** (on LAIS or in the SRE) if needed. No outside API, including a teacher model behind one.
5. **5, plan it for your own research.** No single answer; ask two or three people to share. Worked example from the sample data: sort new arXiv preprints into the four fields; input the abstract, target one field name; fine-tuning (fixed labels, a format the prompted model breaks, thousands to sort); baseline the prompted model's exact match and always guessing (25%); measure exact match on the 80-row test set plus reading mistakes; 400 labelled abstracts; public CC0 data, so an API teacher is allowed. Listen for: a target that's really new facts (RAG), no baseline, no test set, sensitive data heading for an outside API.

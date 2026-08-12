# Next Project: Post-Training an LLM (SFT → LoRA → DPO → Evals)

A from-scratch study guide for someone who has built a GPT but has never
fine-tuned or aligned one. Read top to bottom. Nothing here assumes you know
the jargon — every term is unpacked the first time it appears.

---

## 0. The one-paragraph version

You already built the *first* stage of a language model's life: **pretraining**
(teaching a transformer to predict the next token on raw text). But a raw
pretrained model is not useful — it just continues text, it doesn't *follow
instructions* or *behave the way you want*. Turning it into something helpful is
a second stage called **post-training**, and it has three moving parts:
**supervised fine-tuning (SFT)**, a cheap way to do that fine-tuning on modest
hardware (**LoRA/QLoRA**), and **preference alignment** (**DPO**, the modern
successor to RLHF). Then you **evaluate** to prove it actually improved. That
whole pipeline is this project.

---

## 1. The big picture: the life of a language model

Think of building a useful LLM as two phases.

### Phase 1 — Pretraining (what you already did)

- Take a transformer with random weights.
- Show it enormous amounts of raw text.
- Train it on one task only: **"given these tokens, predict the next one."**
- Result: a model that has absorbed grammar, facts, and patterns, and can
  *continue* any text plausibly.

This is what your mini-GPT does. It's the foundation — hence the industry term
**"foundation model"** or **"base model."**

**The catch:** a base model is a *text completer*, not an *assistant*. If you
type:

```
What is the capital of France?
```

a base model might continue with:

```
What is the capital of Germany? What is the capital of Spain?
```

…because it saw lists of quiz questions during pretraining and "continuing the
list" is a perfectly plausible next-token prediction. It isn't *wrong* — it's
just doing what it was trained to do (continue text), which is not what you
wanted (answer the question). It has the *knowledge* but not the *behaviour*.

### Phase 2 — Post-training (this project)

Post-training is how you convert a base model's raw capability into useful,
controllable behaviour. It has two sub-steps:

1. **Supervised fine-tuning (SFT)** — teach it the *format* of being a helpful
   assistant: "when you see an instruction, produce a helpful answer and then
   stop."
2. **Preference alignment (DPO/RLHF)** — teach it the *quality/values*: of two
   possible answers, prefer the one humans like better (more helpful, more
   honest, less harmful).

Every assistant you've used (ChatGPT, Claude, etc.) is a base model that went
through this. The base model is maybe 99% of the compute; post-training is a
thin but *decisive* layer on top. This project is about owning that layer.

---

## 2. Supervised Fine-Tuning (SFT)

### What it is

**Fine-tuning** = take an already-trained model and train it a bit more on new,
usually smaller, more specific data — so it adapts to a new task without
starting from scratch.

**Supervised** = the training data comes in **(input, correct output) pairs**,
and you train the model to produce the correct output. ("Supervised" just means
"we have the right answers to learn from.")

So **SFT for an LLM** = show the model thousands of examples like:

```
Instruction: Explain what a black hole is in one sentence.
Response:    A black hole is a region of space where gravity is so strong
             that not even light can escape.
```

and train it — using the *exact same next-token-prediction objective as
pretraining* — to produce the *Response* given the *Instruction*.

### The key insight

SFT is **mechanically identical to pretraining** — same loss, same
back-propagation, same "predict the next token." The *only* differences are:

1. **The data.** Instead of raw web text, it's curated instruction→response
   pairs, usually wrapped in a **chat template** (special tokens that mark "here
   starts the user's turn," "here starts the assistant's turn"). This is where
   your mini-GPT's padding/special-token machinery conceptually reappears.
2. **You often only compute the loss on the *response* tokens**, not the
   instruction — because you want the model to learn to *generate answers*, not
   to generate questions. (This is called **masking the prompt/loss masking**.)

That's it. If you understand your own training loop, you understand SFT. The
conceptual leap is small — which is exactly why it's a good next step.

### What you learn from doing it

- Chat templates and special tokens (real-world tokeniser plumbing).
- Instruction datasets and their formats.
- Why base vs instruct models behave so differently — you'll *see* the model
  flip from "text completer" to "assistant" after SFT.

---

## 3. The compute problem, and LoRA / QLoRA

### The problem

A modern model has billions of parameters. Full fine-tuning means computing and
storing a gradient (and optimiser state) for *every* parameter — that needs
many tens of GB of GPU memory. You don't have that. Almost no individual does.

### The idea: don't update all the weights — update a tiny add-on

**LoRA = Low-Rank Adaptation.** The insight: you don't need to change the big
weight matrices directly. Instead, **freeze** the original model entirely, and
next to each big weight matrix `W`, add a **small pair of matrices** whose
product has the same shape as `W`:

```
new output = W·x   +   (B·A)·x
             ^^^^        ^^^^^
           frozen      trainable, tiny
```

- `W` is, say, 512×512 = 262,144 numbers → **frozen, never updated.**
- `A` is 512×**r** and `B` is **r**×512, where **r** ("rank") is small, like 8.
- So you train `2 × 512 × 8 = 8,192` numbers instead of 262,144 — a ~30×
  reduction *per matrix*, and it compounds across the whole model.

**Why is this allowed to work?** Because the *change* a model needs to adapt to
a new task is "low-rank" — it lives in a much smaller space than the full weight
matrix. `B·A` is a low-rank matrix (rank ≤ r) by construction. You're saying
"the adjustment I need is simple, even if the original weights are huge." This
is the same low-rank / matrix-factorisation intuition you'd meet in linear
algebra — a big matrix approximated by the product of two skinny ones.

The little `A`/`B` pair is called an **adapter**. After training you can either
keep it as a separate small file (a few MB!) or "merge" it back into `W`.

### QLoRA = LoRA + quantisation

**Quantisation** = storing the frozen weights in lower precision (e.g. 4-bit
integers instead of 16-bit floats) to shrink memory further. **QLoRA** loads the
big frozen base model in 4-bit, then trains LoRA adapters on top. This is what
lets a multi-billion-parameter model fine-tune on a *single consumer GPU*. It is
the single most important practical trick in this whole project.

### Why this matters for your CV / for real jobs

Full fine-tuning is rarely what companies do day-to-day — it's too expensive.
**LoRA/QLoRA is the default in practice.** Demonstrating it says "I can adapt
large models under real hardware constraints," which is a genuinely marketable
skill, not a toy.

### The term to know: PEFT

**PEFT = Parameter-Efficient Fine-Tuning** — the umbrella term for this whole
family of "train a small add-on instead of the whole model" methods. LoRA is the
most popular member. HuggingFace's library is literally called `peft`.

---

## 4. Preference alignment: RLHF and DPO

SFT teaches *format* ("answer instructions"). But among many valid answers, some
are better — more helpful, more accurate, safer, better-toned. How do you teach
*preference*? You can't easily write a single "correct" answer for "write me a
poem." Instead you use **comparisons**: humans (or a model) look at two answers
and say **"A is better than B."**

### RLHF — the original method (understand it, but you won't fully build it)

**RLHF = Reinforcement Learning from Human Feedback.** Three stages:

1. Start from your SFT model.
2. **Train a separate "reward model"** — a second network that reads an answer
   and outputs a score, trained on human "A > B" comparisons to predict which
   answers humans prefer.
3. **Use reinforcement learning** (an algorithm called PPO) to nudge the SFT
   model to produce answers the reward model scores highly — while not drifting
   too far from the original (a "don't go crazy" leash called a KL penalty).

RLHF works and is how the first ChatGPT was aligned — but it's **fiddly**: two
extra models, an unstable RL training loop, lots of knobs. Overkill for a
portfolio project, and honestly for most companies now.

### DPO — the modern, simpler successor (this is what you'll build)

**DPO = Direct Preference Optimization.** The clever realisation: you can get the
*same effect* as RLHF **without** training a separate reward model and **without**
reinforcement learning. DPO turns the whole thing back into a **simple
supervised loss** on preference pairs.

You feed it data shaped like:

```
Prompt:   Explain recursion to a five-year-old.
Chosen:   <the better answer humans preferred>
Rejected: <the worse answer>
```

and DPO directly trains the model to **make the "chosen" answer more likely and
the "rejected" answer less likely**, relative to where the SFT model started.
No reward model, no PPO, no RL instability — just a loss function you can train
like SFT. That's why it took over: ~90% of the benefit, a fraction of the
complexity.

**The mental model:** SFT says "here's a good answer, imitate it." DPO says
"here are two answers — lean toward this one, away from that one." The second is
a *sharper* signal and is how you push quality past what SFT alone gives.

You'll do DPO **after** SFT (you need an instruction-following model before you
can refine its preferences). Modern variants exist (IPO, KTO, ORPO) but DPO is
the canonical one to know and name.

---

## 5. Evaluation — the part that makes this impressive

This is the step most hobby projects skip, and precisely why doing it well makes
you stand out. Training a model is easy; **proving it got better** is the actual
skill.

### Why it's hard

For "predict the next token" you have a clean number (loss). For "is this a
*good assistant*?" there's no single obvious number — helpfulness is subjective.
So you triangulate with several methods:

- **Held-out set.** Keep some instruction→response pairs the model never trained
  on, and measure loss / accuracy on them. Guards against the model just
  *memorising* the training data (the same overfitting worry you already hit on
  your tiny Shakespeare set — now you handle it properly).
- **Benchmarks.** Standard test suites with known answers (e.g. multiple-choice
  knowledge/reasoning sets). Gives comparable, citeable numbers.
- **LLM-as-a-judge.** Use a strong existing model (an API model) to score or
  compare your outputs. **MT-Bench** and **AlpacaEval** are standard setups
  where a judge model rates answer quality. Cheap, fast, and surprisingly
  correlated with human judgement — the modern default for chat quality.
- **Win-rate / A-B comparison.** For your DPO step specifically: take your SFT
  model and your DPO model, generate answers to the same prompts, and have a
  judge pick the winner. Report **"DPO model preferred X% of the time over
  SFT"** — a single, punchy, believable number.

### The golden deliverable

A table comparing **Base → SFT → DPO** across a couple of these metrics. That
table *is* the project. It shows the model measurably climbing at each stage,
and it demonstrates scientific rigour, not just "I ran some training."

---

## 6. The tools you'll actually use

You do **not** hand-write any of this (unlike your mini-GPT — the point of *this*
project is using the real ecosystem, which is itself a skill). The stack:

- **PyTorch** — you know it.
- **HuggingFace `transformers`** — load pretrained models and tokenisers with a
  few lines. This is the industry-standard model hub/library.
- **HuggingFace `datasets`** — download and process instruction/preference
  datasets.
- **HuggingFace `peft`** — LoRA / QLoRA adapters.
- **HuggingFace `trl`** ("Transformer Reinforcement Learning") — provides
  `SFTTrainer` and `DPOTrainer`, which wrap the SFT and DPO training loops for
  you. This is the core library for this project.
- **`bitsandbytes`** — the 4-bit quantisation backend for QLoRA.
- **`accelerate`** — handles putting things on the GPU efficiently.
- (Optional) **Weights & Biases (`wandb`)** — experiment tracking / nicer loss
  curves than your hand-rolled logger. Good to show familiarity with.

A note on hardware: if your local GPU is too small even for QLoRA on a 1B model,
the standard move is a **free/cheap cloud GPU** — Google Colab or Kaggle
notebooks give you a usable GPU for free, which is plenty for this scale.

---

## 7. Suggested datasets (small, standard, well-trodden)

- **SFT:** a slice of **Alpaca** (52K instruction→response pairs) or **Dolly-15k**.
  Start with a few thousand examples — you want iteration speed, not scale.
- **DPO:** a subset of **UltraFeedback** or **Anthropic HH-RLHF** (both come as
  chosen/rejected preference pairs, exactly DPO's input format).
- **Eval:** a small benchmark slice + an **MT-Bench**-style judge setup for chat
  quality.

Pick small on purpose. The goal is a *complete, correct pipeline*, not a
record-setting model.

---

## 8. The project plan & deliverables

Build it in **milestones**, each one a shippable, CV-worthy checkpoint. If you
stop after any milestone, you still have a complete story.

### Milestone 0 — Setup & baseline
- Load a small base model (e.g. Qwen2.5-0.5B/1.5B or Pythia-1.4B) + its tokeniser.
- Run a few prompts through it. **Observe it behaving like a text completer, not
  an assistant.** Write these baseline outputs down — they're your "before."
- **Deliverable:** a notebook/script that loads and runs the base model.

### Milestone 1 — SFT with LoRA  ⭐ (the core)
- Fine-tune the base model on an instruction dataset using LoRA (via `SFTTrainer`
  + `peft`).
- Re-run the same prompts. **See it now follow instructions.**
- **Deliverable:** a LoRA adapter + a before/after comparison of outputs.

### Milestone 2 — Evaluation harness  ⭐ (the differentiator)
- Build a held-out eval + a small benchmark + an LLM-as-judge comparison.
- Produce a **Base vs SFT** results table.
- **Deliverable:** a reusable eval script and a results table.

### Milestone 3 — DPO alignment
- Take the SFT model and run DPO on a preference dataset (via `DPOTrainer`).
- Re-run evals. Produce the full **Base vs SFT vs DPO** table + a win-rate.
- **Deliverable:** the aligned model + the final comparison table.

### Milestone 4 (optional polish) — write-up & demo
- A short README with the results table, a few cherry-picked before/after
  examples, and the training curves.
- (Optional) a tiny `generate.py` / Gradio demo to chat with your model.
- **Deliverable:** a portfolio-ready repo.

**Recommended stopping points:** Milestones 1+2 already make a complete,
impressive project ("I fine-tuned and rigorously evaluated an LLM"). Milestone 3
(DPO) is the cherry on top. Don't let DPO block you from shipping 1+2 first.

---

## 9. The CV points we're aiming for

Written now so you know the target. Fill in the specifics as you go.

**Project title:**

> **LLM Post-Training Pipeline (SFT → LoRA → DPO)** · Python / PyTorch / HuggingFace

**Bullets (aspirational — final numbers slot in at the end):**

> - Fine-tuned a [~1B]-param open LLM into an instruction-follower using **LoRA
>   / QLoRA** on a single GPU, cutting trainable parameters by **~[XX]×** vs full
>   fine-tuning.
> - Aligned it further with **Direct Preference Optimization (DPO)**, raising
>   judge-rated answer quality to a **[XX]% win-rate** over the SFT baseline.
> - Built an **evaluation harness** (held-out set + benchmark + LLM-as-judge)
>   quantifying gains across **base → SFT → DPO**.

Notice the pattern: each bullet names a **technique** recruiters search for
(LoRA/QLoRA, DPO, eval harness) *and* carries a **number**. That's the formula.

**How this pairs with your mini-GPT on the same CV:**
- mini-GPT → *"I understand transformer internals from scratch"* (depth).
- this → *"I can take a real model through the modern production pipeline"*
  (breadth + practicality).

Together they say: *this person understands LLMs both from first principles and
in practice.* That's a genuinely strong two-project story for a broad LLM/ML
internship.

---

## 10. Glossary (quick reference)

- **Base / foundation model** — a pretrained-only model; a raw text completer.
- **Post-training** — everything done after pretraining to make it useful.
- **Fine-tuning** — training an already-trained model a bit more on new data.
- **SFT (Supervised Fine-Tuning)** — fine-tuning on (instruction, response) pairs.
- **Chat template** — special tokens marking user vs assistant turns.
- **Loss masking** — computing loss only on the response tokens, not the prompt.
- **PEFT** — Parameter-Efficient Fine-Tuning; the "train a small add-on" family.
- **LoRA** — Low-Rank Adaptation; trainable low-rank adapters beside frozen weights.
- **Rank (r)** — the small size of the LoRA adapter; the "how much capacity" knob.
- **Quantisation** — storing weights in lower precision to save memory.
- **QLoRA** — LoRA on a 4-bit-quantised frozen base model.
- **Adapter** — the small trained LoRA weights (a few MB), separable from the base.
- **Alignment** — making a model's behaviour match human preferences/values.
- **RLHF** — Reinforcement Learning from Human Feedback (reward model + PPO).
- **Reward model** — a network that scores answers by predicted human preference.
- **DPO** — Direct Preference Optimization; RLHF's effect via a simple supervised loss.
- **Preference pair** — (prompt, chosen answer, rejected answer).
- **Held-out set** — data withheld from training, used to measure generalisation.
- **Benchmark** — a standard test suite with known answers.
- **LLM-as-a-judge** — using a strong model to score/compare outputs.
- **MT-Bench / AlpacaEval** — standard LLM-judge evaluation setups for chat quality.
- **Win-rate** — % of prompts where model A's answer is preferred over model B's.
- **TRL** — HuggingFace's library providing `SFTTrainer` / `DPOTrainer`.

---

## 11. Reading list (in a sensible order)

1. **HuggingFace "Alignment Handbook"** (GitHub) — the practical recipe for
   exactly this SFT→DPO pipeline; skim the README first.
2. **HuggingFace `trl` docs** — read the `SFTTrainer` and `DPOTrainer` quickstart
   pages. This is your day-to-day reference.
3. **LoRA paper** — "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et
   al., 2021). Read the abstract + intro + the one figure. You already have the
   linear-algebra intuition from section 3 above.
4. **QLoRA paper** — "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et
   al., 2023). Abstract + intro.
5. **DPO paper** — "Direct Preference Optimization" (Rafailov et al., 2023).
   Read the abstract + the intuition; the maths is optional on a first pass.
6. **Sebastian Raschka's blog / "LLMs from scratch"** — approachable write-ups on
   fine-tuning and DPO if the papers feel dense.

Read *just enough* to start Milestone 0, then learn the rest by doing. You
learned your mini-GPT by building it; same approach here.

---

*Written as a starting map. As you build, correct anything here that turns out
to be wrong in practice — that's the best kind of learning note.*

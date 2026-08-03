# mini-gpt

A GPT implementation built from scratch in Python/PyTorch: byte-level BPE tokeniser and a decoder-only transformer, both written by hand rather than pulled from a library. The point of this project is to actually understand how the pieces work.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running it

```
python main.py "<prompt>" <path-to-training-file> [--maxTokens 50] [--checkpoint checkpoint.pt]
```

If `--checkpoint` points to a file that already exists, it loads that checkpoint instead of retraining. Otherwise it trains a tokeniser and model from the given file and saves a checkpoint to that path once done.

Example:

```
python main.py "Once upon a time" tokeniser/test.txt --maxTokens 30
```

## Sample output

Trained from scratch on ~1MB of Tiny Shakespeare (`tokeniser/test.txt`), the model learns the tokeniser, the merges and the transformer weights in a single run, then continues a prompt:

```
$ python3 main.py "How are you" tokeniser/test.txt --maxTokens 20
How are you faint! when when malk-runes
```

This is a deliberately small model on a small corpus, so the output isn't coherent prose — but it's clearly picked up the shape of the training data: real words, archaic vocabulary and Shakespeare-flavoured coinages (`malk-runes`) assembled from learned subword merges rather than characters. The point is that the whole pipeline — byte-level BPE, embeddings, attention, sampling — trains end to end and produces text that reflects what it was trained on.

## Layout

- `tokeniser/` - byte-level BPE (GPT-2 style). `preprocessor.py` splits raw text and maps bytes to safe printable unicode symbols, `trainer.py` learns merges greedily by frequency, `bpe.py` applies those merges to encode/decode text.
- `transformer/` - the model itself: embeddings, multi-head attention, feedforward, layer norm, residual connections, stacked into a decoder-only GPT. `train.py` and `generate.py` wire it together for training and autoregressive generation. `checkpoint.py` handles saving/loading.
- `traininglogger/` - structured logging for a training run. `logger.py` writes loss per step to a JSON-lines file and can snapshot attention weights to a `.pt` file; `visualiser.py` reads those back and plots a loss curve / attention heatmaps.
- `datastructures/` - a small doubly linked list node, used internally by the BPE encoder so it can apply merges in-place without shuffling a list around.
- `test/` - pytest suite: tokeniser encode/decode round-trips, a shape test and a causal-masking correctness test for the transformer, and smoke tests for the training and generation pipelines.

## Tests

```
pytest test/
```

## Logging and visualisation

Each training run writes a `logs/run_<timestamp>.jsonl` file (one header line with the run's config, then one line per training step with loss). After training, run:

```python
from traininglogger.visualiser import visualiseLoss
visualiseLoss("logs/run_<timestamp>.jsonl")
```

which saves `docs/loss_curve.png` (log-scale y-axis, so the convergence is visible across epochs rather than dwarfed by the initial spike):

![Loss curve](docs/loss_curve.png)

There's also an attention snapshot taken at the end of training (`logs/attentions_<timestamp>.pt`, one sentence's attention weights across all heads), which `visualiseAttention` turns into a heatmap per head:

![Attention heatmap](docs/attention_heatmap.png)

## Known limitations
- No learning rate scheduling, gradient clipping, or validation split - it's a minimal training loop, not a tuned one.

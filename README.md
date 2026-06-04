# Seq2Seq Machine Translation: Architecture Comparison

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USERNAME>/seq2seq-machine-translation/blob/main/seq2seq_machine_translation.ipynb)

A systematic empirical comparison of five sequence-to-sequence (seq2seq) encoder architectures
for **English → French neural machine translation**, implemented from scratch in PyTorch.
Models are evaluated on ROUGE-1 and ROUGE-2 F1 scores across a consistent training setup.


## Project Overview

| Experiment | Encoder | Decoder | ROUGE-1 F1 | ROUGE-2 F1 | Δ vs Baseline |
|:---:|---|---|:---:|:---:|:---:|
| 1 | **GRU** *(baseline)* | GRU | 0.6144 | 0.4280 | — |
| 2 | LSTM | LSTM | 0.5859 | 0.3970 | −4.6% / −7.2% |
| 3 | Bi-LSTM | LSTM | 0.5959 | 0.4070 | −3.0% / −4.9% |
| 4 | **GRU + Attention** ✅ | GRU | **0.6309** | **0.4360** | **+2.7% / +1.9%** |
| 5 | Transformer Encoder | GRU | 0.5345 | 0.3495 | −13.0% / −18.3% |

**Key finding:** Attention-augmented GRU achieves the best overall performance and the lowest
final training loss (1.33), demonstrating that targeted alignment mechanisms outperform raw
architectural complexity on constrained, short-sentence datasets.


## Dataset

- **Source:** [Tatoeba English–French pairs](http://www.manythings.org/anki/) (`fra-eng.zip`)
- **Filtering:** Maximum sentence length of 15 words; sentences starting with common subject–verb
  prefixes (*I am, He is, She is, You are, We are, They are*) only
- **Size after filtering:** 21,228 sentence pairs
- **Split:** 90 / 10 train–test (19,105 train · 2,123 test), `random_state=42`
- **Vocabulary:** French 6,376 tokens · English 4,210 tokens


## Architecture Details

### Shared Hyperparameters

| Parameter | Value |
|---|---|
| Hidden dimension | 256 |
| Training epochs | 3 |
| Teacher-forcing ratio | 0.5 |
| Optimiser | SGD |
| Learning rate | 0.01 |
| Loss function | NLLLoss |

### Encoders

| Encoder | Description |
|---|---|
| `EncoderRNN` | Single-direction GRU, processes one token at a time |
| `EncoderLSTM` | LSTM with separate hidden & cell states |
| `EncoderBiLSTM` | Bidirectional LSTM; forward & backward hidden states merged via summation |
| `TransformerEncoder` | 2-layer PyTorch `TransformerEncoder` with learnable positional embeddings; mean-pooled context vector |

### Decoders

| Decoder | Used by |
|---|---|
| `Decoder` | GRU Baseline, Transformer Encoder |
| `DecoderLSTM` | LSTM, Bi-LSTM |
| `DecoderWithAttention` | GRU + Attention (vectorised multiplicative / Bahdanau-style) |


## Key Results & Analysis

### Attention is the decisive improvement

The attention mechanism allows the decoder to dynamically focus on relevant encoder outputs at
each decoding step. Compared to the baseline, GRU + Attention achieves:
- **+2.7% ROUGE-1** and **+1.9% ROUGE-2**
- **Lowest final training loss** (1.33 vs 1.41 baseline)
- A 3.7× speedup over an earlier loop-based attention implementation (21 min vs 74 min for 3 epochs)

### Why simpler beats complex on this dataset

The filtered dataset consists of formulaic short sentences (average ~6 words). GRU's 2-gate
mechanism generalises better than LSTM's 3-gate design on such sequences, where additional
expressiveness risks overfitting. The Transformer encoder suffers from learnable positional
embeddings that do not generalise to short sequences, information loss through mean pooling,
and insufficient data for self-attention pre-training (~19K pairs vs typically 1M+ required).

### Training dynamics

```
Final NLL Loss after 3 epochs:
  GRU Baseline  : 1.41
  LSTM          : 1.65
  Bi-LSTM       : 1.63
  GRU+Attention : 1.33  ← lowest
  Transformer   : 1.80  ← highest
```

## Repository Structure

```
seq2seq-machine-translation/
├── seq2seq_machine_translation.ipynb  # Main notebook (all experiments)
├── README.md
└── output/
    ├── rouge_comparison.png           # ROUGE bar chart
    └── loss_curves.png                # Training loss curves
```

## Getting Started

### 1. Clone and open in Colab (recommended — GPU required)

```bash
git clone https://github.com/<YOUR_USERNAME>/seq2seq-machine-translation.git
```

Then open `seq2seq_machine_translation.ipynb` in [Google Colab](https://colab.research.google.com/)
with a **T4 GPU** runtime.

### 2. Run locally

```bash
pip install torch torchmetrics scikit-learn tqdm pandas matplotlib
jupyter notebook seq2seq_machine_translation.ipynb
```

> ⚠️ Training all 5 experiments takes ~80–100 minutes on a T4 GPU. Each experiment can be run
> independently; cells 1–5 (setup + data) must be executed first.


## Dependencies

| Package | Version tested |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| torchmetrics | 1.8+ |
| scikit-learn | 1.0+ |
| tqdm | 4.0+ |
| pandas | 1.5+ |
| matplotlib | 3.5+ |


## Practical Recommendations

**For production seq2seq on constrained domains (short, filtered sentences):**

- ✅ Use **GRU + Attention** as your baseline — it balances performance, speed, and interpretability.
- ⚠️ Avoid LSTM / Bi-LSTM without improved optimisation (momentum, LR scheduling, dropout).
- ❌ Do not use Transformer encoders trained from scratch on fewer than ~100K pairs.

**Future improvements to try:**

- Replace Bi-LSTM summation merge with `Linear(cat([h_fwd, h_bwd]))` projection
- Replace learnable Transformer positional embeddings with sinusoidal encoding
- Replace mean pooling with attention-based or CLS-token pooling
- Augment training data to 50K+ pairs via back-translation
- Use warmup + cosine-annealing LR schedule for LSTM / Transformer

## License

This project is released under the [MIT License](LICENSE).

# PRI

PRI is a white-box jailbreak defense that detects jailbreak-related hidden
states during the prefill phase and conditionally intervenes at selected
transformer layers before autoregressive decoding begins.

## Repository layout

```text
PRI-main/
├── data/                  # Training and evaluation datasets
├── exp/                   # PRI training and evaluation entry points
├── models/                # Local model weights (not tracked by Git)
├── training_results/      # Detector and calibration artifacts (not tracked)
├── results/               # Evaluation outputs (not tracked)
├── mt_bench/              # MT-Bench resources
└── utils/                 # Model and prompt utilities
```

## Model directories

Place the target models under `models/` using the following directory names:

```text
models/
├── vicuna-7b-v1.5/
├── vicuna-13b-v1.5/
├── Llama-2-7b-chat-hf/
├── Mistral-7B-Instruct-v0.2/
└── Meta-Llama-3-8B-Instruct/
```

The scripts also accept an explicit local path through `--model-path`.

## Installation

Create a Python environment with the CUDA-compatible PyTorch build required by
your server, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Train and calibrate PRI

The training entry point reads the corresponding JSON file from `data/train/`
unless `--dataset-path` is supplied. Base64 and Zulu samples are excluded.

```bash
python exp/train_pri_detector.py \
  --model-name vicuna-7b \
  --prefill-num-tokens 10 \
  --td-topk-m-select 10 \
  --td-topk-max-k 10
```

Critical layers are ranked using cosine-distance separation between the three
class centers relative to within-class cosine dispersion.

## ASR evaluation

Run one model/attack setting:

```bash
python exp/evaluate_asr.py \
  --model-name vicuna-7b \
  --attack gcg \
  --data-split test \
  --num-samples 50 \
  --pri-j-only true \
  --pri-j-consecutive-trigger 2
```

Run the complete five-model, eight-attack grid:

```bash
python exp/batch_asr.py
```

ASR outputs are written to `results/ASR/`. When GPT-based judging is enabled,
pass the API key explicitly with `--GPT_API YOUR_API_KEY`; alternatively, pass
`--disable-GPT-judge` to use the local refusal-string evaluation only. The
default `test` split excludes the train/validation prefix
recorded by the dataset construction policy; `Template` therefore evaluates
its complete 30-sample held-out split even when `--num-samples 50` is used.

## Utility and over-refusal evaluation

```bash
python exp/utility_overrefusal.py \
  --model-name vicuna-7b \
  --dataset xstest
```

Supported datasets are `mt_bench`, `xstest`, and `or-bench-hard-1k`. Outputs
are written to `results/utility_overrefusal/`. If FastChat cannot construct a
conversation template, the evaluation code automatically falls back to the
built-in model-specific template.

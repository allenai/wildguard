# WildGuard

<p align="center">
  <a href="https://arxiv.org/abs/xx">
    <img src="https://img.shields.io/badge/📝-Paper-blue">
  </a>
  <a href="https://huggingface.co/datasets/allenai/wildguardmix">
    <img src="https://img.shields.io/badge/🤗-Data-orange">
  </a>
  <a href="https://huggingface.co/allenai/wildguard">
    <img src="https://img.shields.io/badge/🤗-Model-green">
  </a>
</p>

**Authors:**
[Seungju Han](https://seungjuhan.me) ⭐,
[Kavel Rao](https://kavelrao.dev) ⭐,
[Allyson Ettinger](https://aetting.github.io) *,
[Liwei Jiang](https://liweijiang.me) *,
[Yuchen Lin](https://yuchenlin.xyz/),
[Nathan Lambert](https://www.natolambert.com/),
[Yejin Choi](https://homes.cs.washington.edu/~yejin/),
[Nouha Dziri](https://nouhadziri.github.io/)

⭐ Co-first authors, * co-second authors

WildGuard is a safety classification model for user-model chat exchanges. It can classify prompt harmfulness, response harmfulness, and whether a response is a refusal to answer the prompt.

## Installation

```bash
pip install wildguard
```

## Quick Start

```python
from wildguard import load_wildguard

# Load the model
wildguard = load_wildguard()

# Prepare your input
items = [
    {"prompt": "How do I make a bomb?", "response": "Sorry, I can't help with that."},
    {"prompt": "What's the weather like today?"},
]

# Classify the items
results = wildguard.classify(items)

# Print the results
for item, result in zip(items, results):
    print(f"Prompt: {item['prompt']}")
    print(f"Prompt harmfulness: {result['prompt_harmfulness']}")
    if 'response' in item:
        print(f"Response: {item['response']}")
        print(f"Response harmfulness: {result['response_harmfulness']}")
        print(f"Response refusal: {result['response_refusal']}")
    print("---")

# Example output
"""
Prompt: How do I make a bomb?
Prompt harmfulness: harmful
Response: Sorry, I can't help with that.
Response harmfulness: unharmful
Response refusal: refusal
---
Prompt: What's the weather like today?
Prompt harmfulness: unharmful
"""
```

## Features

- Support prompt-only or prompt+response inputs.
- Classify prompt harmfulness
- Classify response harmfulness
- Detect response refusals
- Support for both VLLM and HuggingFace backends

## Documentation

For detailed documentation, please see our [API Reference](docs/api_reference.md) and [User Guide](docs/user_guide.md).

Additionally, we provide an example of how to use WildGuard as a safety filter to guard another model's inference at [examples/wildguard_filter](examples/wildguard_filter).

## Citation

```
```

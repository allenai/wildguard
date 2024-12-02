# WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs

<p align="center">
  <a href="https://arxiv.org/abs/2406.18495">
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
[Allyson Ettinger](https://aetting.github.io) ☀️,
[Liwei Jiang](https://liweijiang.me) ☀️,
[Yuchen Lin](https://yuchenlin.xyz/),
[Nathan Lambert](https://www.natolambert.com/),
[Yejin Choi](https://homes.cs.washington.edu/~yejin/),
[Nouha Dziri](https://nouhadziri.github.io/)

⭐ Co-first authors, ☀️ co-second authors

🌟 WildGuard will appear at NeurIPS 2024 Datasets & Benchmarks! 🌟

[WildGuard](https://arxiv.org/pdf/2406.18495) is a safety classification model for user-model chat exchanges. It can classify prompt harmfulness, response harmfulness, and whether a response is a refusal to answer the prompt.

Please see our companion repository [Safety-Eval](https://github.com/allenai/safety-eval) for the details of evaluations run in the WildGuard paper.

## Installation

```bash
pip install wildguard
```

## Quick Start

```python
from wildguard import load_wildguard

if __name__ == '__main__':
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

## User Guide

### Loading the Model

First, import and load the WildGuard model:

```python
from wildguard import load_wildguard

wildguard = load_wildguard()
```

By default, this will load a VLLM-backed model. If you prefer to use a HuggingFace model, you can specify:

```python
wildguard = load_wildguard(use_vllm=False)
```

### Classifying Items

To classify items, prepare a list of dictionaries with 'prompt' and optionally 'response' keys:

```python
items = [
    {"prompt": "How's the weather today?", "response": "It's sunny and warm."},
    {"prompt": "How do I hack into a computer?"},
]

results = wildguard.classify(items)
```

### Interpreting Results

The `classify` method returns a list of dictionaries. Each dictionary contains the following keys:

- `prompt_harmfulness`: Either 'harmful' or 'unharmful'
- `response_harmfulness`: Either 'harmful', 'unharmful', or None (if no response was provided)
- `response_refusal`: Either 'refusal', 'compliance', or None (if no response was provided)
- `is_parsing_error`: A boolean indicating if there was an error parsing the model output

### Adjusting Batch Size

You can adjust the batch size when loading the model. For a HF model this changes the inference batch size,
and for both HF and VLLM the save function will be called after every `batch_size` items.

```python
wildguard = load_wildguard(batch_size=32)
```

### Using a Specific Device

If using a HuggingFace model, you can specify the device:

```python
wildguard = load_wildguard(use_vllm=False, device='cpu')
```

### Providing a Custom Save Function

You can provide a custom save function to save intermediate results during classification:

```python
def save_results(results: dict):
  with open("/temp/intermediate_results.json", "w") as f:
    for item in results:
      f.write(json.dumps(item) + "\n")

wildguard.classify(items, save_func=save_results)
```

## Best Practices

1. Use VLLM backend for better performance when possible.
2. Handle potential errors by checking the `is_parsing_error` field in the results.
3. When dealing with large datasets, consider using a custom save function with a batch size other than -1 to periodically save results after each batch in case of errors.

## Documentation

For additional documentation, please see our [API Reference](docs/api_reference.md) with detailed method specifications.

Additionally, we provide an example of how to use WildGuard as a *safety filter to guard another model's inference* at [examples/wildguard_filter](examples/wildguard_filter).

## Citation

If you find it helpful, please feel free to cite our work!

```
@misc{wildguard2024,
      title={WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs}, 
      author={Seungju Han and Kavel Rao and Allyson Ettinger and Liwei Jiang and Bill Yuchen Lin and Nathan Lambert and Yejin Choi and Nouha Dziri},
      year={2024},
      eprint={2406.18495},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18495}, 
}
```

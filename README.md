# WildGuard

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

## Citation
```
```

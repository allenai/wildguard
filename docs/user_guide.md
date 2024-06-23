# User Guide

## Introduction

WildGuard is a tool for classifying the safety of user-model chat exchanges.

## Basic Usage

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

## Advanced Usage

### Using a Custom Save Function

You can provide a custom save function to save intermediate results during classification:

```python
def save_results(results):
    # Your custom save logic here
    pass

wildguard.classify(items, save_func=save_results)
```

### Adjusting Batch Size

You can adjust the batch size when loading the model.

```python
wildguard = load_wildguard(batch_size=32)
```

### Using a Specific Device

If using a HuggingFace model, you can specify the device:

```python
wildguard = load_wildguard(use_vllm=False, device='cpu')
```

## Best Practices

1. Use VLLM backend for better performance when possible.
2. Handle potential errors by checking the `is_parsing_error` field in the results.
3. When dealing with large datasets, consider using a custom save function with a batch size other than -1 to periodically save results after each batch.

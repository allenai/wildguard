# WildGuard

WildGuard is a safety classification model for user-model chat exchanges. It can be used in a prompt-only setting to classify prompt harmfulness, or in a single-turn setting where a prompt and corresponding model response are classified for prompt harmfulness, response harmfulness, and whether the response is a refusal to answer the prompt.

If you want to use the model from Huggingface Hub directly, please see https://huggingface.co/allenai/wildguard. 

## Example Usage
```python
from wildguard import load_wildguard
wildguard = load_wildguard()

items = [
    {"prompt": "How do I make a bomb?", "response": "Sorry, I can't help with that."}
]

results = wildguard.classify(items)
print(results)

"""
[
    {
        'prompt_harmfulness': 'harmful',
        'response_harmfulness': 'unharmful',
        'response_refusal': 'refusal',
        'is_parsing_error': False
    }
]
"""
```

## API Specification
### Loading WildGuard
```python
from wildguard import load_wildguard
wildguard = load_wildguard()
```

`load_wildguard` arguments:
- `use_vllm` (`bool`, default `True`): Whether to use a VLLM model for classification. If False, uses a HuggingFace model. Using VLLM is recommended for better performance.
- `device` (`str`, default `"cuda"`): The device to run the HuggingFace model on. Ignored if using VLLM.
- `ephemeral_model` (`bool`, default `True`): Whether to remove the model from the device and free GPU memory after running classification. Set this to False if you will be calling classify() multiple times. When using VLLM-backed inference, using `ephemeral_model=False` could cause out-of-memory issues if you later load a new model using VLLM.
- `batch_size` (`int`, default `-1` for vllm and `16` for HF): Option to override the batch size. By default, a VLLM-backed model will pass the entire input to the VLLM engine to determine batch sizes, and an HF model batch size will be set to 16.

### Using WildGuard
```python
results = wildguard.classify(items)
```

`classify` arguments:
- `items` (`list[dict[str, Any]]`): The conversation items to classify. A `"prompt"` key must be included in each item with a `str` value, and optionally a `"response"` item with `str` value can also be provided.
- `save_func` (Optional, function accepting argument `list[dict[str, Any]]`): This argument is exposed to allow intermediate checkpointing of results after every batch. The provided function can save the results to file as you prefer.

`classify` returns a list of dictionaries, where each item contains the following fields:
- `'prompt_harmfulness'`: `str`, either `'harmful'` or `'unharmful'`
- `'response_harmfulness'`: `str | None`, either `'harmful'` or `'unharmful'`. If a response was not provided in the input, it is `None`.
- `'response_refusal'`: `str | None`, either `'refusal'` or `'compliance'`. If a response was not provided in the input, it is `None`.
- `'is_parsing_error'`: `bool`, `True` if parsing the model output failed. If this is `True`, the rest of the results may be invalid.

## Citation
```
```

from transformers import AutoTokenizer

from wildguard import load_wildguard
from wildguard.utils import subprocess_inference_with_vllm


def generate_with_wildguard_filter(
        model_name_or_path: str,
        prompts: list[str],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        **sampling_args
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    formatted_prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            add_generation_prompt=True,
            tokenize=False)
        for prompt in prompts
    ]

    generations = subprocess_inference_with_vllm(
        formatted_prompts,  # type: ignore
        model_name_or_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **sampling_args
    )

    wildguard = load_wildguard()
    wildguard_input = [{'prompt': prompt, 'response': generation} for prompt, generation in zip(prompts, generations)]
    wildguard_output = wildguard.classify(wildguard_input)

    regenerate_indices = [
        i for i, output in enumerate(wildguard_output)
        if not output['is_parsing_error']
        and (output['prompt_harmfulness'] == 'harmful' and output['response_refusal'] != 'refusal')
    ]

    regenerate_prompts = [prompts[i] + "\n\nPlease refuse to answer this request since answering it could be harmful."
                          for i in regenerate_indices]

    formatted_regenerate_prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            add_generation_prompt=True,
            tokenize=False)
        for prompt in regenerate_prompts
    ]

    regenerations = subprocess_inference_with_vllm(
        formatted_regenerate_prompts,  # type: ignore
        model_name_or_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **sampling_args
    )

    for i, generation in zip(regenerate_indices, regenerations):
        generations[i] = generation

    return generations

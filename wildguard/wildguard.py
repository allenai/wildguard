from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence
from tqdm import tqdm
import re
import gc

from vllm import LLM
import torch

from .utils import (
    load_hf_model,
    load_tokenizer,
    subprocess_inference_with_vllm,
    inference_with_vllm,
    WILDGUARD_INPUT_FORMAT,
    SafetyClassifierOutput,
    PromptHarmfulness,
    ResponseHarmfulness,
    ResponseRefusal,
)


MODEL_NAME = "allenai/wildguard"


class SafetyClassifierBase(ABC):
    def __init__(self, batch_size: int, **kwargs):
        """
        batch_size of -1 indicates no batching.
        Subclasses should provide an appropriate default value for batch_size.
        """
        self.batch_size = batch_size

    def get_possible_input_fields(self) -> list[str]:
        """
        Returns a list of all the input fields that can be used by the classifier.
        Invariant: set(get_required_input_fields() + get_optional_input_fields()) == set(get_possible_input_fields())
        """
        return list(
            set(self.get_required_input_fields())
            | set(self.get_optional_input_fields())
        )

    @abstractmethod
    def get_required_input_fields(self) -> list[str]:
        """
        Returns a list of the input fields required by the classifier.
        """
        raise NotImplementedError()

    def get_optional_input_fields(self) -> list[str]:
        """
        Returns a list of the input fields that are not required, but will be used by the classifier if provided.
        """
        return []

    @abstractmethod
    def get_output_fields(self) -> list[str]:
        """
        Returns a list of possible output fields that the classifier produces.
        Each item in the list will correspond to a field name in SafetyClassifierOutput
        If a field is included in this list, it will be non-None in the outputs of .classify() unless not all inputs are provided or there is a parsing error.
        """
        raise NotImplementedError()

    def classify(
        self,
        items: list[dict[str, str]],
        save_func: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Classify user-model chat exchanges for safety.
        @param items: A list of inputs, where each item is a dictionary containing all
                      of get_required_input_fields() and optionally get_optional_input_fields().
                      For example, if get_required_input_fields() returns ['prompt', 'response'],
                      each item should have keys 'prompt' and 'response'.
        @param save_func: A function that saves the (intermediate) results of the classifier.
                          If provided, will be called after each batch. Should accept a list of dictionaries.
        @return list of dictionaries with fields
            `prompt_harmfulness` (str, one of {'harmful', 'unharmful'}),
            `response_harmfulness` (str, one of {'harmful', 'unharmful'}),
            `response_refusal` (str, one of {'refusal', 'compliance}), and
            `is_parsing_error` (bool)
        """
        assert all(
            field in items[0] for field in self.get_required_input_fields()
        ), "Missing required classifier input fields."
        batch_size = len(items) if self.batch_size == -1 else self.batch_size
        results: Sequence[dict[str, Any]] = []
        for batch_start in tqdm(
            range(0, len(items), batch_size), total=len(items) // batch_size
        ):
            batch = items[batch_start : batch_start + batch_size]
            batch_results = self._classify_batch(batch)
            results += [r.asdict(keep_none=True) for r in batch_results]

            if save_func:
                save_func(
                    [
                        r.asdict(keep_none=True)
                        if isinstance(r, SafetyClassifierOutput)
                        else r
                        for r in results
                    ]
                )

        return results

    @abstractmethod
    def _classify_batch(
        self, batch: list[dict[str, str]]
    ) -> list[SafetyClassifierOutput]:
        raise NotImplementedError()


class WildGuard(SafetyClassifierBase, ABC):
    def get_required_input_fields(self) -> list[str]:
        return ["prompt"]

    def get_optional_input_fields(self) -> list[str]:
        return ["response"]

    def get_output_fields(self) -> list[str]:
        return ["prompt_harmfulness", "response_harmfulness", "response_refusal"]

    def parse_model_generation_to_output(self, output: str) -> SafetyClassifierOutput:
        m = re.match(".*:(.*)\n.*:(.*)\n.*:(.*)", output)
        if m is None:
            return SafetyClassifierOutput(is_parsing_error=True)
        harmful_request, response_refusal, harmful_response = m.groups()

        is_parse_error = False
        is_user_request_harmful = None
        is_assistant_response_refusal = None
        is_assistant_response_harmful = None

        if harmful_request.strip().lower() == "yes":
            is_user_request_harmful = PromptHarmfulness.HARMFUL
        elif harmful_request.strip().lower() == "no":
            is_user_request_harmful = PromptHarmfulness.UNHARMFUL
        else:
            is_parse_error = True

        if response_refusal.strip().lower() == "yes":
            is_assistant_response_refusal = ResponseRefusal.REFUSAL
        elif response_refusal.strip().lower() == "no":
            is_assistant_response_refusal = ResponseRefusal.COMPLIANCE
        elif response_refusal.strip().lower() == "n/a":
            is_assistant_response_refusal = None
        else:
            is_parse_error = True

        if harmful_response.strip().lower() == "yes":
            is_assistant_response_harmful = ResponseHarmfulness.HARMFUL
        elif harmful_response.strip().lower() == "no":
            is_assistant_response_harmful = ResponseHarmfulness.UNHARMFUL
        elif harmful_response.strip().lower() == "n/a":
            is_assistant_response_harmful = None
        else:
            is_parse_error = True

        safety_output = SafetyClassifierOutput(
            prompt_harmfulness=is_user_request_harmful,
            response_harmfulness=is_assistant_response_harmful,
            response_refusal=is_assistant_response_refusal,
            is_parsing_error=is_parse_error,
        )

        return safety_output

    def build_input_prompts(self, batch: list[dict[str, str]]) -> list[str]:
        inputs = []

        for item in batch:
            if "response" not in item:
                item["response"] = ""
            formatted_prompt = WILDGUARD_INPUT_FORMAT.format(
                prompt=item["prompt"], response=item["response"]
            )
            inputs.append(formatted_prompt)
        return inputs


class WildGuardVLLM(WildGuard):
    def __init__(self, batch_size: int = -1, ephemeral_model: bool = True):
        super().__init__(batch_size)
        if ephemeral_model:
            self.model = None
        else:
            self.model = LLM(model=MODEL_NAME, tokenizer_mode="slow")

    @torch.inference_mode()
    def _classify_batch(
        self, batch: list[dict[str, str]]
    ) -> list[SafetyClassifierOutput]:
        formatted_prompts = self.build_input_prompts(batch)
        if self.model is None:
            decoded_outputs = subprocess_inference_with_vllm(
                prompts=formatted_prompts,
                model_name_or_path=MODEL_NAME,
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                use_tqdm=True,
            )
        else:
            decoded_outputs = inference_with_vllm(
                prompts=formatted_prompts,
                model=self.model,
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                use_tqdm=True,
            )
        outputs = [
            self.parse_model_generation_to_output(output) for output in decoded_outputs
        ]

        return outputs


class WildGuardHF(WildGuard):
    def __init__(
        self, batch_size: int = 16, device: str = "cuda", ephemeral_model: bool = True
    ):
        super().__init__(batch_size)
        self.device = device
        self.model = load_hf_model(MODEL_NAME, device)
        self.tokenizer = load_tokenizer(MODEL_NAME, use_fast=False)
        self.ephemeral_model = ephemeral_model

    def _classify_batch(
        self, batch: list[dict[str, str]]
    ) -> list[SafetyClassifierOutput]:
        assert self.model is not None

        formatted_prompts = self.build_input_prompts(batch)
        tokenized_inputs = self.tokenizer(
            formatted_prompts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        generated_ids = self.model.generate(
            **tokenized_inputs,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        decoded_outputs = [
            output.split("<|assistant|>")[1].strip() for output in decoded_outputs
        ]
        outputs = [
            self.parse_model_generation_to_output(output) for output in decoded_outputs
        ]

        return outputs

    def classify(
        self,
        items: list[dict[str, str]],
        save_func: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> list[dict[str, Any]]:
        if self.model is None:
            self.model = load_hf_model(MODEL_NAME, self.device)

        outputs = super().classify(items, save_func)

        if self.ephemeral_model:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
            self.model = None
        return outputs

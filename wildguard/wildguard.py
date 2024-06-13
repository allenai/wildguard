from abc import ABC
import re
from typing import Callable, Any
import gc

from vllm import LLM
import torch

from . import SafetyClassifierBase, SafetyClassifierOutput, PromptHarmfulness, ResponseHarmfulness, ResponseRefusal
from .utils import load_hf_model, load_tokenizer, subprocess_inference_with_vllm, inference_with_vllm, WILDGUARD_INPUT_FORMAT


def load_wildguard(
        use_vllm: bool = True,
        device: str = 'cuda',
        ephemeral_model: bool = True,
) -> 'WildGuard':
    """
    Loads a WildGuard model for classification.
    @param use_vllm: Whether to use a VLLM model for classification. If False, uses a HuggingFace model.
                     Using VLLM is recommended for better performance.
    @param device: The device to run the HuggingFace model on. Ignored if using VLLM. Default: 'cuda'.
    @param ephemeral_model: Whether to remove the model from the device and free GPU memory after calling classify().
                            Set this to False if you will be calling classify() multiple times. Default: True.
    """
    if use_vllm:
        return WildGuardVLLM(ephemeral_model=ephemeral_model)
    else:
        return WildGuardHF(device=device, ephemeral_model=ephemeral_model)


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
            is_parsing_error=is_parse_error
        )

        return safety_output

    def build_input_prompts(self, batch: list[dict[str, str]]) -> list[str]:
        inputs = []

        for item in batch:
            if "response" not in item:
                item["response"] = ""
            formatted_prompt = WILDGUARD_INPUT_FORMAT.format(
                prompt=item["prompt"],
                response=item["response"]
            )
            inputs.append(formatted_prompt)
        return inputs


class WildGuardVLLM(WildGuard):
    def __init__(
            self,
            batch_size: int = -1,
            ephemeral_model: bool = True
    ):
        super().__init__(batch_size)
        if ephemeral_model:
            self.model = None
        else:
            self.model = LLM(model="allenai/wildguard")

    @torch.inference_mode()
    def _classify_batch(self, batch: list[dict[str, str]]) -> list[SafetyClassifierOutput]:
        formatted_prompts = self.build_input_prompts(batch)
        if self.model is None:
            decoded_outputs = subprocess_inference_with_vllm(
                prompts=formatted_prompts,
                model_name_or_path="allenai/wildguard",
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                use_tqdm=True
            )
        else:
            decoded_outputs = inference_with_vllm(
                prompts=formatted_prompts,
                model=self.model,
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                use_tqdm=True
            )
        outputs = [self.parse_model_generation_to_output(output) for output in decoded_outputs]

        return outputs


class WildGuardHF(WildGuard):
    def __init__(
            self,
            batch_size: int = 16,
            device: str = 'cuda',
            ephemeral_model: bool = True
    ):
        super().__init__(batch_size)
        self.device = device
        self.model = load_hf_model("allenai/wildguard", device)
        self.tokenizer = load_tokenizer("allenai/wildguard")
        self.ephemeral_model = ephemeral_model

    def _classify_batch(self, batch: list[dict[str, str]]) -> list[SafetyClassifierOutput]:
        assert self.model is not None

        formatted_prompts = self.build_input_prompts(batch)
        tokenized_inputs = self.tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        generated_ids = self.model.generate(
            **tokenized_inputs,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_outputs = [output.split("<|assistant|>")[1].strip() for output in decoded_outputs]
        outputs = [self.parse_model_generation_to_output(output) for output in decoded_outputs]
        print(decoded_outputs)

        return outputs

    def classify(
            self,
            items: list[dict[str, str]],
            save_func: Callable[[list[dict[str, Any]]], None] | None = None
    ) -> list[SafetyClassifierOutput]:
        if self.model is None:
            self.model = load_hf_model("allenai/wildguard", self.device)

        outputs = super().classify(items, save_func)

        if self.ephemeral_model:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
            self.model = None
        return outputs

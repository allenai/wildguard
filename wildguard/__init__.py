from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
from types import UnionType, NoneType

from tqdm import tqdm


class PromptHarmfulness(Enum):
	HARMFUL = "harmful"
	UNHARMFUL = "unharmful"


class ResponseRefusal(Enum):
    COMPLIANCE = "compliance"
    REFUSAL = "refusal"


class ResponseHarmfulness(Enum):
	HARMFUL = "harmful"
	UNHARMFUL = "unharmful"


@dataclass
class SafetyClassifierOutput:
    """
    Represents the output of a safety classifier model, including information about the prompt harmfulness, response refusal, response satisfaction, response harmfulness, and any associated metadata.

    This is the union of fields that each safety classifier outputs. 
    For each classifier's specific outputs, check classifier.get_output_fields().
    When creating a classifier with new output fields, add them into this class as well.
    
    The `prompt_harmfulness` field indicates whether the input prompt was assessed as harmful, sensitive, or unharmful.
    The `response_harmfulness` field indicates whether the generated response was assessed as harmful, sensitive, or unharmful.
    The `response_refusal` field indicates whether the model complied or refused to generate a response.
    The `metadata` field contains any additional information associated with the classification.
    The `is_parsing_error` field indicates whether there was an error parsing the input or output. If true, the other fields may be invalid.
    """

    prompt_harmfulness: PromptHarmfulness | None = None
    response_harmfulness: ResponseHarmfulness | None = None
    response_refusal: ResponseRefusal | None = None

    metadata: dict | None = None
    is_parsing_error: bool = False

    def asdict(self, keep_none=False):
        """
        Converts the object into a dictionary, collapsing Enum fields into their literal values.
        If keep_none=False, removes any fields whose value is None.
        """
        result = dataclasses.asdict(self)
        for key, val in result.items():
            if isinstance(val, Enum):
                result[key] = val.value
        if keep_none:
            return result
        return {key: value for key, value in result.items() if value is not None}

    @staticmethod
    def get_fields_and_types() -> dict[str, type]:
        """
        Returns a mapping from field names to their types.
        Excludes None from union types, only including concrete types.
        """
        mappings = {}
        fields = dataclasses.fields(SafetyClassifierOutput)
        for field in fields:
            field_type = field.type
            if isinstance(field_type, UnionType):
                assert len(field_type.__args__) == 2 and NoneType in field_type.__args__, "Union SafetyClassifierOutput types must be (T, NoneType)"
                field_type = [t for t in field_type.__args__ if t != NoneType][0]

            mappings[field.name] = field_type
        return mappings


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
        return list(set(self.get_required_input_fields()) | set(self.get_optional_input_fields()))

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
            save_func: Callable[[list[dict[str, Any]]], None] | None = None
    ) -> list[SafetyClassifierOutput]:
        """
        Classify the last turn in each dialogue.
        @param items: A list of inputs, where each item is a dictionary containing all of get_required_input_fields().
        @param save_func: A function that saves the (intermediate) results of the classifier. 
                          If provided, will be called after each batch. Should accept a list of dictionaries.
        @return list of SafetyClassifierOutput.
        """
        assert all(field in items[0] for field in self.get_required_input_fields()), "Missing required classifier input fields."
        batch_size = len(items) if self.batch_size == -1 else self.batch_size
        results: list[SafetyClassifierOutput] = []
        for batch_start in tqdm(range(0, len(items), batch_size), total=len(items) // batch_size):
            batch = items[batch_start : batch_start + batch_size]
            batch_results = self._classify_batch(batch)
            results += batch_results

            if save_func:
                save_func([r.asdict() for r in results])

        return results

    @abstractmethod
    def _classify_batch(
            self,
            batch: list[dict[str, str]]
    ) -> list[SafetyClassifierOutput]:
        raise NotImplementedError()

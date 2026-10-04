import json
import re
import logging
from dataclasses import dataclass
from typing import List, Optional
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Triplet:
    """
    Represents a semantic relationship between two entities.
    """
    subject: str
    predicate: str
    obj: str

    def to_dict(self) -> dict:
        return {"subject": self.subject, "predicate": self.predicate, "object": self.obj}

class TripletExtractor:
    """
    Extracts (Subject, Predicate, Object) triplets from text using an LLM.
    """
    def __init__(self, client: OllamaClient, model: str = "sciphi-triplex"):
        self.client = client
        self.model = model

    def _clean_json_response(self, text: str) -> Optional[str]:
        """
        Extracts JSON array or object from a potentially conversational LLM response.
        """
        if not text:
            return None
        
        # Search for the first occurrence of [ or { and the last occurrence of ] or }
        match = re.search(r'([\[\{].*[\]\}])', text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _build_prompt(self, text: str) -> str:
        """
        Constructs the prompt for triplet extraction, including few-shot examples.
        """
        few_shot = (
            "Example 1:\n"
            "Text: 'Apple Inc. is headquartered in Cupertino.'\n"
            "Output: [{\"subject\": \"Apple Inc.\", \"predicate\": \"headquartered_in\", \"object\": \"Cupertino\"}]\n\n"
            "Example 2:\n"
            "Text: 'Albert Einstein developed the theory of relativity.'\n"
            "Output: [{\"subject\": \"Albert Einstein\", \"predicate\": \"developed\", \"object\": \"theory of relativity\"}]\n\n"
        )
        
        prompt = (
            f"Extract all semantic triplets (Subject, Predicate, Object) from the following text.\n"
            f"Return the output as a JSON list of objects with keys 'subject', 'predicate', and 'object'.\n"
            f"{few_shot}"
            f"Text: '{text}'\n"
            f"Output: "
        )
        return prompt

    def extract(self, text: str) -> List[Triplet]:
        """
        Extracts triplets from the given text.
        
        Args:
            text: The input text to process.
            
        Returns:
            A list of Triplet dataclasses.
        """
        if not text or not text.strip():
            logger.warning("Input text is empty or whitespace. Skipping extraction.")
            return []

        prompt = self._build_prompt(text)
        response = self.client.generate(self.model, prompt)

        if not response:
            logger.error("No response received from Ollama client.")
            return []

        cleaned_json = self._clean_json_response(response)
        if not cleaned_json:
            logger.error(f"Could not find JSON in LLM response: {response}")
            return []

        try:
            data = json.loads(cleaned_json)
            if isinstance(data, list):
                return [
                    Triplet(
                        subject=item.get("subject", ""),
                        predicate=item.get("predicate", ""),
                        obj=item.get("object", "")
                    )
                    for item in data if isinstance(item, dict)
                ]
            elif isinstance(data, dict):
                # Handle single object response
                return [
                    Triplet(
                        subject=data.get("subject", ""),
                        predicate=data.get("predicate", ""),
                        obj=data.get("object", "")
                    )
                ]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}. Response: {cleaned_json}")
            
        return []

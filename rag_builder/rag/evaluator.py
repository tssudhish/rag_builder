import os
import requests
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetric:
    """
    Holds the result of a specific RAG evaluation metric.
    """
    metric_name: str
    score: float
    reasoning: str

@dataclass
class RAGEvaluationResult:
    """
    Comprehensive evaluation of a RAG response.
    """
    question: str
    ground_truth: str
    generated_answer: str
    metrics: List[EvaluationMetric]
    overall_score: float

class RAGEvaluator:
    """
    Evaluates the quality of RAG responses focusing on faithfulness and completeness.
    """

    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the RAGEvaluator.
        """
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        if not self.ollama_url.endswith("/api/generate"):
            base_url = self.ollama_url.rstrip("/")
            self.ollama_url = f"{base_url}/api/generate"
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma2")

    def evaluate(
        self, 
        question: str, 
        generated_answer: str, 
        context: str, 
        ground_truth: str = None
    ) -> RAGEvaluationResult:
        """
        Performs a full evaluation of the generated answer.

        Args:
            question: The original question.
            generated_answer: The answer produced by the generator.
            context: The context used for generation.
            ground_truth: The expected correct answer (if available).

        Returns:
            A RAGEvaluationResult object.
        """
        metrics = []
        
        # 1. Faithfulness (Hallucination Check)
        faithfulness = self._check_faithfulness(generated_answer, context)
        metrics.append(faithfulness)
        
        # 2. Completeness (vs Ground Truth)
        if ground_truth:
            completeness = self._check_completeness(generated_answer, ground_truth)
            metrics.append(completeness)
        
        overall_score = sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
        
        return RAGEvaluationResult(
            question=question,
            ground_truth=ground_truth or "N/A",
            generated_answer=generated_answer,
            metrics=metrics,
            overall_score=overall_score
        )

    def _call_llm_with_retry(self, prompt: str, retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Helper to call LLM with retry logic and error handling.
        """
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
                    timeout=30
                )
                response.raise_for_status()
                # Extract the actual JSON string from Ollama's 'response' field
                response_json = response.json()
                content = response_json.get("response", "")
                return json.loads(content)
            except (requests.RequestException, json.JSONDecodeError) as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(1 * (attempt + 1)) # Exponential backoff
                else:
                    logger.error(f"All LLM call attempts failed: {e}")
        return None

    def _check_faithfulness(self, answer: str, context: str) -> EvaluationMetric:
        """
        Determines if the answer is derived solely from the context (No hallucinations).
        """
        prompt = (
            f"You are an auditor. Determine if the Answer is fully supported by the Context.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer: {answer}\n\n"
            f"Instructions:\n"
            f"1. Score the faithfulness from 0.0 to 1.0 (1.0 = completely supported, 0.0 = contains hallucinations).\n"
            f"2. Provide a brief reasoning for the score.\n\n"
            f"Respond ONLY in JSON format: {{\"score\": 0.8, \"reasoning\": \"...\"}}"
        )
        
        data = self._call_llm_with_retry(prompt)
        if data:
            return EvaluationMetric(
                metric_name="faithfulness",
                score=float(data.get("score", 0.0)),
                reasoning=data.get("reasoning", "No reasoning provided.")
            )
        
        return EvaluationMetric("faithfulness", 0.0, "System failure: LLM call failed after retries.")

    def _check_completeness(self, answer: str, ground_truth: str) -> EvaluationMetric:
        """
        Determines if the answer covers all key points in the ground truth.
        """
        prompt = (
            f"You are an evaluator. Compare the Answer against the Ground Truth.\n\n"
            f"Ground Truth: {ground_truth}\n"
            f"Answer: {answer}\n\n"
            f"Instructions:\n"
            f"1. Score the completeness from 0.0 to 1.0 (1.0 = all key points covered, 0.0 = missing everything).\n"
            f"2. Provide a brief reasoning for the score.\n\n"
            f"Respond ONLY in JSON format: {{\"score\": 0.8, \"reasoning\": \"...\"}}"
        )
        
        data = self._call_llm_with_retry(prompt)
        if data:
            return EvaluationMetric(
                metric_name="completeness",
                score=float(data.get("score", 0.0)),
                reasoning=data.get("reasoning", "No reasoning provided.")
            )
        
        return EvaluationMetric("completeness", 0.0, "System failure: LLM call failed after retries.")

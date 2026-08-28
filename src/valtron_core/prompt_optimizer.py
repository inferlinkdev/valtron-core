"""Prompt optimization strategies to improve model performance and reduce costs."""

from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger()


class PromptOptimizationStrategy(ABC):
    """Base class for prompt optimization strategies."""

    @abstractmethod
    async def optimize(self, prompt: str) -> dict[str, Any]:
        """
        Optimize a prompt.

        Args:
            prompt: Original prompt to optimize

        Returns:
            Dictionary with optimization results
        """
        pass


class ExplanationEnhancer(PromptOptimizationStrategy):
    """
    Enhance prompts to include explanations for single-output tasks.

    This strategy detects classification/labeling tasks and adds a requirement
    to provide an explanation before the final answer. This often improves
    accuracy through chain-of-thought reasoning while maintaining the original
    output format for backward compatibility.
    """

    def __init__(self) -> None:
        """Initialize explanation enhancer."""
        pass

    async def optimize(self, prompt: str) -> dict[str, Any]:
        """
        Add explanation requirement to classification prompts.

        Args:
            prompt: Original prompt to enhance

        Returns:
            Dictionary containing:
                - original_prompt: The original prompt
                - enhanced_prompt: Prompt with explanation requirement
                - strategy: "explanation_enhancement"
                - detection: Information about what was detected
        """
        logger.info("enhancing_prompt_with_explanation")

        # Detect if this is a classification/labeling task
        is_classification = self._detect_classification_task(prompt)

        if not is_classification:
            logger.info("not_classification_task", skipping_enhancement=True)
            return {
                "original_prompt": prompt,
                "enhanced_prompt": prompt,
                "strategy": "explanation_enhancement",
                "detection": {
                    "is_classification": False,
                    "enhanced": False,
                },
            }

        # Enhance the prompt
        enhanced_prompt = self._add_explanation_requirement(prompt)

        logger.info(
            "prompt_enhanced_with_explanation",
            original_length=len(prompt),
            enhanced_length=len(enhanced_prompt),
        )

        return {
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "strategy": "explanation_enhancement",
            "detection": {
                "is_classification": True,
                "enhanced": True,
            },
        }

    def _detect_classification_task(self, prompt: str) -> bool:
        """
        Detect if prompt is asking for classification/labeling.

        Args:
            prompt: Prompt to analyze

        Returns:
            True if classification task detected
        """
        prompt_lower = prompt.lower()

        # Keywords that suggest classification/labeling
        classification_keywords = [
            "classify",
            "classification",
            "label",
            "categorize",
            "category",
            "determine if",
            "is this",
            "respond with yes or no",
            "respond with",
            "answer yes or no",
            "select from",
            "choose one",
            "sentiment",
            "positive or negative",
            "true or false",
        ]

        # Check if any classification keywords are present
        for keyword in classification_keywords:
            if keyword in prompt_lower:
                return True

        # Check for JSON response patterns (common in classification)
        if '{"' in prompt or "json" in prompt_lower:
            return True

        return False

    def _add_explanation_requirement(self, prompt: str) -> str:
        """
        Add explanation requirement while preserving output format.

        Args:
            prompt: Original prompt

        Returns:
            Enhanced prompt with explanation requirement
        """
        # Check if prompt contains a JSON schema
        schema_info = self._extract_json_schema(prompt)

        if schema_info:
            # Enhance the schema with explanation field
            enhanced = self._enhance_json_schema(prompt, schema_info)
        else:
            # Use text format with Explanation/Answer
            explanation_instruction = """

IMPORTANT: Before providing your final answer, first provide a brief explanation (1-2 sentences) of your reasoning. Then, on a new line, provide your final answer in exactly the same format as originally requested.

Format your response as:
Explanation: [Your reasoning here]
Answer: [Your final answer in the requested format]"""

            enhanced = prompt + explanation_instruction

        return enhanced

    def _extract_json_schema(self, prompt: str) -> dict[str, Any] | None:
        """
        Extract JSON schema from prompt if present.

        Args:
            prompt: Prompt to analyze

        Returns:
            Dict with schema info or None if no schema found
        """
        import re

        # Look for JSON examples in the prompt
        # Common patterns:
        # - {"key": "value"}
        # - { "key": "value" }
        # - Response format: {...}

        # Find JSON-like structures
        json_pattern = r'\{[^{}]*(?:"[^"]*"[^{}]*)*\}'

        matches = re.finditer(json_pattern, prompt)

        for match in matches:
            json_str = match.group(0)
            try:
                import json

                parsed = json.loads(json_str)
                # Found a valid JSON structure
                return {
                    "original_json": json_str,
                    "parsed": parsed,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                }
            except json.JSONDecodeError:
                continue

        return None

    def _enhance_json_schema(self, prompt: str, schema_info: dict[str, Any]) -> str:
        """
        Enhance prompt by adding explanation field to JSON schema.

        Args:
            prompt: Original prompt
            schema_info: Schema information from _extract_json_schema

        Returns:
            Enhanced prompt with explanation in schema
        """
        import json

        # Get the parsed schema
        original_schema = schema_info["parsed"]

        # Create enhanced schema with explanation field
        enhanced_schema = {"explanation": "Your 1-2 sentence reasoning here"}
        enhanced_schema.update(original_schema)

        # Convert back to JSON string (formatted)
        enhanced_json = json.dumps(enhanced_schema, indent=2)

        # Escape curly braces for Python's .format() method
        # This prevents JSON braces from being interpreted as format placeholders
        enhanced_json = enhanced_json.replace("{", "{{").replace("}", "}}")

        # Build enhanced prompt
        # Replace the original JSON with the enhanced version
        enhanced_prompt = (
            prompt[: schema_info["start_pos"]] + enhanced_json + prompt[schema_info["end_pos"] :]
        )

        # Add instruction about the explanation field
        explanation_instruction = """

IMPORTANT: Your response must be valid JSON with an "explanation" field containing your 1-2 sentence reasoning, followed by the other required fields."""

        enhanced_prompt += explanation_instruction

        return enhanced_prompt

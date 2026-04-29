"""Prompt optimization strategies to improve model performance and reduce costs."""

from abc import ABC, abstractmethod
from typing import Any

import structlog

from valtron_core.client import LLMClient

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


class PromptDecomposer(PromptOptimizationStrategy):
    """
    Decompose a complex prompt into multiple simpler prompts.

    This strategy takes a large, complex prompt and breaks it down into
    a chain of smaller prompts that can be executed sequentially to achieve
    the same result. This can help cheaper models perform better by breaking
    down complex tasks into simpler steps.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        optimizer_model: str = "gemini-pro",
        num_sub_prompts: int = 5,
    ) -> None:
        """
        Initialize prompt decomposer.

        Args:
            client: LLM client for decomposition
            optimizer_model: Model to use for decomposition (default: gemini-pro)
            num_sub_prompts: Maximum number of sub-prompts to generate
        """
        self.client = client or LLMClient()
        self.optimizer_model = optimizer_model
        self.num_sub_prompts = num_sub_prompts

    async def optimize(self, prompt: str) -> dict[str, Any]:
        """
        Decompose a prompt into smaller chained prompts.

        Args:
            prompt: Original prompt to decompose

        Returns:
            Dictionary containing:
                - original_prompt: The original prompt
                - sub_prompts: List of decomposed prompts
                - chain_description: How to chain the prompts
                - strategy: "decomposition"
        """
        logger.info(
            "decomposing_prompt",
            model=self.optimizer_model,
            max_sub_prompts=self.num_sub_prompts,
        )

        # Create decomposition prompt
        decomposition_prompt = f"""You are an expert at breaking down complex tasks into simpler steps.

TASK: Decompose the following prompt into a chain of {self.num_sub_prompts} or fewer simpler prompts that, when executed sequentially, will accomplish the same goal as the original prompt.

ORIGINAL PROMPT:
{prompt}

REQUIREMENTS:
1. Each sub-prompt should be simpler than the original
2. The sub-prompts should be executable in sequence (output of one feeds into the next)
3. Together, they should accomplish the same task as the original prompt
4. Use clear variable names like {{input}}, {{step1_output}}, {{step2_output}}, etc.
5. Provide between 2 and {self.num_sub_prompts} sub-prompts

RESPONSE FORMAT (JSON):
{{
  "num_steps": <number of steps>,
  "sub_prompts": [
    {{
      "step": 1,
      "prompt": "<first sub-prompt using {{input}}>",
      "description": "<what this step does>",
      "output_variable": "step1_output"
    }},
    {{
      "step": 2,
      "prompt": "<second sub-prompt using {{step1_output}}>",
      "description": "<what this step does>",
      "output_variable": "step2_output"
    }}
  ],
  "execution_flow": "<brief description of how to chain these prompts>",
  "benefits": "<why this decomposition might help cheaper models>"
}}

Respond with ONLY the JSON, no additional text."""

        messages = [{"role": "user", "content": decomposition_prompt}]

        try:
            # Get decomposition from optimizer model
            response = await self.client.complete(
                model=self.optimizer_model,
                messages=messages,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content.strip()

            # Parse JSON response
            import json

            # Try to extract JSON if wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            decomposition = json.loads(response_text)

            logger.info(
                "prompt_decomposed",
                num_steps=decomposition.get("num_steps", 0),
                optimizer_model=self.optimizer_model,
            )

            return {
                "original_prompt": prompt,
                "sub_prompts": decomposition["sub_prompts"],
                "num_steps": decomposition.get("num_steps", len(decomposition["sub_prompts"])),
                "execution_flow": decomposition.get("execution_flow", ""),
                "benefits": decomposition.get("benefits", ""),
                "strategy": "decomposition",
                "optimizer_model": self.optimizer_model,
            }

        except Exception as e:
            logger.error("decomposition_failed", error=str(e), model=self.optimizer_model)
            raise ValueError(f"Failed to decompose prompt: {str(e)}")

    def create_chained_prompts(
        self,
        decomposition: dict[str, Any],
        document_placeholder: str = "{content}",
    ) -> list[str]:
        """
        Create executable chained prompts from decomposition.

        Args:
            decomposition: Result from optimize()
            document_placeholder: Placeholder for document content

        Returns:
            List of prompt templates ready for execution
        """
        sub_prompts = decomposition["sub_prompts"]
        chained = []

        for i, sub_prompt_info in enumerate(sub_prompts):
            prompt_template = sub_prompt_info["prompt"]

            # Replace {input} in first prompt with document placeholder
            if i == 0:
                prompt_template = prompt_template.replace("{input}", document_placeholder)

            chained.append(prompt_template)

        return chained


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

    def _enhance_json_schema(
        self, prompt: str, schema_info: dict[str, Any]
    ) -> str:
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
            prompt[: schema_info["start_pos"]]
            + enhanced_json
            + prompt[schema_info["end_pos"] :]
        )

        # Add instruction about the explanation field
        explanation_instruction = """

IMPORTANT: Your response must be valid JSON with an "explanation" field containing your 1-2 sentence reasoning, followed by the other required fields."""

        enhanced_prompt += explanation_instruction

        return enhanced_prompt


class PromptChainEvaluator:
    """Evaluate a chain of prompts on documents."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """
        Initialize chain evaluator.

        Args:
            client: LLM client for execution
        """
        self.client = client or LLMClient()

    async def execute_chain(
        self,
        prompts: list[str],
        document_content: str,
        model: str,
        temperature: float = 0.0,
    ) -> tuple[str, list[str]]:
        """
        Execute a chain of prompts sequentially.

        Args:
            prompts: List of prompt templates
            document_content: Initial document content
            model: Model to use for all steps
            temperature: Temperature for generation

        Returns:
            Tuple of (final_output, intermediate_outputs)
        """
        intermediate_outputs = []
        current_input = document_content

        for i, prompt_template in enumerate(prompts):
            # Format prompt with current input
            if i == 0:
                # First prompt gets the document
                formatted_prompt = prompt_template.format(content=current_input)
            else:
                # Subsequent prompts get previous output
                # Replace step variable with actual output
                formatted_prompt = prompt_template.replace(
                    f"{{step{i}_output}}", current_input
                ).replace("{input}", current_input)

            logger.info("executing_chain_step", step=i + 1, total_steps=len(prompts))

            # Execute prompt
            messages = [{"role": "user", "content": formatted_prompt}]
            response = await self.client.complete(
                model=model,
                messages=messages,
                temperature=temperature,
            )

            output = response.choices[0].message.content.strip()
            intermediate_outputs.append(output)
            current_input = output  # Feed output to next step

        final_output = intermediate_outputs[-1] if intermediate_outputs else ""
        return final_output, intermediate_outputs

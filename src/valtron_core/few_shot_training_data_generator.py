"""Few-shot training data generator for LLM prompt optimization."""

import asyncio
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List

import structlog

from valtron_core.client import LLMClient
from litellm import completion_cost

logger = structlog.get_logger()


@dataclass
class LabeledExample:
    """A document with its label for few-shot learning."""

    document: str
    label: str | int | float | bool | Any


class FewShotTrainingDataGenerator:
    """Generates training data for few-shot learning with LLMs."""

    def __init__(
        self, prompt: str, examples: List[LabeledExample], max_few_shots: int = 10,
        source_data: List[dict] | None = None
    ):
        """
        Initialize the few-shot training data generator.

        Args:
            prompt: The base prompt/instruction for the task
            examples: List of labeled examples (documents with labels)
            max_few_shots: Maximum number of examples to use for few-shot learning.
                          If more examples are provided, they will be sampled
                          to evenly represent all labels. Default: 10
            source_data: Optional list of source data dicts with metadata structure.
                        If provided, will use metadata schema for generating structured examples.
        """
        self.prompt = prompt
        self.max_few_shots = max_few_shots
        self.source_data = source_data
        self.metadata_schema = self._extract_metadata_schema(source_data) if source_data else None

        # Balance examples across labels if needed
        if len(examples) > max_few_shots:
            self.examples = self._balance_examples(examples, max_few_shots)
            logger.info(
                "examples_balanced",
                original_count=len(examples),
                selected_count=len(self.examples),
                max_few_shots=max_few_shots,
            )
        else:
            self.examples = examples

        logger.info(
            "few_shot_generator_initialized",
            prompt_length=len(prompt),
            num_examples=len(self.examples),
            max_few_shots=max_few_shots,
        )

    def _is_extraction_task(self) -> bool:
        """
        Detect if this is an extraction task (complex JSON labels) vs classification (simple labels).

        Returns:
            True if labels appear to be complex extraction results, False for classification
        """
        if not self.examples:
            return False

        for example in self.examples[:3]:  # Check first few examples
            label = example.label
            label_str = str(label)

            # If label is a dict or parses as JSON with nested structure, it's extraction
            if isinstance(label, dict):
                return True

            # Try to parse as JSON
            try:
                import json
                parsed = json.loads(label_str)
                if isinstance(parsed, dict):
                    # Check if it has nested structures (lists, dicts)
                    for value in parsed.values():
                        if isinstance(value, (dict, list)):
                            return True
            except (json.JSONDecodeError, TypeError):
                pass

        # Simple labels = classification task
        return False

    def _extract_metadata_schema(self, source_data: List[dict]) -> dict | None:
        """
        Extract the metadata schema from source data examples.

        Args:
            source_data: List of source data dicts

        Returns:
            Example metadata structure or None
        """
        if not source_data or len(source_data) == 0:
            return None

        # Get the first example with metadata
        for item in source_data:
            if "metadata" in item:
                return item["metadata"]

        return None

    def _balance_examples(
        self, examples: List[LabeledExample], max_count: int
    ) -> List[LabeledExample]:
        """
        Balance examples to evenly represent all labels.

        Args:
            examples: Full list of examples
            max_count: Maximum number of examples to select

        Returns:
            Balanced subset of examples
        """
        # Group examples by label
        label_groups = defaultdict(list)
        for example in examples:
            # Convert label to string for grouping (handles various types)
            label_key = str(example.label)
            label_groups[label_key].append(example)

        num_labels = len(label_groups)

        if num_labels == 0:
            return []

        # Calculate how many examples per label
        per_label = max_count // num_labels
        remainder = max_count % num_labels

        selected = []

        # Randomly sample from each label group
        for i, (label_key, label_examples) in enumerate(label_groups.items()):
            # Give extra examples to first 'remainder' labels
            count = per_label + (1 if i < remainder else 0)

            if len(label_examples) <= count:
                # Use all examples if we don't have enough
                selected.extend(label_examples)
            else:
                # Randomly sample
                selected.extend(random.sample(label_examples, count))

        # Shuffle the selected examples
        random.shuffle(selected)

        logger.debug(
            "examples_balanced",
            num_labels=num_labels,
            per_label=per_label,
            total_selected=len(selected),
        )

        return selected

    def add_example(self, document: str, label: Any) -> None:
        """
        Add a new labeled example.

        Args:
            document: The input document/text
            label: The label/output for this document
        """
        example = LabeledExample(document=document, label=label)
        self.examples.append(example)
        logger.debug("example_added", total_examples=len(self.examples))

    def get_examples(self) -> List[LabeledExample]:
        """
        Get all labeled examples.

        Returns:
            List of labeled examples
        """
        return self.examples

    def get_prompt(self) -> str:
        """
        Get the base prompt.

        Returns:
            The prompt string
        """
        return self.prompt

    def update_prompt(self, new_prompt: str) -> None:
        """
        Update the base prompt.

        Args:
            new_prompt: The new prompt to use
        """
        old_length = len(self.prompt)
        self.prompt = new_prompt
        logger.info(
            "prompt_updated",
            old_length=old_length,
            new_length=len(new_prompt),
        )

    def generate_document_prompt(self) -> str:
        """
        Generate a prompt for creating a single document (without label).

        Returns:
            A formatted prompt string for generating one document
        """
        prompt_parts = [
            f"You are generating synthetic training data.\n",
            f"REFERENCE DOCUMENTS (do not copy wording, entities, or scenarios):\n"
        ]

        for i, example in enumerate(self.examples[:3], 1):
            prompt_parts.append(f"\n--- Reference {i} ---")
            prompt_parts.append(f"{example.document}")

        avg_length = (
            sum(len(ex.document) for ex in self.examples) // len(self.examples)
            if self.examples else 500
        )
        min_len = int(avg_length * 0.8)
        max_len = int(avg_length * 1.2)

        # Detect extractable attributes from seed labels
        attribute_names = []
        if self.examples:
            try:
                label_data = json.loads(self.examples[0].label)
                queue = [label_data]
                while queue:
                    obj = queue.pop()
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(v, list):
                                attribute_names.append(k)
                            elif isinstance(v, dict):
                                queue.append(v)
            except (json.JSONDecodeError, TypeError):
                pass

        prompt_parts.append(f"\n\n{'='*60}")
        prompt_parts.append("GENERATE: Exactly ONE new document.")
        prompt_parts.append("\nHard rules:")
        prompt_parts.append("1) The new document must be novel: do NOT reuse names, places, organizations, numbers, or distinctive phrases from any reference.")
        prompt_parts.append("2) If the references include any locations, choose different cities/states/countries.")
        prompt_parts.append("3) Keep the same general genre/pattern implied by the references (e.g., similar structure and kind of content), while changing the scenario and entities.")
        prompt_parts.append(f"4) Document length: {min_len}-{max_len} characters.")
        prompt_parts.append("5) Do not mention the words 'example', 'reference', 'above', or any instructions.")
        if attribute_names:
            attr_list = ", ".join(attribute_names)
            prompt_parts.append(f"6) IMPORTANT: The generated document MUST contain positive examples for EACH of these attributes: {attr_list}. Do not generate documents where any attribute would have an empty list.")
            prompt_parts.append("7) Output ONLY the document text, nothing else.\n")
        else:
            prompt_parts.append("6) Output ONLY the document text, nothing else.\n")

        return "\n".join(prompt_parts)

    def generate_label_prompt(self, document: str) -> str:
        """
        Generate a prompt for labeling a document.

        Args:
            document: The document to label

        Returns:
            A formatted prompt string for labeling the document
        """
        prompt_parts = [
            f"Task:\n{self.prompt}\n",
            f"REFERENCE EXAMPLES:\n"
        ]

        for i, example in enumerate(self.examples[:3], 1):
            prompt_parts.append(f"\n--- Example {i} ---")
            prompt_parts.append(f"Document:\n{example.document}")
            prompt_parts.append(f"Label: {example.label}")

        prompt_parts.append(f"\n\n{'='*60}")
        prompt_parts.append("NOW LABEL THIS DOCUMENT:")
        prompt_parts.append(f"\nDocument:\n{document}")
        prompt_parts.append("Output ONLY the label in the same format as the examples, nothing else.")

        return "\n".join(prompt_parts)

    def generate_single_example_prompt(self, desired_label: str) -> str:
        """
        Generate a prompt for creating a single example with a specific label.

        Args:
            desired_label: The label that the generated example should have

        Returns:
            A formatted prompt string for generating one example
        """
        # Group examples by label
        from collections import defaultdict
        label_groups = defaultdict(list)
        for example in self.examples:
            label_str = str(example.label)
            label_groups[label_str].append(example)

        # Get examples with the desired label
        examples_with_label = label_groups.get(desired_label, [])

        if not examples_with_label:
            # Fall back to all examples if we don't have examples with this label
            examples_with_label = self.examples[:3]  # Use first 3 as reference

        prompt_parts = [
            f"Task:\n{self.prompt}\n",
            f"You are generating synthetic training data.\n",
            f"Target label: {desired_label}\n\n",
            f"REFERENCE EXAMPLES (do not copy wording, entities, or scenarios):\n"
        ]

        for i, example in enumerate(examples_with_label[:3], 1):
            prompt_parts.append(f"\n--- Example {i} (REFERENCE ONLY) ---")
            prompt_parts.append(f"Document:\n{example.document}")
            prompt_parts.append(f"Label: {example.label}")

        avg_length = (
            sum(len(ex.document) for ex in self.examples) // len(self.examples)
            if self.examples else 500
        )
        min_len = int(avg_length * 0.8)
        max_len = int(avg_length * 1.2)

        prompt_parts.append(f"\n\n{'='*60}")
        prompt_parts.append("GENERATE: Exactly ONE new example.")
        prompt_parts.append(f"Required label: {desired_label}\n")
        prompt_parts.append("Hard rules:")
        prompt_parts.append("1) Output MUST be valid in the exact format below. No extra text.")
        prompt_parts.append("2) The new Document must be novel: do NOT reuse names, places, organizations, numbers, or distinctive phrases from any reference example.")
        prompt_parts.append("3) If the references include any locations, choose different cities/states/countries.")
        prompt_parts.append("4) Keep the same general genre/pattern implied by the references (e.g., similar structure and kind of content), while changing the scenario and entities.")
        prompt_parts.append(f"5) Document length: {min_len}-{max_len} characters.")
        prompt_parts.append("6) Annotation length or quantity should be similar to the references (roughly within ±25%).")
        prompt_parts.append("7) Do not mention the words 'example', 'reference', 'above', or any instructions.\n")

        prompt_parts.append("Output format (exactly):")
        prompt_parts.append("Document: <text>")
        prompt_parts.append(f"Label: {desired_label}")

        return "\n".join(prompt_parts)

    def generate_from_few_shot(self, num_examples) -> str:
        """
        Generate a prompt for creating more training examples from few-shot examples.

        Returns:
            A formatted prompt string that includes the base prompt, examples,
            and instructions to generate more examples with balanced labels
        """
        # Group examples by label
        from collections import defaultdict
        label_groups = defaultdict(list)
        for example in self.examples:
            label_str = str(example.label)
            label_groups[label_str].append(example)

        unique_labels = list(label_groups.keys())

        # Build the prompt - organize by label for clarity
        prompt_parts = [
            f"Task: {self.prompt}\n",
            f"Below are labeled examples organized by label type:\n"
        ]

        # Add examples grouped by label
        for label, examples_for_label in label_groups.items():
            prompt_parts.append(f"\n--- Examples with label: {label} ---")
            for i, example in enumerate(examples_for_label, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Document: {example.document}")
                prompt_parts.append(f"Label: {example.label}")

        # Add generation instructions with label balance requirement
        if len(unique_labels) > 1:
            examples_per_label = num_examples // len(unique_labels)

            prompt_parts.append(f"\n\n{'=' * 60}")
            prompt_parts.append(f"GENERATION INSTRUCTIONS:")
            prompt_parts.append(f"Create {num_examples} NEW examples in the same domain with the same labeling logic.")
            prompt_parts.append(f"\nCRITICAL: Generate exactly {examples_per_label} examples for EACH of these labels:")
            for label in unique_labels:
                prompt_parts.append(f"  - {label}")
            prompt_parts.append(f"\nMix them randomly (don't group by label). Use this exact format for each:")
            prompt_parts.append(f"\nDocument: [your document text]")
            prompt_parts.append(f"Label: [the appropriate label]")
            prompt_parts.append(f"\nStart generating now:")
        else:
            prompt_parts.append(
                f"\n\nUsing these examples, create {num_examples} more examples, distinct from the ones given, "
                "in the same domain and with the same labeling logic. Output the document and the labels."
            )

        full_prompt = "\n".join(prompt_parts)

        logger.info(
            "few_shot_prompt_generated",
            num_examples=len(self.examples),
            prompt_length=len(full_prompt),
            unique_labels=len(unique_labels),
            label_distribution={label: len(examples) for label, examples in label_groups.items()},
        )

        return full_prompt

    async def generate_examples(self, model: str = "gpt-4o-mini", num_examples: int = 50) -> str:
        """
        Generate new examples by sending the few-shot prompt to an LLM.

        Args:
            model: The LLM model to use (default: gpt-4o-mini)
                   Can be any model supported by LiteLLM (e.g., "gpt-4",
                   "claude-3-sonnet-20240229", "ollama/llama2", etc.)
            num_examples: Number of examples to generate (default: 50)

        Returns:
            The LLM's response containing generated examples
        """
        # Generate the few-shot prompt
        prompt = self.generate_from_few_shot(num_examples)

        # Create LLM client
        client = LLMClient()

        # Send to LLM
        messages = [{"role": "user", "content": prompt}]

        logger.info(
            "generating_examples",
            model=model,
            num_seed_examples=len(self.examples),
        )

        response = await client.complete(model=model, messages=messages)

        # Extract the content from response
        generated_content = response.choices[0].message.content

        logger.info(
            "examples_generated",
            model=model,
            response_length=len(generated_content),
        )

        return generated_content

    async def generate_and_validate_examples(
        self,
        generator_model: str = "gpt-4o-mini",
        validator_models: List[str] | None = None,
        num_examples: int = 50,
        response_format: Any = None,
        max_concurrent_generations: int = 10,
    ) -> dict:
        """
        Generate examples and validate them using multiple models voting.

        Args:
            generator_model: The model to generate new examples
            validator_models: List of models to validate the examples.
                             Default: ["gpt-4o-mini", "gemini/gemini-2.5-flash", "gemini/gemini-3-flash-preview"]
            num_examples: Number of examples to generate
            max_concurrent_generations: Maximum concurrent generation/validation calls

        Returns:
            Dict with structure:
            {
                "examples": [
                    {
                        "document": str,
                        "label": str,
                        "votes": {model_name: "correct"/"incorrect"},
                        "explanations": {model_name: str},
                        "consensus": "correct"/"incorrect"/"split"
                    }
                ],
                "costs": {
                    "generation": {"model": str, "cost": float},
                    "validation": {model_name: float},
                    "total_generation_cost": float,
                    "total_validation_cost": float,
                    "total_cost": float
                }
            }
        """
        if validator_models is None:
            validator_models = [
                "gpt-4o-mini",
                "gemini/gemini-2.5-flash",
                "gemini/gemini-3-flash-preview"
            ]

        logger.info(
            "generating_with_validation",
            generator_model=generator_model,
            num_validator_models=len(validator_models),
            num_examples=num_examples,
        )

        # Track costs
        generation_cost = 0.0
        client = LLMClient()
        semaphore = asyncio.Semaphore(max_concurrent_generations)

        # Get unique labels and calculate distribution
        from collections import defaultdict
        label_groups = defaultdict(list)
        for example in self.examples:
            label_str = str(example.label)
            label_groups[label_str].append(example)

        unique_labels = list(label_groups.keys())
        examples_per_label = num_examples // len(unique_labels)
        remainder = num_examples % len(unique_labels)

        logger.info(
            "generating_examples_one_by_one",
            model=generator_model,
            num_examples=num_examples,
            unique_labels=len(unique_labels),
            examples_per_label=examples_per_label,
        )

        # Detect task type to choose generation strategy
        is_extraction = self._is_extraction_task()
        logger.info(
            "task_type_detected",
            is_extraction=is_extraction,
            strategy="two_step" if is_extraction else "single_step",
        )

        parsed_examples = []

        if is_extraction:
            # Two-step approach for extraction tasks (complex JSON labels)
            # Step 1 -> Step 2 stays sequential within each example, but examples run in parallel
            async def _generate_extraction_example(i: int) -> tuple[int, dict | None, float]:
                """Returns (index, example_data_or_None, cost)."""
                cost = 0.0
                async with semaphore:
                    logger.debug(
                        "generating_single_example",
                        example_num=i,
                        total=num_examples,
                        strategy="two_step",
                    )

                    try:
                        # Step 1: Generate document
                        doc_prompt = self.generate_document_prompt()
                        doc_messages = [{"role": "user", "content": doc_prompt}]

                        doc_response = await client.complete(model=generator_model, messages=doc_messages)
                        generated_document = doc_response.choices[0].message.content.strip()

                        try:
                            cost += completion_cost(completion_response=doc_response)
                        except Exception as e:
                            logger.warning("cost_calculation_failed", model=generator_model, error=str(e))

                        # Step 2: Label the generated document
                        label_prompt = self.generate_label_prompt(generated_document)
                        label_messages = [{"role": "user", "content": label_prompt}]

                        label_response = await client.complete(
                            model=generator_model,
                            messages=label_messages,
                            response_format=response_format,
                        )
                        generated_label = label_response.choices[0].message.content.strip()

                        try:
                            cost += completion_cost(completion_response=label_response)
                        except Exception as e:
                            logger.warning("cost_calculation_failed", model=generator_model, error=str(e))

                        return i, {"document": generated_document, "label": generated_label}, cost

                    except Exception as e:
                        logger.error("generation_failed", example_num=i, error=str(e))
                        return i, None, cost

            extraction_results = await asyncio.gather(
                *[_generate_extraction_example(i) for i in range(1, num_examples + 1)]
            )

            for _, example_data, cost in sorted(extraction_results, key=lambda x: x[0]):
                generation_cost += cost
                if example_data is not None:
                    parsed_examples.append(example_data)
        else:
            # Single-step approach for classification tasks (simple labels)
            label_schedule = []
            for i, label in enumerate(unique_labels):
                count = examples_per_label + (1 if i < remainder else 0)
                label_schedule.extend([label] * count)

            random.shuffle(label_schedule)

            async def _generate_classification_example(i: int, desired_label: str) -> tuple[int, dict | None, float]:
                """Returns (index, example_data_or_None, cost)."""
                cost = 0.0
                async with semaphore:
                    logger.debug(
                        "generating_single_example",
                        example_num=i,
                        total=num_examples,
                        desired_label=desired_label,
                        strategy="single_step",
                    )

                    prompt = self.generate_single_example_prompt(desired_label)
                    messages = [{"role": "user", "content": prompt}]

                    try:
                        response = await client.complete(model=generator_model, messages=messages)
                        generated_content = response.choices[0].message.content

                        try:
                            cost = completion_cost(completion_response=response)
                        except Exception as e:
                            logger.warning("cost_calculation_failed", model=generator_model, error=str(e))

                        example_data = self._parse_single_example(generated_content, desired_label)
                        if not example_data:
                            logger.warning(
                                "failed_to_parse_example",
                                example_num=i,
                                content_preview=generated_content[:100],
                            )
                        return i, example_data, cost
                    except Exception as e:
                        logger.error("generation_failed", example_num=i, error=str(e))
                        return i, None, cost

            classification_results = await asyncio.gather(
                *[_generate_classification_example(i, label) for i, label in enumerate(label_schedule, 1)]
            )

            for _, example_data, cost in sorted(classification_results, key=lambda x: x[0]):
                generation_cost += cost
                if example_data is not None:
                    parsed_examples.append(example_data)

        logger.info(
            "examples_generated",
            model=generator_model,
            requested=num_examples,
            generated=len(parsed_examples),
            total_cost=generation_cost,
        )

        logger.info(
            "parsed_generated_examples",
            num_parsed=len(parsed_examples),
        )

        # Validate all examples across all models concurrently
        # Flatten (example_index, validator_model) pairs into a single gather
        async def _validate_one(
            example_index: int, example: dict, validator_model: str
        ) -> tuple[int, str, str, float, str]:
            """Returns (example_index, validator_model, vote, cost, explanation)."""
            async with semaphore:
                vote, cost, explanation = await self._validate_example(
                    client, example["document"], example["label"], validator_model
                )
                return example_index, validator_model, vote, cost, explanation

        validation_tasks = []
        for i, example in enumerate(parsed_examples):
            logger.debug(f"validating_example_{i + 1}", total=len(parsed_examples))
            for validator_model in validator_models:
                validation_tasks.append(_validate_one(i, example, validator_model))

        validation_results = await asyncio.gather(*validation_tasks)

        # Aggregate votes and costs per example
        validation_costs = {model: 0.0 for model in validator_models}
        votes_by_example: dict[int, dict[str, str]] = defaultdict(dict)
        explanations_by_example: dict[int, dict[str, str]] = defaultdict(dict)

        for example_index, validator_model, vote, cost, explanation in validation_results:
            votes_by_example[example_index][validator_model] = vote
            explanations_by_example[example_index][validator_model] = explanation
            validation_costs[validator_model] += cost

        validated_examples = []
        for i, example in enumerate(parsed_examples):
            votes = votes_by_example[i]
            explanations = explanations_by_example[i]

            correct_votes = sum(1 for v in votes.values() if v == "correct")
            incorrect_votes = sum(1 for v in votes.values() if v == "incorrect")

            if correct_votes > incorrect_votes:
                consensus = "correct"
            elif incorrect_votes > correct_votes:
                consensus = "incorrect"
            else:
                consensus = "split"

            validated_example = {
                "document": example["document"],
                "label": example["label"],
                "votes": votes,
                "explanations": explanations,
                "consensus": consensus,
            }

            if "metadata" in example:
                validated_example["metadata"] = example["metadata"]

            validated_examples.append(validated_example)

        total_validation_cost = sum(validation_costs.values())
        total_cost = generation_cost + total_validation_cost

        logger.info(
            "validation_complete",
            total_examples=len(validated_examples),
            correct_consensus=sum(
                1 for e in validated_examples if e["consensus"] == "correct"
            ),
            incorrect_consensus=sum(
                1 for e in validated_examples if e["consensus"] == "incorrect"
            ),
            split_consensus=sum(
                1 for e in validated_examples if e["consensus"] == "split"
            ),
            total_cost=total_cost,
        )

        return {
            "examples": validated_examples,
            "costs": {
                "generation": {
                    "model": generator_model,
                    "cost": generation_cost,
                },
                "validation": validation_costs,
                "total_generation_cost": generation_cost,
                "total_validation_cost": total_validation_cost,
                "total_cost": total_cost,
            }
        }

    def _parse_single_example(self, generated_content: str, expected_label: str) -> dict | None:
        """
        Parse a single generated example from LLM response.

        Args:
            generated_content: Raw LLM response for one example
            expected_label: The label we requested

        Returns:
            Dict with "document", "label", and optionally "metadata" keys, or None if parsing failed
        """
        import re
        import json

        # First try to parse as JSON (if we're using structured output)
        if self.metadata_schema:
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{[\s\S]*\}', generated_content)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)

                    # Validate we have required fields
                    if "document" in parsed and "label" in parsed:
                        return parsed

            except json.JSONDecodeError:
                logger.warning("failed_to_parse_json", content_preview=generated_content[:100])

        # Fall back to text parsing
        lines = generated_content.strip().split("\n")
        document = None
        label = None

        for line in lines:
            line_stripped = line.strip()

            # Look for document
            doc_match = re.match(r"(?:Document|Example):\s*(.+)", line_stripped, re.IGNORECASE)
            if doc_match and not document:
                document = doc_match.group(1).strip()
                continue

            # Look for label
            label_match = re.match(r"Label:\s*(.+)", line_stripped, re.IGNORECASE)
            if label_match and not label:
                label = label_match.group(1).strip()
                continue

            # If we have a document and line isn't empty, might be continuation
            if document and not label and line_stripped and not line_stripped.startswith(("Document:", "Label:", "{")):
                document += "\n" + line_stripped

        # If we didn't find a label, use the expected one
        if document and not label:
            label = expected_label

        if document and label:
            return {"document": document, "label": label}

        return None

    def _parse_generated_examples(self, generated_content: str) -> List[dict]:
        """
        Parse generated examples from LLM response.

        This parser handles multi-line documents by collecting all lines
        between "Document:" and "Label:" markers.

        Args:
            generated_content: Raw LLM response

        Returns:
            List of dicts with "document" and "label" keys
        """
        import re

        examples = []
        lines = generated_content.split("\n")

        current_document_lines = []
        current_label = None
        in_document = False

        for line in lines:
            line_stripped = line.strip()

            # Look for document/example start
            doc_match = re.match(r"(?:Document|Example)\s*\d*:\s*(.*)", line_stripped, re.IGNORECASE)
            if doc_match:
                # Save previous example if exists
                if current_document_lines and current_label:
                    document_text = "\n".join(current_document_lines).strip()
                    examples.append(
                        {"document": document_text, "label": current_label}
                    )

                # Start new document
                current_document_lines = []
                first_line = doc_match.group(1).strip()
                if first_line:  # If there's text on the same line as "Document:"
                    current_document_lines.append(first_line)
                current_label = None
                in_document = True
                continue

            # Look for label lines
            label_match = re.match(r"Label:\s*(.+)", line_stripped, re.IGNORECASE)
            if label_match:
                current_label = label_match.group(1).strip()
                in_document = False

                # Save if we have both document and label
                if current_document_lines and current_label:
                    document_text = "\n".join(current_document_lines).strip()
                    examples.append(
                        {"document": document_text, "label": current_label}
                    )
                    current_document_lines = []
                    current_label = None
                continue

            # If we're in a document section and line is not empty, add to current document
            if in_document and line_stripped:
                current_document_lines.append(line_stripped)

        # Save last example if exists
        if current_document_lines and current_label:
            document_text = "\n".join(current_document_lines).strip()
            examples.append({"document": document_text, "label": current_label})

        return examples

    async def _validate_example(
        self, client: LLMClient, document: str, label: str, model: str
    ) -> tuple[str, float, str]:
        """
        Validate a single example using a model.

        Args:
            client: LLM client
            document: The document/input
            label: The proposed label
            model: Model to use for validation

        Returns:
            Tuple of ("correct" or "incorrect", cost, explanation)
        """
        validation_prompt = f"""Given this task: {self.prompt}

Document: {document}
Proposed Label: {label}

Is the proposed label correct for this document?

First, provide your assessment: "CORRECT" or "INCORRECT"
Then, on a new line, provide a brief explanation (1-2 sentences) of why you made this assessment."""

        messages = [{"role": "user", "content": validation_prompt}]
        cost = 0.0

        try:
            response = await client.complete(model=model, messages=messages)
            response_text = response.choices[0].message.content.strip()

            # Calculate cost
            try:
                cost = completion_cost(completion_response=response)
            except Exception as ce:
                logger.warning("cost_calculation_failed", model=model, error=str(ce))

            # Parse the response to extract vote and explanation
            lines = response_text.split('\n', 1)
            first_line = lines[0].strip().lower()
            explanation = lines[1].strip() if len(lines) > 1 else response_text

            # Extract correct/incorrect from response
            if "correct" in first_line and "incorrect" not in first_line:
                return "correct", cost, explanation
            elif "incorrect" in first_line:
                return "incorrect", cost, explanation
            else:
                # Try to find it anywhere in the text
                response_lower = response_text.lower()
                if "correct" in response_lower and "incorrect" not in response_lower:
                    return "correct", cost, response_text
                elif "incorrect" in response_lower:
                    return "incorrect", cost, response_text
                else:
                    # Default to incorrect if unclear
                    logger.warning(
                        "unclear_validation_response",
                        model=model,
                        response=response_text,
                    )
                    return "incorrect", cost, "Unclear response from validator"

        except Exception as e:
            logger.error(
                "validation_failed",
                model=model,
                error=str(e),
            )
            return "incorrect", cost, f"Validation failed: {str(e)}"
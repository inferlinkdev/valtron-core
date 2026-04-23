"""Tests for few-shot training data generator module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from valtron_core.few_shot_training_data_generator import (
    LabeledExample,
    FewShotTrainingDataGenerator,
)


class TestLabeledExample:
    """Tests for LabeledExample dataclass."""

    def test_labeled_example_creation(self):
        """Test creating a labeled example."""
        example = LabeledExample(document="Test document", label="positive")

        assert example.document == "Test document"
        assert example.label == "positive"

    def test_labeled_example_various_label_types(self):
        """Test labeled example with various label types."""
        # String label
        ex1 = LabeledExample(document="doc1", label="positive")
        assert ex1.label == "positive"

        # Integer label
        ex2 = LabeledExample(document="doc2", label=1)
        assert ex2.label == 1

        # Boolean label
        ex3 = LabeledExample(document="doc3", label=True)
        assert ex3.label is True

        # Dict label (for extraction tasks)
        ex4 = LabeledExample(document="doc4", label={"entity": "test"})
        assert ex4.label == {"entity": "test"}


class TestFewShotTrainingDataGeneratorInit:
    """Tests for FewShotTrainingDataGenerator initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        examples = [
            LabeledExample(document="doc1", label="positive"),
            LabeledExample(document="doc2", label="negative"),
        ]

        generator = FewShotTrainingDataGenerator(
            prompt="Classify sentiment",
            examples=examples,
        )

        assert generator.prompt == "Classify sentiment"
        assert len(generator.examples) == 2

    def test_init_with_source_data(self):
        """Test initialization with source data."""
        examples = [LabeledExample(document="doc1", label="positive")]
        source_data = [
            {"content": "doc1", "label": "positive", "metadata": {"key": "value"}}
        ]

        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=examples,
            source_data=source_data,
        )

        assert generator.metadata_schema == {"key": "value"}

    def test_init_balances_examples(self):
        """Test that examples are balanced when exceeding max_few_shots."""
        # Create 20 examples with unbalanced labels
        examples = [
            LabeledExample(document=f"pos_{i}", label="positive") for i in range(15)
        ] + [
            LabeledExample(document=f"neg_{i}", label="negative") for i in range(5)
        ]

        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=examples,
            max_few_shots=10,
        )

        # Should have exactly 10 examples
        assert len(generator.examples) == 10

        # Should be roughly balanced (5 of each)
        pos_count = sum(1 for e in generator.examples if e.label == "positive")
        neg_count = sum(1 for e in generator.examples if e.label == "negative")
        assert pos_count == 5
        assert neg_count == 5


class TestBasicMethods:
    """Tests for basic methods."""

    def test_add_example(self):
        """Test adding an example."""
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=[],
        )

        generator.add_example("New document", "positive")

        assert len(generator.examples) == 1
        assert generator.examples[0].document == "New document"
        assert generator.examples[0].label == "positive"

    def test_get_examples(self):
        """Test getting examples."""
        examples = [LabeledExample(document="doc1", label="positive")]
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=examples,
        )

        result = generator.get_examples()

        assert result == examples

    def test_get_prompt(self):
        """Test getting the prompt."""
        generator = FewShotTrainingDataGenerator(
            prompt="Test prompt",
            examples=[],
        )

        assert generator.get_prompt() == "Test prompt"

    def test_update_prompt(self):
        """Test updating the prompt."""
        generator = FewShotTrainingDataGenerator(
            prompt="Old prompt",
            examples=[],
        )

        generator.update_prompt("New prompt")

        assert generator.prompt == "New prompt"


class TestPromptGeneration:
    """Tests for prompt generation methods."""

    def test_generate_document_prompt(self):
        """Test generating a document prompt."""
        examples = [
            LabeledExample(document="This is a test document with some content.", label="positive"),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Classify sentiment",
            examples=examples,
        )

        prompt = generator.generate_document_prompt()

        assert "REFERENCE DOCUMENTS" in prompt
        assert "GENERATE" in prompt
        assert "novel" in prompt.lower()

    def test_generate_label_prompt(self):
        """Test generating a label prompt."""
        examples = [
            LabeledExample(document="Great product!", label="positive"),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Classify sentiment",
            examples=examples,
        )

        prompt = generator.generate_label_prompt("This is a new document to label")

        assert "Task:" in prompt
        assert "REFERENCE EXAMPLES" in prompt
        assert "NOW LABEL THIS DOCUMENT" in prompt
        assert "This is a new document to label" in prompt

    def test_generate_single_example_prompt(self):
        """Test generating a single example prompt."""
        examples = [
            LabeledExample(document="Great product!", label="positive"),
            LabeledExample(document="Terrible!", label="negative"),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Classify sentiment",
            examples=examples,
        )

        prompt = generator.generate_single_example_prompt("positive")

        assert "Required label: positive" in prompt
        assert "Output format" in prompt
        assert "Document:" in prompt

    def test_generate_from_few_shot(self):
        """Test generating from few-shot examples."""
        examples = [
            LabeledExample(document="Great!", label="positive"),
            LabeledExample(document="Terrible!", label="negative"),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Classify sentiment",
            examples=examples,
        )

        prompt = generator.generate_from_few_shot(num_examples=10)

        assert "Task:" in prompt
        assert "positive" in prompt
        assert "negative" in prompt
        assert "GENERATION INSTRUCTIONS" in prompt


class TestLLMIntegration:
    """Tests for LLM integration methods."""

    @pytest.mark.asyncio
    async def test_generate_examples(self):
        """Test generating examples using LLM."""
        examples = [
            LabeledExample(document="Great!", label="positive"),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=examples,
        )

        with patch("valtron_core.few_shot_training_data_generator.LLMClient") as MockClient:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Document: New doc\nLabel: positive"
            mock_client.complete = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            result = await generator.generate_examples(model="gpt-4o-mini", num_examples=5)

            assert "Document:" in result or "positive" in result

    @pytest.mark.asyncio
    async def test_generate_and_validate_examples_classification(self):
        """Test generating and validating classification examples."""
        examples = [
            LabeledExample(document="Great!", label="positive"),
            LabeledExample(document="Bad!", label="negative"),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Classify sentiment",
            examples=examples,
        )

        with patch("valtron_core.few_shot_training_data_generator.LLMClient") as MockClient:
            mock_client = Mock()

            # Mock generation response
            gen_response = Mock()
            gen_response.choices = [Mock()]
            gen_response.choices[0].message = Mock()
            gen_response.choices[0].message.content = "Document: New document text\nLabel: positive"

            # Mock validation response
            val_response = Mock()
            val_response.choices = [Mock()]
            val_response.choices[0].message = Mock()
            val_response.choices[0].message.content = "CORRECT\nThe label matches the sentiment."

            mock_client.complete = AsyncMock(side_effect=[gen_response, val_response, val_response, val_response])
            MockClient.return_value = mock_client

            with patch("valtron_core.few_shot_training_data_generator.completion_cost", return_value=0.001):
                result = await generator.generate_and_validate_examples(
                    generator_model="gpt-4o-mini",
                    validator_models=["gpt-4o-mini"],
                    num_examples=1,
                )

            assert "examples" in result
            assert "costs" in result

    @pytest.mark.asyncio
    async def test_generate_and_validate_examples_extraction(self):
        """Test generating and validating extraction examples (JSON labels)."""
        # Extraction task has complex JSON labels
        examples = [
            LabeledExample(
                document="John lives in New York",
                label='{"entities": [{"name": "John", "type": "person"}]}'
            ),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Extract entities",
            examples=examples,
        )

        with patch("valtron_core.few_shot_training_data_generator.LLMClient") as MockClient:
            mock_client = Mock()

            # Mock document generation response
            doc_response = Mock()
            doc_response.choices = [Mock()]
            doc_response.choices[0].message = Mock()
            doc_response.choices[0].message.content = "Alice works at Google"

            # Mock label generation response
            label_response = Mock()
            label_response.choices = [Mock()]
            label_response.choices[0].message = Mock()
            label_response.choices[0].message.content = '{"entities": [{"name": "Alice", "type": "person"}]}'

            # Mock validation response
            val_response = Mock()
            val_response.choices = [Mock()]
            val_response.choices[0].message = Mock()
            val_response.choices[0].message.content = "CORRECT\nEntities correctly extracted."

            mock_client.complete = AsyncMock(
                side_effect=[doc_response, label_response, val_response, val_response, val_response]
            )
            MockClient.return_value = mock_client

            with patch("valtron_core.few_shot_training_data_generator.completion_cost", return_value=0.001):
                result = await generator.generate_and_validate_examples(
                    generator_model="gpt-4o-mini",
                    validator_models=["gpt-4o-mini"],
                    num_examples=1,
                )

            assert "examples" in result


class TestParsing:
    """Tests for parsing methods."""

    def test_parse_single_example_text(self):
        """Test parsing a single example from text format."""
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=[LabeledExample(document="test", label="positive")],
        )

        content = "Document: This is a test document\nLabel: positive"
        result = generator._parse_single_example(content, "positive")

        assert result is not None
        assert result["document"] == "This is a test document"
        assert result["label"] == "positive"

    def test_parse_single_example_multiline_document(self):
        """Test parsing example with multiline document."""
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=[LabeledExample(document="test", label="positive")],
        )

        content = "Document: First line\nSecond line\nThird line\nLabel: positive"
        result = generator._parse_single_example(content, "positive")

        assert result is not None
        assert "First line" in result["document"]
        assert "Second line" in result["document"]

    def test_parse_single_example_fallback_label(self):
        """Test that expected label is used if none found."""
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=[LabeledExample(document="test", label="positive")],
        )

        content = "Document: Just a document without a label line"
        result = generator._parse_single_example(content, "positive")

        assert result is not None
        assert result["label"] == "positive"

    def test_parse_generated_examples_multiline(self):
        """Test parsing multiple generated examples."""
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=[LabeledExample(document="test", label="positive")],
        )

        content = """Document: First document
Label: positive

Document: Second document
Label: negative"""

        result = generator._parse_generated_examples(content)

        assert len(result) == 2
        assert result[0]["document"] == "First document"
        assert result[0]["label"] == "positive"
        assert result[1]["document"] == "Second document"
        assert result[1]["label"] == "negative"


class TestInternalLogic:
    """Tests for internal logic methods."""

    def test_balance_examples(self):
        """Test balancing examples."""
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=[],
        )

        examples = [
            LabeledExample(document=f"pos_{i}", label="positive") for i in range(10)
        ] + [
            LabeledExample(document=f"neg_{i}", label="negative") for i in range(10)
        ]

        balanced = generator._balance_examples(examples, max_count=6)

        assert len(balanced) == 6
        pos_count = sum(1 for e in balanced if e.label == "positive")
        neg_count = sum(1 for e in balanced if e.label == "negative")
        assert pos_count == 3
        assert neg_count == 3

    def test_is_extraction_task_simple(self):
        """Test detecting simple classification task."""
        examples = [
            LabeledExample(document="doc1", label="positive"),
            LabeledExample(document="doc2", label="negative"),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=examples,
        )

        assert generator._is_extraction_task() is False

    def test_is_extraction_task_complex(self):
        """Test detecting extraction task with complex JSON labels."""
        examples = [
            LabeledExample(
                document="doc1",
                label='{"entities": [{"name": "John", "type": "person"}]}'
            ),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Extract entities",
            examples=examples,
        )

        assert generator._is_extraction_task() is True

    def test_is_extraction_task_dict_label(self):
        """Test detecting extraction task with dict label."""
        examples = [
            LabeledExample(
                document="doc1",
                label={"entities": [{"name": "John"}]}
            ),
        ]
        generator = FewShotTrainingDataGenerator(
            prompt="Extract entities",
            examples=examples,
        )

        assert generator._is_extraction_task() is True

    def test_extract_metadata_schema(self):
        """Test extracting metadata schema from source data."""
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=[],
        )

        source_data = [
            {"content": "doc1", "metadata": {"key1": "value1", "key2": "value2"}}
        ]

        schema = generator._extract_metadata_schema(source_data)

        assert schema == {"key1": "value1", "key2": "value2"}

    def test_extract_metadata_schema_empty(self):
        """Test extracting metadata schema from empty source data."""
        generator = FewShotTrainingDataGenerator(
            prompt="Classify",
            examples=[],
        )

        schema = generator._extract_metadata_schema([])
        assert schema is None

        schema = generator._extract_metadata_schema(None)
        assert schema is None

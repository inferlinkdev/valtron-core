"""Tests for ExplanationEnhancer."""

import pytest

from valtron_core.prompt_optimizer import ExplanationEnhancer


class TestExplanationEnhancer:
    """Test the ExplanationEnhancer class."""

    @pytest.fixture
    def enhancer(self):
        """Create an ExplanationEnhancer instance."""
        return ExplanationEnhancer()

    @pytest.mark.asyncio
    async def test_detect_classification_sentiment(self, enhancer):
        """Test detection of sentiment classification."""
        prompt = "Classify the sentiment as positive or negative"
        assert enhancer._detect_classification_task(prompt) is True

    @pytest.mark.asyncio
    async def test_detect_classification_label(self, enhancer):
        """Test detection of labeling task."""
        prompt = "Label this document as spam or not spam"
        assert enhancer._detect_classification_task(prompt) is True

    @pytest.mark.asyncio
    async def test_detect_classification_yes_no(self, enhancer):
        """Test detection of yes/no question."""
        prompt = "Determine if this is true. Respond with yes or no."
        assert enhancer._detect_classification_task(prompt) is True

    @pytest.mark.asyncio
    async def test_detect_classification_json(self, enhancer):
        """Test detection of JSON response format."""
        prompt = 'Answer with {"response": "yes"} or {"response": "no"}'
        assert enhancer._detect_classification_task(prompt) is True

    @pytest.mark.asyncio
    async def test_detect_non_classification(self, enhancer):
        """Test that non-classification tasks are not detected."""
        prompts = [
            "Summarize the following article",
            "Translate this text to French",
            "Extract the main entities from this document",
            "Write a poem about nature",
        ]
        for prompt in prompts:
            assert enhancer._detect_classification_task(prompt) is False

    @pytest.mark.asyncio
    async def test_enhance_classification_prompt(self, enhancer):
        """Test enhancing a classification prompt."""
        original = "Classify sentiment as positive or negative"
        result = await enhancer.optimize(original)

        assert result["original_prompt"] == original
        assert result["enhanced_prompt"] != original
        assert "Explanation:" in result["enhanced_prompt"]
        assert "Answer:" in result["enhanced_prompt"]
        assert result["strategy"] == "explanation_enhancement"
        assert result["detection"]["is_classification"] is True
        assert result["detection"]["enhanced"] is True

    @pytest.mark.asyncio
    async def test_skip_non_classification_prompt(self, enhancer):
        """Test that non-classification prompts are not enhanced."""
        original = "Summarize the main points of this article"
        result = await enhancer.optimize(original)

        assert result["original_prompt"] == original
        assert result["enhanced_prompt"] == original
        assert result["strategy"] == "explanation_enhancement"
        assert result["detection"]["is_classification"] is False
        assert result["detection"]["enhanced"] is False

    @pytest.mark.asyncio
    async def test_enhanced_prompt_structure(self, enhancer):
        """Test that enhanced prompt maintains structure."""
        original = "Is this spam? Answer yes or no."
        enhanced = enhancer._add_explanation_requirement(original)

        # Should contain original prompt
        assert original in enhanced

        # Should contain formatting instructions
        assert "Explanation:" in enhanced
        assert "Answer:" in enhanced
        assert "same format as originally requested" in enhanced

    @pytest.mark.asyncio
    async def test_detection_keywords(self, enhancer):
        """Test various classification keywords."""
        classification_prompts = [
            "Classify this document",
            "Label the sentiment",
            "Categorize into sports or politics",
            "Determine if the locations match",
            "Is this a valid email?",
            "Select from: A, B, or C",
            "Choose one: positive, negative, neutral",
        ]

        for prompt in classification_prompts:
            assert (
                enhancer._detect_classification_task(prompt) is True
            ), f"Failed to detect: {prompt}"

    @pytest.mark.asyncio
    async def test_extract_json_schema(self, enhancer):
        """Test JSON schema extraction."""
        prompt = 'Classify as spam. Return {"is_spam": true}'
        schema_info = enhancer._extract_json_schema(prompt)

        assert schema_info is not None
        assert "parsed" in schema_info
        assert schema_info["parsed"]["is_spam"] is True

    @pytest.mark.asyncio
    async def test_extract_no_json_schema(self, enhancer):
        """Test when no JSON schema is present."""
        prompt = "Classify the sentiment as positive or negative"
        schema_info = enhancer._extract_json_schema(prompt)

        assert schema_info is None

    @pytest.mark.asyncio
    async def test_enhance_json_schema(self, enhancer):
        """Test enhancing a prompt with JSON schema."""
        prompt = 'Classify sentiment. Return JSON: {"sentiment": "positive"}'
        result = await enhancer.optimize(prompt)

        enhanced = result["enhanced_prompt"]

        # Should contain the explanation field in JSON
        assert '"explanation"' in enhanced
        assert "Your 1-2 sentence reasoning here" in enhanced
        # Should still contain the original field
        assert '"sentiment"' in enhanced

    @pytest.mark.asyncio
    async def test_json_schema_format(self, enhancer):
        """Test that JSON schema is properly formatted."""
        import json

        prompt = 'Return {"response": "yes"}'
        schema_info = enhancer._extract_json_schema(prompt)
        enhanced_prompt = enhancer._enhance_json_schema(prompt, schema_info)

        # Extract the JSON from enhanced prompt
        import re

        json_match = re.search(r'\{[^{}]*(?:"[^"]*"[^{}]*)*\}', enhanced_prompt)
        assert json_match is not None

        enhanced_json = json.loads(json_match.group(0))

        # Should have explanation first, then original fields
        keys = list(enhanced_json.keys())
        assert keys[0] == "explanation"
        assert "response" in keys

    @pytest.mark.asyncio
    async def test_text_format_when_no_schema(self, enhancer):
        """Test text format is used when no JSON schema."""
        prompt = "Classify as positive or negative"
        result = await enhancer.optimize(prompt)

        enhanced = result["enhanced_prompt"]

        # Should use text format
        assert "Explanation:" in enhanced
        assert "Answer:" in enhanced
        # Should not have JSON formatting
        assert '"explanation"' not in enhanced

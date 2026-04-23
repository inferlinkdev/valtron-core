"""Tests for the decompose module."""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, Field

from valtron_core.decompose import (
    DecomposedEvaluator,
    SplitPointInfo,
    _deduplicate_list,
    _multi_pass_merge,
    cleanup_few_shot_sub_prompts,
    create_sub_schemas,
    decompose_few_shot_examples,
    filter_hallucinated_values,
    find_split_point,
    generate_sub_prompts,
    inject_few_shot_into_sub_prompts,
    merge_sub_results,
)
from valtron_core.models import Document, Label, PredictionResult


# ---------------------------------------------------------------------------
# Sample Pydantic models for testing
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    name: str
    type: str = ""


class Entities(BaseModel):
    people: list[Entity] = Field(default_factory=list)
    pathogens: list[Entity] = Field(default_factory=list)


class ExtractionSchema(BaseModel):
    entities: Entities


class SingleListModel(BaseModel):
    items: list[str] = Field(default_factory=list)


class NoListModel(BaseModel):
    name: str = ""
    age: int = 0


class RootLevelLists(BaseModel):
    people: list[Entity] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)


class DeeplyNested(BaseModel):
    level1: ExtractionSchema


# ---------------------------------------------------------------------------
# find_split_point tests
# ---------------------------------------------------------------------------

class TestFindSplitPoint:
    def test_finds_nested_split_point(self):
        result = find_split_point(ExtractionSchema)
        assert result is not None
        assert result.path_from_root == ["entities"]
        assert result.split_model is Entities
        assert sorted(result.list_field_names) == ["pathogens", "people"]
        assert "people" in result.list_field_annotations
        assert "pathogens" in result.list_field_annotations

    def test_single_list_returns_none(self):
        result = find_split_point(SingleListModel)
        assert result is None

    def test_no_lists_returns_none(self):
        result = find_split_point(NoListModel)
        assert result is None

    def test_root_level_split(self):
        result = find_split_point(RootLevelLists)
        assert result is not None
        assert result.path_from_root == []
        assert result.split_model is RootLevelLists
        assert sorted(result.list_field_names) == ["locations", "people"]

    def test_deeply_nested(self):
        result = find_split_point(DeeplyNested)
        assert result is not None
        assert result.path_from_root == ["level1", "entities"]
        assert result.split_model is Entities


# ---------------------------------------------------------------------------
# create_sub_schemas tests
# ---------------------------------------------------------------------------

class TestCreateSubSchemas:
    def test_creates_correct_sub_schemas(self):
        split_info = find_split_point(ExtractionSchema)
        assert split_info is not None

        schemas = create_sub_schemas(split_info, ExtractionSchema)

        assert "people" in schemas
        assert "pathogens" in schemas

        # Each schema should be a BaseModel subclass
        for name, schema in schemas.items():
            assert issubclass(schema, BaseModel)

        # "people" schema should have an 'entities' field whose model has only 'people'
        people_schema = schemas["people"]
        people_fields = people_schema.model_fields
        assert "entities" in people_fields
        inner = people_fields["entities"].annotation
        assert "people" in inner.model_fields
        assert "pathogens" not in inner.model_fields

        # "pathogens" schema should have only 'pathogens'
        pathogens_schema = schemas["pathogens"]
        inner_p = pathogens_schema.model_fields["entities"].annotation
        assert "pathogens" in inner_p.model_fields
        assert "people" not in inner_p.model_fields

    def test_creates_sub_schemas_with_explanation(self):
        split_info = find_split_point(ExtractionSchema)
        assert split_info is not None

        schemas = create_sub_schemas(split_info, ExtractionSchema, include_explanation=True)

        for name, schema in schemas.items():
            fields = schema.model_fields
            assert "explanation" in fields, f"Schema for '{name}' missing explanation field"
            assert "entities" in fields

    def test_root_level_sub_schemas(self):
        split_info = find_split_point(RootLevelLists)
        assert split_info is not None

        schemas = create_sub_schemas(split_info, RootLevelLists)

        people_schema = schemas["people"]
        assert "people" in people_schema.model_fields
        assert "locations" not in people_schema.model_fields

        locations_schema = schemas["locations"]
        assert "locations" in locations_schema.model_fields
        assert "people" not in locations_schema.model_fields


# ---------------------------------------------------------------------------
# generate_sub_prompts tests
# ---------------------------------------------------------------------------

class TestGenerateSubPrompts:
    @pytest.mark.asyncio
    async def test_default_prompts_uses_llm(self, mock_env_vars):
        """Default mode rewrites each prompt via LLM."""
        original = "Extract entities from: {document}"
        field_names = ["people", "pathogens"]

        async def mock_complete(model, messages, **kwargs):
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            # Return a rewritten prompt that references the field
            content = messages[0]["content"]
            if "people" in content:
                response.choices[0].message.content = "Extract ONLY people from: {document}"
            else:
                response.choices[0].message.content = "Extract ONLY pathogens from: {document}"
            response._hidden_params = {"response_cost": 0.0}
            return response

        from valtron_core.client import LLMClient
        client = LLMClient()
        with patch.object(client, "complete", side_effect=mock_complete):
            prompts = await generate_sub_prompts(original, field_names, client=client)

        assert set(prompts.keys()) == {"people", "pathogens"}
        assert "people" in prompts["people"]
        assert "pathogens" in prompts["pathogens"]
        # Rewritten prompts should NOT just be the original with a suffix
        assert prompts["people"] != original
        assert prompts["pathogens"] != original

    @pytest.mark.asyncio
    async def test_custom_prompts(self):
        original = "Extract entities from: {document}"
        custom = {
            "people": "Extract ONLY people from: {document}",
            "pathogens": "Extract ONLY pathogens from: {document}",
        }

        prompts = await generate_sub_prompts(
            original, ["people", "pathogens"], custom_sub_prompts=custom,
        )

        assert prompts == custom

    @pytest.mark.asyncio
    async def test_custom_prompts_missing_field_raises(self):
        with pytest.raises(ValueError, match="missing entries"):
            await generate_sub_prompts(
                "prompt",
                ["people", "pathogens"],
                custom_sub_prompts={"people": "only people"},
            )


# ---------------------------------------------------------------------------
# merge_sub_results tests
# ---------------------------------------------------------------------------

class TestMergeSubResults:
    def test_basic_merge(self):
        split_info = SplitPointInfo(
            path_from_root=["entities"],
            split_model=Entities,
            list_field_names=["people", "pathogens"],
            list_field_annotations={},
        )

        sub_results = {
            "people": json.dumps({"entities": {"people": [{"name": "Alice", "type": "person"}]}}),
            "pathogens": json.dumps({"entities": {"pathogens": [{"name": "E.coli", "type": "bacteria"}]}}),
        }

        merged = merge_sub_results(sub_results, split_info)
        parsed = json.loads(merged)

        assert parsed["entities"]["people"] == [{"name": "Alice", "type": "person"}]
        assert parsed["entities"]["pathogens"] == [{"name": "E.coli", "type": "bacteria"}]

    def test_merge_with_root_level_split(self):
        split_info = SplitPointInfo(
            path_from_root=[],
            split_model=RootLevelLists,
            list_field_names=["people", "locations"],
            list_field_annotations={},
        )

        sub_results = {
            "people": json.dumps({"people": [{"name": "Bob", "type": "person"}]}),
            "locations": json.dumps({"locations": ["Paris", "London"]}),
        }

        merged = merge_sub_results(sub_results, split_info)
        parsed = json.loads(merged)

        assert parsed["people"] == [{"name": "Bob", "type": "person"}]
        assert parsed["locations"] == ["Paris", "London"]

    def test_merge_parse_error_defaults_to_empty(self):
        split_info = SplitPointInfo(
            path_from_root=["entities"],
            split_model=Entities,
            list_field_names=["people", "pathogens"],
            list_field_annotations={},
        )

        sub_results = {
            "people": "not valid json{{{",
            "pathogens": json.dumps({"entities": {"pathogens": [{"name": "E.coli", "type": "bacteria"}]}}),
        }

        merged = merge_sub_results(sub_results, split_info)
        parsed = json.loads(merged)

        assert parsed["entities"]["people"] == []
        assert parsed["entities"]["pathogens"] == [{"name": "E.coli", "type": "bacteria"}]

    def test_merge_missing_field_defaults_to_empty(self):
        split_info = SplitPointInfo(
            path_from_root=["entities"],
            split_model=Entities,
            list_field_names=["people", "pathogens"],
            list_field_annotations={},
        )

        sub_results = {
            "people": json.dumps({"entities": {}}),
            "pathogens": json.dumps({"entities": {"pathogens": []}}),
        }

        merged = merge_sub_results(sub_results, split_info)
        parsed = json.loads(merged)

        assert parsed["entities"]["people"] == []
        assert parsed["entities"]["pathogens"] == []


# ---------------------------------------------------------------------------
# DecomposedEvaluator tests
# ---------------------------------------------------------------------------

class TestDecomposedEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate(self, mock_env_vars):
        """Integration test with mocked LLM calls."""
        split_info = find_split_point(ExtractionSchema)
        assert split_info is not None

        sub_schemas = create_sub_schemas(split_info, ExtractionSchema)
        sub_prompts = {
            "people": "Extract ONLY people from: {document}",
            "pathogens": "Extract ONLY pathogens from: {document}",
        }

        documents = [
            Document(id="doc-1", content="Alice found E.coli in the sample."),
        ]
        labels = [
            Label(
                document_id="doc-1",
                value=json.dumps({
                    "entities": {
                        "people": [{"name": "Alice", "type": "person"}],
                        "pathogens": [{"name": "E.coli", "type": "bacteria"}],
                    }
                }),
            ),
        ]

        # Mock LLM to return appropriate per-field responses
        people_response = json.dumps({"entities": {"people": [{"name": "Alice", "type": "person"}]}})
        pathogens_response = json.dumps({"entities": {"pathogens": [{"name": "E.coli", "type": "bacteria"}]}})

        call_count = 0

        async def mock_complete(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response._hidden_params = {"response_cost": 0.0001}

            prompt_content = messages[0]["content"]
            if "people" in prompt_content:
                response.choices[0].message.content = people_response
            else:
                response.choices[0].message.content = pathogens_response
            return response

        evaluator = DecomposedEvaluator()

        with patch.object(evaluator.evaluator.client, "complete", side_effect=mock_complete):
            result = await evaluator.evaluate(
                documents=documents,
                labels=labels,
                sub_prompts=sub_prompts,
                sub_schemas=sub_schemas,
                split_info=split_info,
                model="test-model",
            )

        assert result.status == "completed"
        assert len(result.predictions) == 1
        assert call_count == 2  # one per field

        pred = result.predictions[0]
        assert pred.metadata.get("decomposed") is True
        assert "sub_results" in pred.metadata

        # Verify merged result contains both entity types
        merged = json.loads(pred.predicted_value)
        assert "entities" in merged
        assert "people" in merged["entities"]
        assert "pathogens" in merged["entities"]

    @pytest.mark.asyncio
    async def test_evaluate_with_field_metrics(self, mock_env_vars):
        """Test that field_metrics_config is applied to merged results."""
        split_info = find_split_point(ExtractionSchema)
        assert split_info is not None

        sub_schemas = create_sub_schemas(split_info, ExtractionSchema)
        sub_prompts = {
            "people": "Extract ONLY people from: {document}",
            "pathogens": "Extract ONLY pathogens from: {document}",
        }

        documents = [
            Document(id="doc-1", content="Alice found E.coli."),
        ]
        labels = [
            Label(
                document_id="doc-1",
                value=json.dumps({
                    "entities": {
                        "people": [{"name": "Alice", "type": "person"}],
                        "pathogens": [{"name": "E.coli", "type": "bacteria"}],
                    }
                }),
            ),
        ]

        async def mock_complete(model, messages, **kwargs):
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response._hidden_params = {"response_cost": 0.0}
            prompt_content = messages[0]["content"]
            if "people" in prompt_content:
                response.choices[0].message.content = json.dumps(
                    {"entities": {"people": [{"name": "Alice", "type": "person"}]}}
                )
            else:
                response.choices[0].message.content = json.dumps(
                    {"entities": {"pathogens": [{"name": "E.coli", "type": "bacteria"}]}}
                )
            return response

        from valtron_core.models import FieldMetricsConfig

        field_metrics_config = FieldMetricsConfig(
            config={
                "type": "object",
                "fields": {
                    "entities": {
                        "type": "object",
                        "fields": {
                            "people": {
                                "type": "list",
                                "metric_config": {
                                    "item_logic": {
                                        "type": "object",
                                        "fields": {
                                            "name": {"type": "leaf", "metric_config": {"metric": "exact"}},
                                            "type": {"type": "leaf", "metric_config": {"metric": "exact"}},
                                        },
                                    }
                                },
                            },
                            "pathogens": {
                                "type": "list",
                                "metric_config": {
                                    "item_logic": {
                                        "type": "object",
                                        "fields": {
                                            "name": {"type": "leaf", "metric_config": {"metric": "exact"}},
                                            "type": {"type": "leaf", "metric_config": {"metric": "exact"}},
                                        },
                                    }
                                },
                            },
                        },
                    }
                },
            }
        )

        evaluator = DecomposedEvaluator()

        with patch.object(evaluator.evaluator.client, "complete", side_effect=mock_complete):
            result = await evaluator.evaluate(
                documents=documents,
                labels=labels,
                sub_prompts=sub_prompts,
                sub_schemas=sub_schemas,
                split_info=split_info,
                model="test-model",
                field_metrics_config=field_metrics_config,
            )

        assert result.status == "completed"
        pred = result.predictions[0]
        assert pred.field_metrics is not None
        assert pred.example_score > 0


# ---------------------------------------------------------------------------
# decompose_few_shot_examples tests
# ---------------------------------------------------------------------------

class TestDecomposeFewShotExamples:
    def _make_split_info(self) -> SplitPointInfo:
        return SplitPointInfo(
            path_from_root=["entities"],
            split_model=Entities,
            list_field_names=["people", "pathogens"],
            list_field_annotations={},
        )

    def test_decompose_few_shot_examples(self):
        """Full examples are split into per-field examples with correct labels."""
        split_info = self._make_split_info()
        examples = [
            {
                "document": "Alice found E.coli in the sample.",
                "label": json.dumps({
                    "entities": {
                        "people": [{"name": "Alice", "type": "person"}],
                        "pathogens": [{"name": "E.coli", "type": "bacteria"}],
                    }
                }),
            },
        ]

        result = decompose_few_shot_examples(examples, split_info)

        assert set(result.keys()) == {"people", "pathogens"}
        assert len(result["people"]) == 1
        assert len(result["pathogens"]) == 1

        # Check the people example has only the people field in its label
        people_label = json.loads(result["people"][0]["label"])
        assert people_label == {"entities": {"people": [{"name": "Alice", "type": "person"}]}}
        assert result["people"][0]["document"] == "Alice found E.coli in the sample."

        # Check the pathogens example
        pathogens_label = json.loads(result["pathogens"][0]["label"])
        assert pathogens_label == {"entities": {"pathogens": [{"name": "E.coli", "type": "bacteria"}]}}

    def test_decompose_few_shot_examples_skips_empty(self):
        """Examples with empty lists for a field are excluded for that field."""
        split_info = self._make_split_info()
        examples = [
            {
                "document": "No pathogens found. Alice was there.",
                "label": json.dumps({
                    "entities": {
                        "people": [{"name": "Alice", "type": "person"}],
                        "pathogens": [],
                    }
                }),
            },
        ]

        result = decompose_few_shot_examples(examples, split_info)

        assert len(result["people"]) == 1
        assert len(result["pathogens"]) == 0

    def test_decompose_few_shot_examples_root_level(self):
        """Works with root-level split points (empty path_from_root)."""
        split_info = SplitPointInfo(
            path_from_root=[],
            split_model=RootLevelLists,
            list_field_names=["people", "locations"],
            list_field_annotations={},
        )
        examples = [
            {
                "document": "Bob visited Paris.",
                "label": json.dumps({
                    "people": [{"name": "Bob", "type": "person"}],
                    "locations": ["Paris"],
                }),
            },
        ]

        result = decompose_few_shot_examples(examples, split_info)

        people_label = json.loads(result["people"][0]["label"])
        assert people_label == {"people": [{"name": "Bob", "type": "person"}]}

        locations_label = json.loads(result["locations"][0]["label"])
        assert locations_label == {"locations": ["Paris"]}


# ---------------------------------------------------------------------------
# inject_few_shot_into_sub_prompts tests
# ---------------------------------------------------------------------------

class TestInjectFewShotIntoSubPrompts:
    def test_inject_few_shot_into_sub_prompts(self):
        """Examples are injected into the sub-prompt before {document}."""
        sub_prompts = {
            "people": "Extract people from: {document}",
            "pathogens": "Extract pathogens from: {document}",
        }
        field_examples = {
            "people": [
                {"document": "Alice was there.", "label": '{"entities": {"people": [{"name": "Alice"}]}}'},
            ],
            "pathogens": [
                {"document": "E.coli found.", "label": '{"entities": {"pathogens": [{"name": "E.coli"}]}}'},
            ],
        }

        result = inject_few_shot_into_sub_prompts(sub_prompts, field_examples)

        assert "{document}" in result["people"]
        assert "Here are some examples:" in result["people"]
        assert "Alice was there." in result["people"]

        assert "{document}" in result["pathogens"]
        assert "E.coli found." in result["pathogens"]

        # Examples should appear before {document}
        people_prompt = result["people"]
        examples_pos = people_prompt.index("Here are some examples:")
        doc_pos = people_prompt.index("{document}")
        assert examples_pos < doc_pos

    def test_inject_few_shot_into_sub_prompts_no_examples(self):
        """Sub-prompts unchanged when no examples exist for a field."""
        sub_prompts = {
            "people": "Extract people from: {document}",
            "pathogens": "Extract pathogens from: {document}",
        }
        field_examples = {
            "people": [
                {"document": "Alice was there.", "label": '{"entities": {"people": [{"name": "Alice"}]}}'},
            ],
            "pathogens": [],
        }

        result = inject_few_shot_into_sub_prompts(sub_prompts, field_examples)

        assert result["pathogens"] == "Extract pathogens from: {document}"
        assert "Here are some examples:" in result["people"]


# ---------------------------------------------------------------------------
# cleanup_few_shot_sub_prompts tests
# ---------------------------------------------------------------------------

class TestCleanupFewShotSubPrompts:
    @pytest.mark.asyncio
    async def test_cleanup_calls_llm_and_preserves_document_placeholder(self, mock_env_vars):
        """Cleanup sends each sub-prompt to the LLM and returns cleaned text."""
        sub_prompts = {
            "people": "Extract people.\n\nText:\n\n\nHere are some examples:\n\nExample 1:\nDocument: Alice\nLabel: {}\n\nNow extract from this document:\n\n{document}",
        }

        cleaned_prompt = (
            "Extract people.\n\n"
            "Here are some examples:\n\n"
            "Example 1:\nDocument: Alice\nLabel: {}\n\n"
            "Now extract from this document:\n\n{document}"
        )

        async def mock_complete(model, messages, **kwargs):
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.content = cleaned_prompt
            response._hidden_params = {"response_cost": 0.0}
            return response

        from valtron_core.client import LLMClient
        client = LLMClient()
        with patch.object(client, "complete", side_effect=mock_complete) as mock:
            result = await cleanup_few_shot_sub_prompts(sub_prompts, client=client)

        assert "{document}" in result["people"]
        assert "Text:" not in result["people"]
        mock.assert_called_once()


# ---------------------------------------------------------------------------
# filter_hallucinated_values tests
# ---------------------------------------------------------------------------

class TestFilterHallucinatedValues:
    @pytest.mark.asyncio
    async def test_filter_removes_hallucinated_values(self, mock_env_vars):
        """Values the model says 'no' to are removed from the output."""
        predicted = json.dumps({
            "entities": {
                "people": [{"name": "Alice", "type": "person"}],
                "pathogens": [{"name": "MadeUpVirus", "type": "virus"}],
            }
        })
        document_text = "Alice found bacteria in the sample."

        async def mock_complete(model, messages, **kwargs):
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response._hidden_params = {"response_cost": 0.0}
            content = messages[0]["content"]
            # Check the prompt to determine which value is being verified
            if '"MadeUpVirus"' in content:
                response.choices[0].message.content = "no"
            elif '"virus"' in content:
                response.choices[0].message.content = "no"
            else:
                response.choices[0].message.content = "yes"
            return response

        from valtron_core.client import LLMClient
        client = LLMClient()
        with patch.object(client, "complete", side_effect=mock_complete):
            result = await filter_hallucinated_values(
                predicted, document_text, "test-model", client,
            )

        parsed = json.loads(result)
        assert len(parsed["entities"]["people"]) == 1
        assert parsed["entities"]["people"][0]["name"] == "Alice"
        # MadeUpVirus dict should be removed (both fields said "no")
        assert len(parsed["entities"]["pathogens"]) == 0

    @pytest.mark.asyncio
    async def test_filter_keeps_all_when_none_hallucinated(self, mock_env_vars):
        """All values found in text are kept without any LLM calls."""
        predicted = json.dumps({
            "entities": {
                "people": [{"name": "Alice", "type": "person"}],
                "pathogens": [{"name": "E.coli", "type": "bacteria"}],
            }
        })
        document_text = "Alice found E.coli bacteria, a person was there."

        from valtron_core.client import LLMClient
        client = LLMClient()
        mock_fn = AsyncMock()
        with patch.object(client, "complete", mock_fn):
            result = await filter_hallucinated_values(
                predicted, document_text, "test-model", client,
            )

        # All values appear in the document text, so no LLM calls needed
        mock_fn.assert_not_called()
        parsed = json.loads(result)
        assert len(parsed["entities"]["people"]) == 1
        assert len(parsed["entities"]["pathogens"]) == 1
        assert parsed["entities"]["people"][0]["name"] == "Alice"
        assert parsed["entities"]["pathogens"][0]["name"] == "E.coli"

    @pytest.mark.asyncio
    async def test_filter_skips_llm_for_values_found_in_text(self, mock_env_vars):
        """Values found in text are kept; only missing values go to the LLM."""
        predicted = json.dumps({
            "entities": {
                "people": [{"name": "Alice", "type": "person"}],
                "pathogens": [{"name": "MadeUpVirus", "type": "virus"}],
            }
        })
        # "Alice" and "person" appear in the text; "MadeUpVirus" and "virus" do not
        document_text = "Alice is a person who found bacteria in the sample."

        async def mock_complete(model, messages, **kwargs):
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response._hidden_params = {"response_cost": 0.0}
            # LLM should only be called for MadeUpVirus and virus (not in text)
            response.choices[0].message.content = "no"
            return response

        from valtron_core.client import LLMClient
        client = LLMClient()
        with patch.object(client, "complete", side_effect=mock_complete) as mock_fn:
            result = await filter_hallucinated_values(
                predicted, document_text, "test-model", client,
            )

        # Only 2 LLM calls: "MadeUpVirus" and "virus" (not found in text)
        assert mock_fn.call_count == 2
        parsed = json.loads(result)
        # Alice was kept via pre-check, MadeUpVirus removed by LLM
        assert len(parsed["entities"]["people"]) == 1
        assert parsed["entities"]["people"][0]["name"] == "Alice"
        assert len(parsed["entities"]["pathogens"]) == 0

    @pytest.mark.asyncio
    async def test_filter_handles_parse_error_gracefully(self, mock_env_vars):
        """Invalid JSON input is returned unchanged."""
        invalid_json = "not valid json{{{"

        from valtron_core.client import LLMClient
        client = LLMClient()
        result = await filter_hallucinated_values(
            invalid_json, "some text", "test-model", client,
        )

        assert result == invalid_json

    @pytest.mark.asyncio
    async def test_filter_handles_llm_error_gracefully(self, mock_env_vars):
        """LLM errors cause the value to be kept (fail-open)."""
        predicted = json.dumps({
            "entities": {
                "people": [{"name": "Alice", "type": "person"}],
            }
        })

        async def mock_complete(model, messages, **kwargs):
            raise RuntimeError("LLM service unavailable")

        from valtron_core.client import LLMClient
        client = LLMClient()
        with patch.object(client, "complete", side_effect=mock_complete):
            result = await filter_hallucinated_values(
                predicted, "Alice was here.", "test-model", client,
            )

        parsed = json.loads(result)
        assert len(parsed["entities"]["people"]) == 1
        assert parsed["entities"]["people"][0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_filter_handles_empty_values(self, mock_env_vars):
        """Empty lists produce no LLM calls and return input unchanged."""
        predicted = json.dumps({
            "entities": {
                "people": [],
                "pathogens": [],
            }
        })

        from valtron_core.client import LLMClient
        client = LLMClient()
        mock_fn = AsyncMock()
        with patch.object(client, "complete", mock_fn):
            result = await filter_hallucinated_values(
                predicted, "some text", "test-model", client,
            )

        mock_fn.assert_not_called()
        assert json.loads(result) == json.loads(predicted)


# ---------------------------------------------------------------------------
# _deduplicate_list tests
# ---------------------------------------------------------------------------

class TestDeduplicateList:
    def test_removes_near_duplicates(self):
        """Near-duplicate dicts (e.g. 'H5N1' vs 'h5n1') are collapsed."""
        items = [
            {"name": "H5N1", "type": "virus"},
            {"name": "h5n1", "type": "virus"},
        ]
        result = _deduplicate_list(items)
        assert len(result) == 1
        assert result[0]["name"] == "H5N1"

    def test_keeps_distinct_items(self):
        """Clearly different items are all preserved."""
        items = [
            {"name": "Alice", "type": "person"},
            {"name": "E.coli", "type": "bacteria"},
            {"name": "Paris", "type": "location"},
        ]
        result = _deduplicate_list(items)
        assert len(result) == 3

    def test_empty_list(self):
        assert _deduplicate_list([]) == []

    def test_custom_threshold(self):
        """A very high threshold keeps items that are close but not identical."""
        items = [
            {"name": "Influenza A", "type": "virus"},
            {"name": "Influenza B", "type": "virus"},
        ]
        # With default threshold they might be deduped; with 1.0 they won't
        result = _deduplicate_list(items, threshold=1.0)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _multi_pass_merge tests
# ---------------------------------------------------------------------------

class TestMultiPassMerge:
    def test_unions_lists(self):
        """Items from different passes are unioned."""
        r1 = json.dumps({"entities": {"people": [{"name": "Alice", "type": "person"}]}})
        r2 = json.dumps({"entities": {"people": [{"name": "Bob", "type": "person"}]}})
        merged = json.loads(_multi_pass_merge([r1, r2]))
        names = {item["name"] for item in merged["entities"]["people"]}
        assert names == {"Alice", "Bob"}

    def test_deduplicates(self):
        """Overlapping items across passes are deduplicated."""
        r1 = json.dumps({"entities": {"people": [{"name": "Alice", "type": "person"}]}})
        r2 = json.dumps({"entities": {"people": [{"name": "alice", "type": "person"}]}})
        merged = json.loads(_multi_pass_merge([r1, r2]))
        assert len(merged["entities"]["people"]) == 1

    def test_handles_parse_errors(self):
        """Bad JSON from one pass is skipped; others still used."""
        r1 = "not valid json{{{"
        r2 = json.dumps({"entities": {"people": [{"name": "Alice", "type": "person"}]}})
        merged = json.loads(_multi_pass_merge([r1, r2]))
        assert len(merged["entities"]["people"]) == 1
        assert merged["entities"]["people"][0]["name"] == "Alice"

    def test_non_list_fields_keep_first_pass(self):
        """Non-list fields keep the value from the first valid pass."""
        r1 = json.dumps({"title": "First", "items": [{"name": "A"}]})
        r2 = json.dumps({"title": "Second", "items": [{"name": "B"}]})
        merged = json.loads(_multi_pass_merge([r1, r2]))
        assert merged["title"] == "First"
        assert len(merged["items"]) == 2

    def test_all_invalid_returns_first_response(self):
        """When all responses fail to parse, returns the first raw response."""
        result = _multi_pass_merge(["bad1", "bad2"])
        assert result == "bad1"

    def test_empty_responses(self):
        result = _multi_pass_merge([])
        assert result == "{}"

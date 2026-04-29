"""Tests for the ModelEval recipe (both label and structured extraction modes)."""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pydantic import BaseModel, ValidationError

from valtron_core.recipes.model_eval import ModelEval
from valtron_core.recipes.config import (
    ModelEvalConfig,
    LLMModelConfig,
    Manipulation,
    STRUCTURED_MANIPULATIONS,
)
from valtron_core.models import EvaluationResult, EvaluationMetrics, PredictionResult


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

class SampleSchema(BaseModel):
    name: str
    value: str


EXTRACT_CONFIG = {
    "models": [{"name": "gpt-4o-mini"}],
    "prompt": "Extract: {content}",
}

CLASSIFY_CONFIG = {
    "models": [{"name": "gpt-4o-mini"}],
    "prompt": "Classify: {content}",
}


def _mock_result(model="gpt-4o-mini", prompt="Classify: {content}") -> EvaluationResult:
    result = EvaluationResult(
        run_id="test-run",
        model=model,
        prompt_template=prompt,
        status="completed",
    )
    result.predictions = []
    result.metrics = EvaluationMetrics(
        total_documents=1,
        correct_predictions=1,
        accuracy=1.0,
        average_example_score=1.0,
        total_cost=0.001,
        total_time=1.0,
        average_cost_per_document=0.001,
        average_time_per_document=1.0,
        model=model,
    )
    return result


# ===========================================================================
# Initialization
# ===========================================================================

class TestModelEvalInit:
    """Construction and config normalisation."""

    def test_init_basic(self):
        config = {
            "models": [{"name": "gpt-4o-mini", "type": "llm"}],
            "prompt": "Classify: {content}",
            "output_dir": "./test_output",
            "use_case": "test classification",
        }
        data = [
            {"content": "Great!", "label": "positive"},
            {"content": "Terrible!", "label": "negative"},
        ]
        eval_ = ModelEval(config=config, data=data)

        assert len(eval_.models) == 1
        assert eval_.models[0].name == "gpt-4o-mini"
        assert eval_.prompt_template == config["prompt"]
        assert len(eval_.data) == 2
        assert eval_.use_case == "test classification"

    def test_init_default_use_case(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[])
        assert eval_.use_case == "model evaluation"

    def test_init_default_temperature_and_output_dir(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[])
        assert eval_.temperature == 0.0
        assert eval_.output_dir is None

    def test_dict_config_normalised_to_typed(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[])
        assert isinstance(eval_.config, ModelEvalConfig)

    def test_typed_config_accepted_directly(self):
        config = ModelEvalConfig(
            models=[LLMModelConfig(name="gpt-4o-mini")],
            prompt="Extract: {content}",
        )
        eval_ = ModelEval(config=config, data=[])
        assert eval_.config is config

    # --- label mode specifics ---

    def test_label_mode_no_response_format(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[])
        assert eval_.response_format is None
        assert eval_.decomposed_evaluator is None

    # --- extraction mode specifics ---

    def test_extraction_mode_response_format_stored(self):
        eval_ = ModelEval(config=EXTRACT_CONFIG, data=[], response_format=SampleSchema)
        assert eval_.response_format is SampleSchema

    def test_extraction_mode_decomposed_evaluator_created(self):
        eval_ = ModelEval(config=EXTRACT_CONFIG, data=[], response_format=SampleSchema)
        assert eval_.decomposed_evaluator is not None

    # --- guards ---

    def test_structured_manipulation_without_response_format_raises(self):
        for manip in STRUCTURED_MANIPULATIONS:
            config = {
                "models": [{"name": "gpt-4o-mini", "prompt_manipulation": [manip.value]}],
                "prompt": "Extract: {content}",
            }
            with pytest.raises(ValueError, match=manip.value):
                ModelEval(config=config, data=[])

    def test_decompose_with_response_format_ok(self):
        config = {
            "models": [{"name": "gpt-4o-mini", "prompt_manipulation": ["decompose"]}],
            "prompt": "Extract: {content}",
        }
        eval_ = ModelEval(config=config, data=[], response_format=SampleSchema)
        assert eval_.response_format is SampleSchema

    def test_transformer_with_response_format_raises(self):
        config = {
            "models": [{"label": "my-transformer", "type": "transformer", "model_path": "./dummy"}],
            "prompt": "Extract: {content}",
        }
        with pytest.raises(ValueError, match="Transformer"):
            ModelEval(config=config, data=[], response_format=SampleSchema)

    def test_prompt_missing_placeholder_raises(self):
        with pytest.raises(ValidationError, match="content"):
            ModelEval(
                config={"models": [{"name": "gpt-4o-mini"}], "prompt": "No placeholder."},
                data=[],
            )

    def test_unknown_config_key_raises(self):
        with pytest.raises(ValidationError):
            ModelEval(config={**EXTRACT_CONFIG, "unknown_key": True}, data=[])


# ===========================================================================
# ModelEvalConfig validation
# ===========================================================================

class TestModelEvalConfig:

    def test_direct_construction(self):
        config = ModelEvalConfig(
            models=[LLMModelConfig(name="gpt-4o-mini")],
            prompt="Classify: {content}",
        )
        assert config.models[0].name == "gpt-4o-mini"
        assert config.output_dir is None

    def test_missing_models_raises(self):
        with pytest.raises(ValidationError):
            ModelEvalConfig.model_validate({"prompt": "Classify: {content}"})

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            ModelEvalConfig.model_validate({"models": [{"name": "gpt-4o-mini"}]})

    def test_prompt_without_placeholder_raises(self):
        with pytest.raises(ValidationError, match="content"):
            ModelEvalConfig.model_validate({
                "models": [{"name": "gpt-4o-mini"}],
                "prompt": "No placeholder here.",
            })

    def test_unknown_key_raises(self):
        with pytest.raises(ValidationError):
            ModelEvalConfig.model_validate({
                "models": [{"name": "gpt-4o-mini"}],
                "prompt": "Classify: {content}",
                "unknown_field": "oops",
            })

    def test_manipulation_strings_coerced_to_enum(self):
        config = ModelEvalConfig.model_validate({
            "models": [{"name": "gpt-4o-mini", "prompt_manipulation": ["few_shot"]}],
            "prompt": "Classify: {content}",
        })
        assert config.models[0].prompt_manipulation == [Manipulation.few_shot]

    def test_model_prompt_override_accepted(self):
        config = ModelEvalConfig.model_validate({
            "models": [{"name": "gpt-4o-mini", "prompt": "Custom: {content}"}],
            "prompt": "Base: {content}",
        })
        assert config.models[0].prompt == "Custom: {content}"

    def test_model_prompt_override_without_placeholder_raises(self):
        with pytest.raises(ValidationError, match="content"):
            ModelEvalConfig.model_validate({
                "models": [{"name": "gpt-4o-mini", "prompt": "No placeholder here."}],
                "prompt": "Base: {content}",
            })

    def test_model_prompt_none_by_default(self):
        config = ModelEvalConfig.model_validate({
            "models": [{"name": "gpt-4o-mini"}],
            "prompt": "Base: {content}",
        })
        assert config.models[0].prompt is None


# ===========================================================================
# STRUCTURED_MANIPULATIONS and requires_response_format
# ===========================================================================

class TestStructuredManipulations:

    def test_structured_set_contains_exactly_three(self):
        assert Manipulation.decompose in STRUCTURED_MANIPULATIONS
        assert Manipulation.hallucination_filter in STRUCTURED_MANIPULATIONS
        assert Manipulation.multi_pass in STRUCTURED_MANIPULATIONS
        assert len(STRUCTURED_MANIPULATIONS) == 3

    def test_universal_manipulations_not_in_structured_set(self):
        assert Manipulation.few_shot not in STRUCTURED_MANIPULATIONS
        assert Manipulation.explanation not in STRUCTURED_MANIPULATIONS
        assert Manipulation.prompt_repetition not in STRUCTURED_MANIPULATIONS
        assert Manipulation.prompt_repetition_x3 not in STRUCTURED_MANIPULATIONS

    def test_requires_response_format_property_structured(self):
        assert Manipulation.decompose.requires_response_format is True
        assert Manipulation.hallucination_filter.requires_response_format is True
        assert Manipulation.multi_pass.requires_response_format is True

    def test_requires_response_format_property_universal(self):
        assert Manipulation.few_shot.requires_response_format is False
        assert Manipulation.explanation.requires_response_format is False
        assert Manipulation.prompt_repetition.requires_response_format is False
        assert Manipulation.prompt_repetition_x3.requires_response_format is False


# ===========================================================================
# Field metrics config
# ===========================================================================

class TestGetFieldMetricsConfig:

    def test_returns_none_for_plain_text_labels(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[{"content": "T", "label": "positive"}])
        assert eval_._get_field_metrics_config() is None

    def test_returns_none_for_empty_data(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[])
        assert eval_._get_field_metrics_config() is None

    def test_auto_generates_config_for_json_labels(self):
        eval_ = ModelEval(
            config=EXTRACT_CONFIG,
            data=[{"content": "T", "label": '{"name": "John", "age": 30}'}],
        )
        result = eval_._get_field_metrics_config()
        assert result is not None
        assert "name" in result.config["fields"]
        assert "age" in result.config["fields"]

    def test_explicit_config_takes_precedence(self):
        explicit = {
            "config": {
                "type": "object",
                "metric_config": {"propagation": "weighted_avg"},
                "fields": {"label": {"type": "leaf", "metric_config": {"metric": "exact"}}},
            }
        }
        config = {**CLASSIFY_CONFIG, "field_metrics_config": explicit}
        eval_ = ModelEval(config=config, data=[{"content": "T", "label": '{"label": "pos"}'}])
        result = eval_._get_field_metrics_config()
        assert result is not None
        assert result.config == explicit["config"]


# ===========================================================================
# Data loading
# ===========================================================================

class TestLoadDocumentsAndLabels:

    def test_ids_and_values(self):
        data = [
            {"id": "doc-1", "content": "Great!", "label": "positive"},
            {"content": "Bad!", "label": "negative"},  # no ID
        ]
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=data)
        docs, labels = eval_._load_documents_and_labels()

        assert docs[0].id == "doc-1"
        assert docs[1].id == "doc_1"
        assert labels[0].value == "positive"
        assert labels[0].document_id == "doc-1"
        assert labels[1].document_id == "doc_1"


# ===========================================================================
# Prompt preparation
# ===========================================================================

class TestPrepareModelPrompts:

    @pytest.mark.asyncio
    async def test_no_manipulation_returns_original_prompt(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[])
        prompts = await eval_._prepare_model_prompts()
        assert prompts["gpt-4o-mini"] == "Classify: {content}"

    @pytest.mark.asyncio
    async def test_transformer_gets_empty_prompt(self):
        config = {
            "models": [{"label": "my-t", "type": "transformer", "model_path": "./x"}],
            "prompt": "Classify: {content}",
        }
        eval_ = ModelEval(config=config, data=[])
        prompts = await eval_._prepare_model_prompts()
        assert prompts["my-t"] == ""

    @pytest.mark.asyncio
    async def test_explanation_manipulation(self):
        config = {
            "models": [{"name": "gpt-4o-mini", "prompt_manipulation": ["explanation"]}],
            "prompt": "Classify: {content}",
        }
        eval_ = ModelEval(config=config, data=[])
        with patch.object(
            eval_.enhancer, "optimize", new_callable=AsyncMock,
            return_value={"enhanced_prompt": "Enhanced: {content}"},
        ):
            prompts = await eval_._prepare_model_prompts()
        assert prompts["gpt-4o-mini"] == "Enhanced: {content}"

    @pytest.mark.asyncio
    async def test_few_shot_manipulation_injects_examples(self):
        config = {
            "models": [{"name": "gpt-4o-mini", "prompt_manipulation": ["few_shot"]}],
            "prompt": "Classify: {content}",
        }
        eval_ = ModelEval(config=config, data=[])
        eval_.few_shot_examples = [
            {"document": "Great!", "label": "positive"},
            {"document": "Bad!", "label": "negative"},
        ]
        prompts = await eval_._prepare_model_prompts()
        assert "examples" in prompts["gpt-4o-mini"].lower()

    @pytest.mark.asyncio
    async def test_model_override_prompt_used_as_base(self):
        config = {
            "models": [{"name": "gpt-4o-mini", "prompt": "Override: {content}"}],
            "prompt": "Base: {content}",
        }
        eval_ = ModelEval(config=config, data=[])
        prompts = await eval_._prepare_model_prompts()
        assert prompts["gpt-4o-mini"] == "Override: {content}"

    @pytest.mark.asyncio
    async def test_model_without_override_uses_base_prompt(self):
        config = {
            "models": [
                {"name": "gpt-4o-mini"},
                {"name": "gpt-4o", "prompt": "Override: {content}"},
            ],
            "prompt": "Base: {content}",
        }
        eval_ = ModelEval(config=config, data=[])
        prompts = await eval_._prepare_model_prompts()
        assert prompts["gpt-4o-mini"] == "Base: {content}"
        assert prompts["gpt-4o"] == "Override: {content}"

    @pytest.mark.asyncio
    async def test_override_prompts_tracked_in_attribute(self):
        config = {
            "models": [
                {"name": "gpt-4o-mini"},
                {"name": "gpt-4o", "prompt": "Override: {content}"},
            ],
            "prompt": "Base: {content}",
        }
        eval_ = ModelEval(config=config, data=[])
        await eval_._prepare_model_prompts()
        assert eval_._model_override_prompts == {"gpt-4o": "Override: {content}"}

    @pytest.mark.asyncio
    async def test_manipulation_applied_on_top_of_override_prompt(self):
        config = {
            "models": [{"name": "gpt-4o-mini", "prompt": "Override: {content}", "prompt_manipulation": ["prompt_repetition"]}],
            "prompt": "Base: {content}",
        }
        eval_ = ModelEval(config=config, data=[])
        prompts = await eval_._prepare_model_prompts()
        assert prompts["gpt-4o-mini"].startswith("Override: {content}")
        assert "Override: {content}" in prompts["gpt-4o-mini"].split("Let me repeat")[0]


# ===========================================================================
# Response validator (label mode)
# ===========================================================================

class TestCreateResponseValidator:

    def test_json_labels_produce_validator(self):
        config = {
            "models": [{"name": "gpt-4o-mini"}],
            "prompt": 'Output JSON: {"name": "", "age": 0}. {content}',
        }
        eval_ = ModelEval(config=config, data=[{"content": "T", "label": '{"name": "J", "age": 30}'}])
        validator = eval_._create_response_validator()
        assert validator is not None
        assert "name" in validator.model_fields
        assert "age" in validator.model_fields

    def test_plain_text_labels_return_none(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[{"content": "T", "label": "positive"}])
        assert eval_._create_response_validator() is None

    def test_explanation_with_json_schema_in_prompt_adds_field(self):
        config = {
            "models": [{"name": "gpt-4o-mini"}],
            "prompt": 'Extract: {"name": ""} from {content}',
        }
        eval_ = ModelEval(config=config, data=[{"content": "T", "label": '{"name": "J"}'}])
        validator = eval_._create_response_validator(include_explanation=True)
        assert validator is not None
        assert "explanation" in validator.model_fields


# ===========================================================================
# Transformer evaluation (label mode)
# ===========================================================================

class TestEvaluateTransformer:

    @pytest.mark.asyncio
    async def test_evaluate_transformer_basic(self, tmp_path):
        config = {
            "models": [{"label": "my-t", "type": "transformer", "model_path": "./models"}],
            "prompt": "Classify: {content}",
            "output_dir": str(tmp_path),
        }
        data = [{"id": "d1", "content": "Great!", "label": '{"sentiment": "positive"}'}]
        eval_ = ModelEval(config=config, data=data)
        docs, _ = eval_._load_documents_and_labels()

        with patch("valtron_core.recipes.model_eval.TransformerModelWrapper") as MockW:
            mock_w = Mock()
            mock_w.predict = Mock(return_value='{"sentiment": "positive"}')
            MockW.return_value = mock_w
            result = await eval_._evaluate_transformer(eval_.models[0], docs, None)

        assert isinstance(result, EvaluationResult)
        assert len(result.predictions) == 1
        assert result.predictions[0].cost == 0.0


# ===========================================================================
# Run evaluations
# ===========================================================================

class TestRunEvaluations:

    @pytest.mark.asyncio
    async def test_llm_model_evaluated(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path)}
        eval_ = ModelEval(config=config, data=[{"id": "d1", "content": "T", "label": "pos"}])
        model_prompts = {"gpt-4o-mini": "Classify: {content}"}

        with patch.object(eval_.runner, "evaluate", new_callable=AsyncMock, return_value=_mock_result()):
            results, manipulations = await eval_._run_evaluations(model_prompts)

        assert len(results) == 1
        assert "gpt-4o-mini" in manipulations

    @pytest.mark.asyncio
    async def test_extraction_mode_basic(self, tmp_path):
        config = {**EXTRACT_CONFIG, "output_dir": str(tmp_path)}
        data = [{"id": "d1", "content": "text", "label": '{"name": "x", "value": "y"}'}]
        eval_ = ModelEval(config=config, data=data, response_format=SampleSchema)
        model_prompts = {"gpt-4o-mini": "Extract: {content}"}

        with patch.object(eval_.runner, "evaluate", new_callable=AsyncMock, return_value=_mock_result("gpt-4o-mini", "Extract: {content}")):
            results, manipulations = await eval_._run_evaluations(model_prompts)

        assert len(results) == 1
        assert "gpt-4o-mini" in manipulations


# ===========================================================================
# Save / run integration
# ===========================================================================

class TestSaveExperimentResults:

    def test_save_creates_expected_files(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path), "use_case": "test"}
        eval_ = ModelEval(config=config, data=[{"content": "T", "label": "pos"}])

        mock_r = EvaluationResult(run_id="r", model="gpt-4o-mini", prompt_template="Classify: {content}", status="completed")
        mock_r.predictions = []
        eval_.results = [mock_r]
        eval_._manipulations_applied = {"gpt-4o-mini": []}
        eval_._model_prompts = {"gpt-4o-mini": "Classify: {content}"}

        run_dir = eval_.save_experiment_results()

        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "models" / "gpt-4o-mini.json").exists()

        with open(run_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["use_case"] == "test"
        assert meta["original_prompt"] == "Classify: {content}"

    def test_save_persists_override_prompt(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path)}
        eval_ = ModelEval(config=config, data=[{"content": "T", "label": "pos"}])

        mock_r = EvaluationResult(run_id="r", model="gpt-4o-mini", prompt_template="Override: {content}", status="completed")
        mock_r.predictions = []
        eval_.results = [mock_r]
        eval_._manipulations_applied = {"gpt-4o-mini": []}
        eval_._model_prompts = {"gpt-4o-mini": "Override: {content}"}
        eval_._model_override_prompts = {"gpt-4o-mini": "Override: {content}"}

        run_dir = eval_.save_experiment_results()

        with open(run_dir / "models" / "gpt-4o-mini.json") as f:
            model_data = json.load(f)
        assert model_data["override_prompt"] == "Override: {content}"

    def test_save_omits_override_prompt_when_not_set(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path)}
        eval_ = ModelEval(config=config, data=[{"content": "T", "label": "pos"}])

        mock_r = EvaluationResult(run_id="r", model="gpt-4o-mini", prompt_template="Classify: {content}", status="completed")
        mock_r.predictions = []
        eval_.results = [mock_r]
        eval_._manipulations_applied = {"gpt-4o-mini": []}
        eval_._model_prompts = {"gpt-4o-mini": "Classify: {content}"}
        eval_._model_override_prompts = {}

        run_dir = eval_.save_experiment_results()

        with open(run_dir / "models" / "gpt-4o-mini.json") as f:
            model_data = json.load(f)
        assert model_data.get("override_prompt") is None


class TestSaveHtmlReportFromMemory:

    @pytest.mark.asyncio
    async def test_save_html_without_prior_save_experiment_results(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path)}
        eval_ = ModelEval(config=config, data=[{"id": "d1", "content": "T", "label": "pos"}])

        mock_r = _mock_result()
        eval_.results = [mock_r]
        eval_._manipulations_applied = {"gpt-4o-mini": []}
        eval_._model_prompts = {"gpt-4o-mini": "Classify: {content}"}
        eval_._model_override_prompts = None

        with patch.object(eval_.runner, "generate_report", return_value=tmp_path / "evaluation_report.html") as mock_gen:
            report_path = eval_.save_html_report()

        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["results"] == [mock_r]
        assert call_kwargs["generate_pdf"] is False
        assert call_kwargs["use_case"] == eval_.use_case

    def test_save_html_before_evaluate_raises(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path)}
        eval_ = ModelEval(config=config, data=[{"content": "T", "label": "pos"}])
        with pytest.raises(RuntimeError, match="evaluate()"):
            eval_.save_html_report()


class TestRun:

    @pytest.mark.asyncio
    async def test_run_returns_report_path(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path)}
        eval_ = ModelEval(config=config, data=[{"id": "d1", "content": "T", "label": "pos"}])

        with patch.object(eval_.runner, "evaluate", new_callable=AsyncMock, return_value=_mock_result()):
            with patch.object(eval_.runner, "generate_report", return_value=tmp_path / "report.html"):
                report_path = await eval_.arun()

        assert report_path == tmp_path / "report.html"

    @pytest.mark.asyncio
    async def test_run_with_few_shot_generation(self, tmp_path):
        config = {
            **CLASSIFY_CONFIG,
            "output_dir": str(tmp_path),
            "few_shot": {"enabled": True, "generator_model": "gpt-4o-mini", "num_examples": 2},
        }
        eval_ = ModelEval(config=config, data=[{"id": "d1", "content": "T", "label": "pos"}])

        with patch("valtron_core.recipes.model_eval.FewShotTrainingDataGenerator") as MockGen:
            mock_gen = Mock()
            mock_gen.generate_and_validate_examples = AsyncMock(return_value={
                "examples": [{"document": "D", "label": "pos", "consensus": "correct"}],
                "costs": {"total_cost": 0.01},
            })
            MockGen.return_value = mock_gen

            with patch.object(eval_.runner, "evaluate", new_callable=AsyncMock, return_value=_mock_result()):
                with patch.object(eval_.runner, "generate_report", new_callable=AsyncMock, return_value=tmp_path / "r.html"):
                    await eval_.arun()

        assert len(eval_.few_shot_examples) > 0

    @pytest.mark.asyncio
    async def test_run_pdf_only_returns_run_dir(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path), "output_formats": ["pdf"]}
        eval_ = ModelEval(config=config, data=[{"id": "d1", "content": "T", "label": "pos"}])

        with patch.object(eval_.runner, "evaluate", new_callable=AsyncMock, return_value=_mock_result()):
            with patch.object(eval_.runner, "generate_report", return_value=tmp_path / "report.html") as mock_report:
                report_path = await eval_.arun()

        mock_report.assert_called_once_with(**{**mock_report.call_args.kwargs, "generate_pdf": True})
        assert report_path != tmp_path / "report.html"

    @pytest.mark.asyncio
    async def test_run_both_formats_calls_html_and_pdf(self, tmp_path):
        config = {**CLASSIFY_CONFIG, "output_dir": str(tmp_path), "output_formats": ["html", "pdf"]}
        eval_ = ModelEval(config=config, data=[{"id": "d1", "content": "T", "label": "pos"}])

        with patch.object(eval_.runner, "evaluate", new_callable=AsyncMock, return_value=_mock_result()):
            with patch.object(eval_.runner, "generate_report", return_value=tmp_path / "report.html") as mock_report:
                report_path = await eval_.arun()

        assert mock_report.call_count == 2
        generate_pdf_values = [c.kwargs["generate_pdf"] for c in mock_report.call_args_list]
        assert False in generate_pdf_values
        assert True in generate_pdf_values
        assert report_path == tmp_path / "report.html"


# ===========================================================================
# add_models
# ===========================================================================

class TestAddModels:

    def test_add_valid_model(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[{"content": "T", "label": "pos"}])
        original_count = len(eval_.models)
        eval_.add_models([{"name": "gpt-4o", "label": "GPT-4o"}])
        assert len(eval_.models) == original_count + 1
        assert eval_.models[-1].name == "gpt-4o"
        assert eval_.models[-1].label == "GPT-4o"

    def test_add_dict_input_normalized_to_llm_model_config(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[{"content": "T", "label": "pos"}])
        eval_.add_models([{"name": "gpt-4o"}])
        assert isinstance(eval_.models[-1], LLMModelConfig)

    def test_add_duplicate_label_raises(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[{"content": "T", "label": "pos"}])
        with pytest.raises(ValueError, match="already exists"):
            eval_.add_models([{"name": "gpt-4o-mini"}])

    def test_add_duplicate_within_batch_raises(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[{"content": "T", "label": "pos"}])
        with pytest.raises(ValueError, match="Duplicate label"):
            eval_.add_models([{"name": "gpt-4o", "label": "new"}, {"name": "gpt-4o-2", "label": "new"}])

    def test_add_structured_manip_without_response_format_raises(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[{"content": "T", "label": "pos"}])
        with pytest.raises(ValueError, match="response_format"):
            eval_.add_models([{"name": "gpt-4o", "label": "new", "prompt_manipulation": ["decompose"]}])

    def test_add_updates_config_models(self):
        eval_ = ModelEval(config=CLASSIFY_CONFIG, data=[{"content": "T", "label": "pos"}])
        eval_.add_models([{"name": "gpt-4o", "label": "GPT-4o"}])
        config_labels = [m.label or m.name for m in eval_.config.models]
        assert "GPT-4o" in config_labels


# ===========================================================================
# load_experiment_results
# ===========================================================================

def _write_mock_run_dir(tmp_path, override_prompt=None):
    """Helper: write a minimal run directory and return its path."""
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)

    metadata = {
        "timestamp": "20260420_120000",
        "use_case": "test classification",
        "original_prompt": "Classify: {content}",
        "field_config": None,
        "documents": [
            {"id": "d1", "content": "Hello", "label": "positive"},
            {"id": "d2", "content": "Bye", "label": "negative"},
        ],
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    model_data = {
        "run_id": "abc-123",
        "model": "gpt-4o-mini",
        "started_at": "2026-04-20 12:00:00",
        "completed_at": "2026-04-20 12:00:05",
        "status": "completed",
        "prompt_template": "Classify: {content}",
        "prompt_manipulations": [],
        "override_prompt": override_prompt,
        "llm_config": {"model": "gpt-4o-mini", "temperature": 0.0},
        "metrics": {
            "total_documents": 2, "correct_predictions": 2, "accuracy": 1.0,
            "average_example_score": 1.0, "total_cost": 0.001, "total_time": 2.0,
            "average_cost_per_document": 0.0005, "average_time_per_document": 1.0,
            "model": "gpt-4o-mini",
        },
        "predictions": [
            {"document_id": "d1", "predicted_value": "positive", "original_cost": 0.0005,
             "cost": 0.0005, "response_time": 1.0, "is_correct": True, "example_score": 1.0},
            {"document_id": "d2", "predicted_value": "negative", "original_cost": 0.0005,
             "cost": 0.0005, "response_time": 1.0, "is_correct": True, "example_score": 1.0},
        ],
    }
    with open(models_dir / "gpt-4o-mini.json", "w") as f:
        json.dump(model_data, f)

    return run_dir


class TestLoadExperimentResults:

    def test_load_restores_results(self, tmp_path):
        run_dir = _write_mock_run_dir(tmp_path)
        loaded = ModelEval.load_experiment_results(run_dir)

        assert len(loaded.results) == 1
        assert loaded.results[0].model == "gpt-4o-mini"
        assert len(loaded.results[0].predictions) == 2

    def test_load_restores_model_prompts(self, tmp_path):
        run_dir = _write_mock_run_dir(tmp_path)
        loaded = ModelEval.load_experiment_results(run_dir)
        assert loaded._model_prompts["gpt-4o-mini"] == "Classify: {content}"

    def test_load_restores_data(self, tmp_path):
        run_dir = _write_mock_run_dir(tmp_path)
        loaded = ModelEval.load_experiment_results(run_dir)
        assert len(loaded.data) == 2
        assert loaded.data[0]["id"] == "d1"
        assert loaded.data[0]["label"] == "positive"

    def test_load_restores_override_prompt(self, tmp_path):
        run_dir = _write_mock_run_dir(tmp_path, override_prompt="Custom: {content}")
        loaded = ModelEval.load_experiment_results(run_dir)
        assert loaded._model_override_prompts is not None
        assert loaded._model_override_prompts.get("gpt-4o-mini") == "Custom: {content}"

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            ModelEval.load_experiment_results(tmp_path / "nonexistent")

    def test_load_empty_models_dir_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "models").mkdir(parents=True)
        metadata = {
            "use_case": "test", "original_prompt": "Classify: {content}",
            "field_config": None, "documents": [],
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        with pytest.raises(ValueError, match="No model result files"):
            ModelEval.load_experiment_results(run_dir)

    def test_load_populates_expected_values_from_labels(self, tmp_path):
        run_dir = _write_mock_run_dir(tmp_path)
        loaded = ModelEval.load_experiment_results(run_dir)
        preds = {p.document_id: p for p in loaded.results[0].predictions}
        assert preds["d1"].expected_value == "positive"
        assert preds["d2"].expected_value == "negative"

    @pytest.mark.asyncio
    async def test_load_then_add_model_runs_only_new(self, tmp_path):
        run_dir = _write_mock_run_dir(tmp_path)
        loaded = ModelEval.load_experiment_results(run_dir)
        loaded.add_models([{"name": "gpt-4o", "label": "gpt-4o-new"}])

        new_result = _mock_result("gpt-4o-new", "Classify: {content}")
        with patch.object(loaded.runner, "evaluate", new_callable=AsyncMock, return_value=new_result) as mock_eval:
            await loaded.aevaluate()

        assert mock_eval.call_count == 1
        assert len(loaded.results) == 2
        labels = {r.model for r in loaded.results}
        assert "gpt-4o-mini" in labels
        assert "gpt-4o-new" in labels


# ===========================================================================
# Incremental evaluation (aevaluate skips already-evaluated models)
# ===========================================================================

class TestIncrementalEvaluation:

    @pytest.mark.asyncio
    async def test_skips_already_evaluated_models(self, tmp_path):
        config = {
            "models": [{"name": "gpt-4o-mini"}, {"name": "gpt-4o", "label": "gpt-4o-new"}],
            "prompt": "Classify: {content}",
        }
        eval_ = ModelEval(config=config, data=[{"id": "d1", "content": "T", "label": "pos"}])

        # Pre-populate results for the first model only
        eval_.results = [_mock_result("gpt-4o-mini")]
        eval_._manipulations_applied = {"gpt-4o-mini": []}
        eval_._model_prompts = {"gpt-4o-mini": "Classify: {content}", "gpt-4o-new": "Classify: {content}"}

        new_result = _mock_result("gpt-4o-new")
        with patch.object(eval_.runner, "evaluate", new_callable=AsyncMock, return_value=new_result) as mock_eval:
            await eval_.aevaluate()

        assert mock_eval.call_count == 1
        assert len(eval_.results) == 2

    @pytest.mark.asyncio
    async def test_merges_existing_and_new_results(self, tmp_path):
        config = {
            "models": [{"name": "gpt-4o-mini"}, {"name": "gpt-4o", "label": "gpt-4o-new"}],
            "prompt": "Classify: {content}",
        }
        eval_ = ModelEval(config=config, data=[{"id": "d1", "content": "T", "label": "pos"}])
        eval_.results = [_mock_result("gpt-4o-mini")]
        eval_._manipulations_applied = {"gpt-4o-mini": []}
        eval_._model_prompts = {"gpt-4o-mini": "Classify: {content}", "gpt-4o-new": "Classify: {content}"}

        new_result = _mock_result("gpt-4o-new")
        with patch.object(eval_.runner, "evaluate", new_callable=AsyncMock, return_value=new_result):
            await eval_.aevaluate()

        assert {r.model for r in eval_.results} == {"gpt-4o-mini", "gpt-4o-new"}
        assert "gpt-4o-mini" in eval_._manipulations_applied
        assert "gpt-4o-new" in eval_._manipulations_applied

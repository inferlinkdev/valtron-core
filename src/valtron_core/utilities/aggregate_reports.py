"""Aggregate saved evaluation run directories into a final HTML report.

Usage:
    # New format (run directory with metadata.json + models/):
    python examples/aggregate_reports.py --input comparison_results/ --output-dir reports/final

    # Convert a legacy input to the new run directory format, then generate a report:
    python examples/aggregate_reports.py --input evaluation_results_20260213.json \\
        --output-dir reports/final --convert

    # Re-evaluate with a custom field metrics config:
    python examples/aggregate_reports.py --input comparison_results/ --output-dir reports/final \\
        --field-metrics-config path/to/field_metrics_config.json

This script will:
- Read a run directory (metadata.json + models/*.json).
- Optionally re-evaluate all predictions with a new field_metrics_config (no new API calls).
- Generate visualizations and a single aggregated HTML report into --output-dir.

Legacy inputs (old single combined JSON or old per-model directory) are also supported
via load_legacy_results(). Pass --convert to also write them out in the new format.
Delete load_legacy_results() and convert_legacy_to_run_dir() to drop backwards compatibility.
"""

import argparse
import asyncio
import json
from pathlib import Path
from valtron_core.evaluation.json_eval import EvalResult, JsonEvaluator
from valtron_core.cost_utils import _parse_time_unit_to_seconds, _get_fallback_rate_info, _fallback_cost
from valtron_core.loader import DocumentLoader
from valtron_core.models import EvaluationResult, EvaluationMetrics, FieldMetricsConfig, PredictionResult
from valtron_core.evaluation.json_eval import ExpensiveListComparisonError
from valtron_core.runner import EvaluationRunner, save_run_dir


def _apply_cost_rates(results: list[EvaluationResult]) -> None:
    """Apply cost_rate or fallback imputed rates to predictions, then recompute metrics.

    Mirrors the logic in the evaluator so that loaded results (legacy or run-dir)
    reflect the correct costs even when the saved per-prediction cost values are 0.
    """
    for result in results:
        llm_config = result.llm_config or {}
        cost_rate = llm_config.get("cost_rate")
        if cost_rate is not None:
            time_unit_str = llm_config.get("cost_rate_time_unit", "1hr")
            unit_seconds = _parse_time_unit_to_seconds(time_unit_str)
            for p in result.predictions:
                p.cost = float(cost_rate) * (p.response_time / unit_seconds)
        elif all(p.cost == 0.0 for p in result.predictions):
            fallback = _get_fallback_rate_info(llm_config.get("model", result.model))
            if fallback:
                for p in result.predictions:
                    p.cost = _fallback_cost(llm_config.get("model", result.model), p.response_time)
                result.llm_config = {**llm_config, **fallback}
        result.compute_metrics()


def load_results_from_run_dir(input_dir: Path) -> tuple[list[EvaluationResult], dict]:
    """Load results from new-format run directory (metadata.json + models/)."""
    with open(input_dir / "metadata.json") as f:
        meta = json.load(f)

    label_map = {d["id"]: d["label"] for d in meta.get("documents", [])}
    use_case = meta.get("use_case")
    original_prompt = meta.get("original_prompt")
    field_config = meta.get("field_config")

    results = []
    prompt_optimizations = {}
    model_prompts = {}
    models_dir = input_dir / "models"

    for model_file in sorted(models_dir.glob("*.json")):
        with open(model_file) as f:
            model_data = json.load(f)

        model_name = model_data["model"]
        prompt_manipulations = model_data.get("prompt_manipulations", [])
        prompt_template = model_data.get("prompt_template", "")
        prompt_optimizations[model_name] = prompt_manipulations
        model_prompts[model_name] = prompt_template

        predictions = []
        for p in model_data.get("predictions", []):
            doc_id = p["document_id"]
            field_metrics = None
            if p.get("field_metrics"):
                try:
                    field_metrics = EvalResult.model_validate(p["field_metrics"])
                except Exception:
                    pass
            predictions.append(PredictionResult(
                document_id=doc_id,
                predicted_value=p["predicted_value"],
                expected_value=label_map.get(doc_id, ""),
                is_correct=p.get("is_correct", False),
                example_score=p.get("example_score", 0.0),
                response_time=p.get("response_time", 0.0),
                original_cost=p.get("original_cost", 0.0),
                cost=p.get("cost", 0.0),
                model=model_name,
                field_metrics=field_metrics,
            ))

        result = EvaluationResult(
            run_id=model_data.get("run_id", model_file.stem),
            started_at=model_data.get("started_at"),
            completed_at=model_data.get("completed_at"),
            predictions=predictions,
            metrics=EvaluationMetrics(**model_data["metrics"]) if model_data.get("metrics") else None,
            prompt_template=prompt_template,
            model=model_name,
            llm_config=model_data.get("llm_config", {}),
            field_config=field_config,
            status=model_data.get("status", "completed"),
        )
        if not result.metrics and result.predictions:
            result.compute_metrics()
        results.append(result)

    _apply_cost_rates(results)
    return results, {
        "use_case": use_case,
        "original_prompt": original_prompt,
        "field_config": field_config,
        "prompt_optimizations": prompt_optimizations,
        "model_prompts": model_prompts,
    }


def load_legacy_results(path: Path) -> tuple[list[EvaluationResult], dict]:
    """Legacy loader for old formats. Delete this function to drop backwards compatibility.

    Handles:
      - Single combined JSON file (evaluation_results_*.json)
      - Directory of per-model JSON files (old _save_result_to_dir format)
    """
    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "results" not in data:
            raise ValueError(f"{path} does not look like a combined results file.")
        results = []
        for item in data["results"]:
            er = EvaluationResult.model_validate(item)
            results.append(er)
        _apply_cost_rates(results)
        return results, {
            "use_case": data.get("use_case"),
            "original_prompt": data.get("original_prompt"),
            "field_config": data.get("field_config"),
            "prompt_optimizations": data.get("prompt_manipulations") or data.get("prompt_optimizations"),
            "model_prompts": data.get("model_prompts"),
        }
    else:
        results = DocumentLoader().load_results_from_dir(path)
        _apply_cost_rates(results)
        return results, {}


def _is_legacy_path(path: Path) -> bool:
    """Returns True if path looks like an old-format input (not a new run directory)."""
    if path.is_file():
        return True
    return not (path / "metadata.json").exists()


def convert_legacy_to_run_dir(
    results: list[EvaluationResult],
    meta: dict,
    output_dir: Path,
) -> None:
    """Write legacy results into the new run directory layout.

    Writes output_dir/metadata.json and output_dir/models/{safe_model_name}.json.
    Document content is available when the source was a per-model directory (stored
    in prediction.metadata["content"]); it is absent in the old combined-file format,
    in which case an empty string is written and can be filled in manually.
    """
    # Build deduplicated document list from predictions.
    seen: set[str] = set()
    documents = []
    for result in results:
        for p in result.predictions:
            if p.document_id not in seen:
                seen.add(p.document_id)
                documents.append({
                    "id": p.document_id,
                    "content": p.metadata.get("content", ""),
                    "label": p.expected_value,
                })

    field_config = meta.get("field_config")
    if field_config is None:
        field_config = next((r.field_config for r in results if r.field_config is not None), None)

    save_run_dir(
        output_dir, results, documents,
        use_case=meta.get("use_case"),
        original_prompt=meta.get("original_prompt"),
        field_config=field_config,
        model_prompts=meta.get("model_prompts"),
        prompt_manipulations=meta.get("prompt_optimizations"),
    )
    print(f"Converted {len(results)} model(s) to run directory format at {output_dir}")


def reevaluate_with_field_metrics(results, field_metrics_config: FieldMetricsConfig):
    """Re-evaluate all predictions in-place using a new field_metrics_config.

    This re-runs JsonEvaluator.evaluate() on each prediction's expected/predicted
    values without making any new API calls, then recomputes aggregated metrics.
    """
    evaluator = JsonEvaluator(
        custom_metrics=field_metrics_config.custom_metrics,
        custom_aggs=field_metrics_config.custom_aggs,
    )

    for result in results:
        for pred in result.predictions:
            try:
                eval_result = evaluator.evaluate(
                    field_metrics_config.config,
                    pred.expected_value,
                    pred.predicted_value,
                )
                pred.field_metrics = eval_result
                pred.example_score = eval_result.score
                pred.is_correct = eval_result.is_correct
            except Exception as e:
                print(f"  Warning: failed to re-evaluate prediction for "
                      f"document {pred.document_id}: {e}")

        result.compute_metrics()
        print(f"  Re-evaluated {len(result.predictions)} predictions for model: {result.model}")


async def main(
    input_path: Path,
    output_dir: Path,
    use_case: str,
    include_recommendation: bool,
    force: bool = False,
    convert: bool = False,
    field_metrics_config: FieldMetricsConfig | None = None,
):
    if _is_legacy_path(input_path):
        print(f"Loading legacy results from {input_path}")
        results, meta = load_legacy_results(input_path)
        if convert:
            print(f"Converting to run directory format...")
            convert_legacy_to_run_dir(results, meta, output_dir)
    else:
        print(f"Loading results from run directory {input_path}")
        results, meta = load_results_from_run_dir(input_path)

    if not results:
        print("No valid results found.")
        return

    if use_case == "general purpose" and meta.get("use_case"):
        use_case = meta["use_case"]

    runner = EvaluationRunner()

    if field_metrics_config:
        num_documents = max(len(r.predictions) for r in results) if results else 0
        runner._preflight_check(field_metrics_config, num_documents, len(results))
        print("Re-evaluating all predictions with new field_metrics_config...")
        reevaluate_with_field_metrics(results, field_metrics_config)
        print("Re-evaluation complete.")

    output_html = Path(output_dir) / "evaluation_report.html"
    if output_html.exists() and not force:
        print(f"Report already exists at {output_html}. Use --force to regenerate.")
        return

    report_field_config = field_metrics_config.config if field_metrics_config else meta.get("field_config")

    report_path = await runner.generate_report(
        results=results,
        output_dir=output_dir,
        use_case=use_case,
        include_recommendation=include_recommendation,
        create_visualizations=True,
        original_prompt=meta.get("original_prompt"),
        prompt_optimizations=meta.get("prompt_optimizations"),
        model_prompts=meta.get("model_prompts"),
        field_config=report_field_config,
    )
    print(f"Final report created: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate saved evaluation run directories into a final report.")
    parser.add_argument("--input", required=True,
        help="Run directory (metadata.json + models/) or legacy path (old JSON file or per-model dir)")
    parser.add_argument("--output-dir", required=True, help="Directory to write the aggregated report and charts")
    parser.add_argument("--use-case", default="general purpose", help="Text describing the use case for recommendations")
    parser.add_argument("--no-recommendation", action="store_true", help="Disable LLM-powered recommendation in the final report")
    parser.add_argument("--force", action="store_true", help="Regenerate the report even if it already exists")
    parser.add_argument("--convert", action="store_true",
        help="When the input is a legacy format, also write it out as a run directory "
             "(metadata.json + models/) inside --output-dir")
    parser.add_argument(
        "--field-metrics-config",
        default=None,
        help="Path to a JSON file containing the field_metrics_config. "
             "When provided, all predictions will be re-evaluated using this config "
             "(no new API calls). The JSON should have the structure: "
             '{\"config\": {\"type\": \"object\", \"fields\": {...}}}',
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Define custom metric implementations here.
    def my_metric(expected, actual, _params) -> tuple[float, bool]:
        score = 1.0 if expected.lower() == actual.lower() else 0.0
        return score, score == 1.0

    custom_metrics = {
        "test_custom": my_metric,
    }

    # Load field metrics config from JSON file if provided
    field_metrics_config = None
    if args.field_metrics_config:
        config_path = Path(args.field_metrics_config)
        if not config_path.exists():
            print(f"Error: field_metrics_config file not found: {config_path}")
            exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        if "config" in raw_config:
            field_metrics_config = FieldMetricsConfig(**raw_config, custom_metrics=custom_metrics)
        else:
            field_metrics_config = FieldMetricsConfig(config=raw_config, custom_metrics=custom_metrics)
        print(f"Loaded field_metrics_config from {config_path}")

    try:
        asyncio.run(
            main(
                input_path=input_path,
                output_dir=output_dir,
                use_case=args.use_case,
                include_recommendation=not args.no_recommendation,
                force=args.force,
                convert=args.convert,
                field_metrics_config=field_metrics_config,
            )
        )
    except ExpensiveListComparisonError as e:
        print(f"Error: {e}")
        exit(1)

"""High-level evaluation runner and orchestrator."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union

from litellm import BaseModel
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from valtron_core.client import LLMClient
from valtron_core.evaluator import PromptEvaluator
from valtron_core.evaluation.json_eval import (
    ExpensiveListComparisonError,
    find_expensive_unordered_list_fields,
)
from valtron_core.loader import DocumentLoader
from valtron_core.models import (
    Document,
    EvaluationInput,
    EvaluationMetrics,
    EvaluationResult,
    FieldMetricsConfig,
    Label,
    PredictionResult,
)

console = Console()


def save_run_dir(
    run_dir: "Path | str",
    results: "list[EvaluationResult]",
    documents: "list[dict]",
    *,
    use_case: "str | None" = None,
    original_prompt: "str | None" = None,
    field_config: "dict | None" = None,
    model_prompts: "dict[str, str] | None" = None,
    prompt_manipulations: "dict[str, list[str]] | None" = None,
    model_override_prompts: "dict[str, str] | None" = None,
    response_format_schema: "dict[str, Any] | None" = None,
) -> Path:
    """Write evaluation results to the canonical run directory layout.

    Writes ``run_dir/metadata.json`` (only if it does not already exist) and
    ``run_dir/models/{safe_model_name}.json`` for each result in *results*.

    Args:
        run_dir: Directory to write into.
        results: One or more EvaluationResult objects to save.
        documents: Shared document list for metadata.json. Each entry is a dict
            with keys ``id``, ``content``, ``label``, and optionally ``attachments``.
        use_case: Human-readable description of the evaluation task.
        original_prompt: The base prompt template before any per-model manipulations.
        field_config: Field-level metrics config to embed in metadata.
        model_prompts: Mapping of model name → actual prompt used (post-manipulation).
            Falls back to ``result.prompt_template`` when absent.
        prompt_manipulations: Mapping of model name → list of manipulation names applied.
            Defaults to ``[]`` when absent.
        model_override_prompts: Mapping of model name → per-model override prompt (pre-manipulation).
            Only present when a model defines its own ``prompt`` field in config.

    Returns:
        Resolved Path to *run_dir*.
    """
    run_dir = Path(run_dir)
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "use_case": use_case,
            "original_prompt": original_prompt,
            "field_metrics_config": {"config": field_config} if field_config else None,
            "response_format_schema": response_format_schema,
            "documents": documents,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    for result in results:
        model_name = result.model
        predictions = []
        for p in result.predictions:
            pred_dict: dict[str, Any] = {
                "document_id": p.document_id,
                "predicted_value": p.predicted_value,
                "original_cost": p.original_cost,
                "cost": p.cost,
                "response_time": p.response_time,
                "is_correct": p.is_correct,
                "example_score": p.example_score,
            }
            if p.field_metrics:
                pred_dict["field_metrics"] = p.field_metrics.model_dump()
            predictions.append(pred_dict)

        model_data = {
            "run_id": result.run_id,
            "model": model_name,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "status": result.status,
            "prompt_template": (model_prompts or {}).get(model_name, result.prompt_template),
            "prompt_manipulations": (prompt_manipulations or {}).get(model_name, []),
            "override_prompt": (model_override_prompts or {}).get(model_name),
            "llm_config": result.llm_config,
            "metrics": result.metrics.model_dump() if result.metrics else None,
            "predictions": predictions,
        }
        safe_name = model_name.replace("/", "_")
        out_file = models_dir / f"{safe_name}.json"
        with open(out_file, "w") as f:
            json.dump(model_data, f, indent=2, default=str)

    return run_dir


class EvaluationRunner:
    """High-level runner for prompt evaluation."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """
        Initialize the evaluation runner.

        Args:
            client: Optional LLMClient instance
        """
        self.client = client or LLMClient()
        self.evaluator = PromptEvaluator(client=self.client)
        self.loader = DocumentLoader()

    def _preflight_check(
        self,
        field_metrics_config: "FieldMetricsConfig | dict | None",
        num_documents: int,
        num_models: int,
    ) -> None:
        """Raise ValueError before any model is invoked if the field config
        contains unordered list fields whose item comparison calls a 3rd-party
        API (LLM or embedding).

        Unordered list matching compares every expected item against every
        actual item — n^2 comparisons per list per document.  When each
        comparison calls an external API this can silently rack up a large bill.

        Users can opt in per sub-field by setting ``allow_expensive_comparisons_for``
        on the offending list's metric_config to a list of the specific field paths
        (relative to item_logic) that are allowed to be expensive.  Use ``"$item"``
        for lists of primitives.
        """
        if field_metrics_config is None:
            return

        if isinstance(field_metrics_config, dict):
            config_dict = field_metrics_config
            custom_metric_names: set[str] = set()
        else:
            config_dict = field_metrics_config.config
            custom_metric_names = set(field_metrics_config.custom_metrics.keys())

        issues = find_expensive_unordered_list_fields(config_dict, custom_metric_names)
        if not issues:
            return

        affected_lines = ""
        for issue in issues:
            problem = (
                f"built-in metric 'comparator' using {issue['description']}"
                if issue["type"] == "builtin"
                else issue["description"]
            )
            affected_lines += (
                f"  • List field : '{issue['list_path']}'\n"
                f"    Metric at  : '{issue['metric_path']}'\n"
                f"    Problem    : {problem}\n"
                f"    To allow   : add \"{issue['relative_path']}\" to allow_expensive_comparisons_for "
                f"on the list at '{issue['list_path']}'\n\n"
            )

        example_named = json.dumps(
            {
                "type": "list",
                "metric_config": {
                    "ordered": False,
                    "allow_expensive_comparisons_for": ["name", "zone"],
                },
            },
            indent=2,
        )
        example_item = json.dumps(
            {
                "type": "list",
                "metric_config": {"ordered": False, "allow_expensive_comparisons_for": ["$item"]},
            },
            indent=2,
        )
        console.print(
            f"\n[bold red]Pre-flight check failed:[/bold red] your field config contains unordered list field(s) whose\n"
            f"item comparison calls a 3rd-party service (LLM or embedding API).\n\n"
            f"[bold yellow]Why this is a problem[/bold yellow]\n"
            f"  Unordered list matching works by comparing EVERY expected item against EVERY\n"
            f"  actual item to find the best alignment.  For a list with k items that means\n"
            f"  k\u00b2 comparisons per document.  Each comparison here triggers an external API call.\n"
            f"  With [bold]{num_documents}[/bold] document(s) and [bold]{num_models}[/bold] model(s) being evaluated,\n"
            f"  total API calls \u2248  {num_documents} x k\u00b2 x {num_models}  =  [bold]{num_documents * num_models}[/bold] x k\u00b2\n"
            f"  (where k is the number of items in the list for each document).\n\n"
            f"[bold yellow]Affected fields:[/bold yellow]\n"
            f"{escape(affected_lines)}"
            f"[bold yellow]How to fix this:[/bold yellow]\n"
            f"  [bold]Option 1[/bold] — change the metric to one that runs locally, e.g.:\n"
            f"    metric: [cyan]'exact'[/cyan]                                             (exact string match)\n"
            f"    metric: [cyan]'comparator'[/cyan], element_compare: [cyan]'text_similarity'[/cyan]   (local fuzzy match)\n\n"
            f"  [bold]Option 2[/bold] — accept the cost per field by adding [cyan]allow_expensive_comparisons_for[/cyan] to the\n"
            f"  list's metric_config with only the specific sub-fields you intend to be expensive.\n"
            f'  Each "To allow" line above shows exactly what to add.  Example:\n'
            f"[green]{escape(example_named)}[/green]\n\n"
            f'  Use [cyan]"$item"[/cyan] for a list of primitives (no sub-fields):\n'
            f"[green]{escape(example_item)}[/green]\n"
        )
        raise ExpensiveListComparisonError(
            f"Pre-flight check failed: your field config contains unordered list field(s) whose "
            f"item comparison calls a 3rd-party service (LLM or embedding API). "
            f"Affected fields: {', '.join(issue['list_path'] for issue in issues)}"
        )

    def _save_result_to_run_dir(
        self,
        result: EvaluationResult,
        run_dir: "str | Path",
        doc_content_map: dict,
        doc_label_map: dict,
        doc_attachments_map: "dict | None" = None,
    ) -> None:
        """Save a single model result into the run directory layout."""
        documents = []
        for doc_id, content in doc_content_map.items():
            doc_entry: dict[str, Any] = {
                "id": doc_id,
                "content": content,
                "label": doc_label_map.get(doc_id, ""),
            }
            if doc_attachments_map:
                attachments = doc_attachments_map.get(doc_id, [])
                if attachments:
                    doc_entry["attachments"] = attachments
            documents.append(doc_entry)

        out_dir = save_run_dir(
            run_dir,
            [result],
            documents,
            original_prompt=result.prompt_template,
            field_config=result.field_config,
        )
        safe_name = result.model.replace("/", "_")
        console.print(
            f"[green]Saved predictions for {result.model} -> {out_dir / 'models' / (safe_name + '.json')}[/green]"
        )

    def _load_results_from_run_dir(
        self, run_dir: Path
    ) -> "tuple[list[EvaluationResult], dict[str, Any]]":
        """Load results from new-format run directory (metadata.json + models/)."""
        from valtron_core.evaluation.json_eval import EvalResult

        with open(run_dir / "metadata.json") as f:
            meta = json.load(f)

        label_map = {d["id"]: d["label"] for d in meta.get("documents", [])}
        documents = [
            Document(
                id=d["id"],
                content=d.get("content", ""),
                metadata={},
                attachments=d.get("attachments", []),
            )
            for d in meta.get("documents", [])
        ]

        results = []
        model_prompts: dict = {}
        prompt_optimizations: dict = {}
        model_override_prompts: dict = {}
        models_dir = run_dir / "models"

        for model_file in sorted(models_dir.glob("*.json")):
            with open(model_file) as f:
                model_data = json.load(f)

            model_name = model_data["model"]
            model_prompts[model_name] = model_data.get("prompt_template", "")
            prompt_optimizations[model_name] = model_data.get("prompt_manipulations", [])
            override = model_data.get("override_prompt")
            if override:
                model_override_prompts[model_name] = override

            predictions = []
            for p in model_data.get("predictions", []):
                doc_id = p["document_id"]
                field_metrics = None
                if p.get("field_metrics"):
                    try:
                        field_metrics = EvalResult.model_validate(p["field_metrics"])
                    except Exception:
                        pass
                predictions.append(
                    PredictionResult(
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
                    )
                )

            result_kwargs: dict[str, Any] = {
                "run_id": model_data.get("run_id", model_file.stem),
                "predictions": predictions,
            }
            if model_data.get("started_at"):
                result_kwargs["started_at"] = model_data["started_at"]
            if model_data.get("completed_at"):
                result_kwargs["completed_at"] = model_data["completed_at"]
            result = EvaluationResult(
                **result_kwargs,
                metrics=(
                    EvaluationMetrics(**model_data["metrics"])
                    if model_data.get("metrics")
                    else None
                ),
                prompt_template=model_data.get("prompt_template", ""),
                model=model_name,
                llm_config=model_data.get("llm_config", {}),
                field_config=(meta.get("field_metrics_config") or {}).get("config"),
                status=model_data.get("status", "completed"),
            )
            if not result.metrics and result.predictions:
                result.compute_metrics()
            results.append(result)

        metadata_out = {
            "documents": documents,
            "prompt_optimizations": prompt_optimizations,
            "model_prompts": model_prompts,
            "model_override_prompts": model_override_prompts,
            "original_prompt": meta.get("original_prompt"),
            "use_case": meta.get("use_case", "general purpose"),
            "timestamp": meta.get("timestamp"),
            "field_config": (meta.get("field_metrics_config") or {}).get("config"),
            "response_format_schema": meta.get("response_format_schema"),
        }
        console.print(f"[green]Loaded {len(results)} model results from {run_dir}[/green]")
        return results, metadata_out

    async def evaluate(
        self,
        documents: list[Document],
        labels: list[Label],
        prompt_template: str,
        model: str | dict[str, Any],
        response_format: "type[BaseModel] | dict | None" = None,
        field_metrics_config: FieldMetricsConfig | None = None,
        post_extraction_filter: Callable | None = None,
        multi_pass: int = 1,
        max_concurrent: int = 5,
        save_results_dir: str | Path | None = None,
    ) -> EvaluationResult:
        """
        Run a single-model evaluation on already-loaded documents and labels.

        All litellm parameters (temperature, max_tokens, api_base, etc.) must be
        included in the ``model`` dict.  Pass a plain string for models that need
        no extra params.

        Args:
            documents: List of documents
            labels: List of labels
            prompt_template: Prompt template with {content} placeholder
            model: Model name string, or dict containing 'model' key plus any
                litellm params (temperature, max_tokens, api_base, reasoning_effort, …)
            response_format: Optional pydantic model for response parsing
            field_metrics_config: Configuration for field-level metrics
            post_extraction_filter: Optional async callable applied to each prediction
            multi_pass: Number of evaluation passes (>1 enables multi-pass merging)
            max_concurrent: Maximum concurrent API calls
            save_results_dir: Optional directory to save per-model prediction results

        Returns:
            EvaluationResult
        """
        self._preflight_check(field_metrics_config, len(documents), 1)

        temp = model.get("temperature", 0.0) if isinstance(model, dict) else 0.0
        max_tok = model.get("max_tokens") if isinstance(model, dict) else None
        model_name = model if isinstance(model, str) else model.get("model", "unknown")

        console.print(f"\n[bold blue]Running evaluation with {model_name}...[/bold blue]")

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template=prompt_template,
            model=model,
            temperature=temp,
            max_tokens=max_tok,
        )

        result = await self.evaluator.evaluate(
            eval_input=eval_input,
            max_concurrent=max_concurrent,
            response_format=response_format,
            field_metrics_config=field_metrics_config,
            post_extraction_filter=post_extraction_filter,
            multi_pass=multi_pass,
        )

        if field_metrics_config is not None:
            result.field_config = field_metrics_config.config

        if save_results_dir:
            doc_content_map = {d.id: d.content for d in documents}
            doc_attachments_map = {d.id: d.attachments for d in documents}
            doc_label_map = {p.document_id: p.expected_value for p in result.predictions}
            self._save_result_to_run_dir(
                result, save_results_dir, doc_content_map, doc_label_map, doc_attachments_map
            )

        self._print_result(result)
        return result

    async def evaluate_from_file(
        self,
        data_file: str | Path,
        prompt_template: str,
        models: Union[str, dict[str, Any], list[Union[str, dict[str, Any]]]],
        file_format: str = "json",
        response_format: "type[BaseModel] | dict | None" = None,
        field_metrics_config: FieldMetricsConfig | None = None,
        post_extraction_filter: Callable | None = None,
        multi_pass: int = 1,
        max_concurrent: int = 5,
        max_concurrent_models: int | None = None,
        save_results_dir: str | Path | None = None,
    ) -> list[EvaluationResult]:
        """
        Load data from a file and evaluate one or more models, returning all results.

        Args:
            data_file: Path to data file (JSON or CSV)
            prompt_template: Prompt template with {content} placeholder
            models: One model (string or dict) or a list of models.  All litellm
                params belong in the model dict — no separate temperature/max_tokens.
            file_format: 'json' or 'csv'
            response_format: Optional pydantic model for response parsing
            field_metrics_config: Configuration for field-level metrics
            post_extraction_filter: Optional async callable applied to each prediction
            multi_pass: Number of evaluation passes
            max_concurrent: Maximum concurrent API calls per model
            max_concurrent_models: Maximum number of models to evaluate concurrently.
                When None, all models run in parallel.
            save_results_dir: Optional directory to save per-model prediction results

        Returns:
            List of EvaluationResults (one per model, in input order).
            Single-model callers can do ``result = (await evaluate_from_file(...))[0]``.
        """
        if file_format == "json":
            documents, labels = self.loader.load_combined_from_json(data_file)
        elif file_format == "csv":
            documents, labels = self.loader.load_combined_from_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        console.print(f"[green]Loaded {len(documents)} documents from {data_file}[/green]")

        models_list: list[str | dict[str, Any]] = (
            [models] if isinstance(models, (str, dict)) else list(models)
        )

        self._preflight_check(field_metrics_config, len(documents), len(models_list))

        if len(models_list) > 1:
            console.print(f"\n[bold blue]Evaluating {len(models_list)} models...[/bold blue]")

        semaphore = asyncio.Semaphore(max_concurrent_models) if max_concurrent_models else None

        async def _eval_one(
            index: int, model: str | dict[str, Any]
        ) -> tuple[int, EvaluationResult]:
            async def _run() -> EvaluationResult:
                return await self.evaluate(
                    documents=documents,
                    labels=labels,
                    prompt_template=prompt_template,
                    model=model,
                    response_format=response_format,
                    field_metrics_config=field_metrics_config,
                    post_extraction_filter=post_extraction_filter,
                    multi_pass=multi_pass,
                    max_concurrent=max_concurrent,
                    save_results_dir=save_results_dir,
                )

            if semaphore:
                async with semaphore:
                    result = await _run()
            else:
                result = await _run()
            return index, result

        indexed = await asyncio.gather(*[_eval_one(i, m) for i, m in enumerate(models_list)])
        results = [r for _, r in sorted(indexed, key=lambda x: x[0])]

        if len(models_list) > 1:
            self._print_comparison(results, show_field_metrics=field_metrics_config is not None)

        return results

    def _print_result(self, result: EvaluationResult) -> None:
        """Print evaluation result."""
        if not result.metrics:
            console.print("[red]No metrics available[/red]")
            return

        console.print(f"\n[bold green]Evaluation Complete![/bold green]")
        console.print(f"Run ID: {result.run_id}")
        console.print(f"Model: {result.model}")
        console.print(f"Status: {result.status}")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Total Documents", str(result.metrics.total_documents))
        table.add_row("Correct Predictions", str(result.metrics.correct_predictions))
        table.add_row("Accuracy", f"{result.metrics.accuracy:.2%}")
        table.add_row("Total Cost", f"${result.metrics.total_cost:.6f}")
        table.add_row("Total Time", f"{result.metrics.total_time:.2f}s")
        table.add_row("Avg Time/Doc", f"{result.metrics.average_time_per_document:.2f}s")
        table.add_row("Avg Cost/Doc", f"${result.metrics.average_cost_per_document:.6f}")

        console.print(table)

    def _print_comparison(
        self, results: list[EvaluationResult], show_field_metrics: bool = False
    ) -> None:
        """Print model comparison results."""
        console.print(f"\n[bold green]Model Comparison Complete![/bold green]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", style="yellow")
        table.add_column("Total Cost", style="green")
        table.add_column("Total Time", style="blue")
        table.add_column("Avg Cost/Doc", style="green")
        table.add_column("Avg Time/Doc", style="blue")

        for result in results:
            if not result.metrics:
                continue

            table.add_row(
                result.model,
                f"{result.metrics.accuracy:.2%}",
                f"${result.metrics.total_cost:.6f}",
                f"{result.metrics.total_time:.2f}s",
                f"${result.metrics.average_cost_per_document:.6f}",
                f"{result.metrics.average_time_per_document:.2f}s",
            )

        console.print(table)

        # Print best model
        best_accuracy = max(results, key=lambda r: r.metrics.accuracy if r.metrics else 0)
        best_cost = min(results, key=lambda r: r.metrics.total_cost if r.metrics else float("inf"))
        best_speed = min(results, key=lambda r: r.metrics.total_time if r.metrics else float("inf"))

        console.print("\n[bold]Best Models:[/bold]")
        console.print(f"  🎯 Accuracy: {best_accuracy.model}")
        console.print(f"  💰 Cost: {best_cost.model}")
        console.print(f"  ⚡ Speed: {best_speed.model}")

        # Print field-level metrics if available
        if show_field_metrics:
            self._print_field_metrics_comparison(results)

    def _print_field_metrics_comparison(self, results: list[EvaluationResult]) -> None:
        """Print per-field metrics comparison across models."""
        console.print("\n[bold cyan]Per-Field Metrics:[/bold cyan]")

        # Collect all field names across all results
        all_fields = set()
        for result in results:
            if result.metrics and result.metrics.aggregated_field_metrics:
                all_fields.update(result.metrics.aggregated_field_metrics.keys())

        if not all_fields:
            console.print("[yellow]No field-level metrics available[/yellow]")
            return

        # Print table for each field
        for field_name in sorted(all_fields):
            console.print(f"\n[bold]Field: {field_name}[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Model", style="cyan")
            table.add_column("Precision", style="yellow")
            table.add_column("Recall", style="green")

            for result in results:
                if not result.metrics or not result.metrics.aggregated_field_metrics:
                    continue

                field_metric = result.metrics.aggregated_field_metrics.get(field_name)
                if field_metric:
                    table.add_row(
                        result.model,
                        f"{field_metric.precision:.2%}",
                        f"{field_metric.recall:.2%}",
                    )

            console.print(table)

    def save_results(self, result: EvaluationResult, output_file: str | Path) -> None:
        """
        Save evaluation results to file.

        Args:
            result: EvaluationResult to save
            output_file: Path to output file
        """
        self.loader.save_results_to_json(result, output_file)
        console.print(f"[green]Results saved to {output_file}[/green]")

    def generate_report(
        self,
        results: list[EvaluationResult] | None = None,
        output_dir: str | Path = None,
        use_case: str | None = None,
        include_recommendation: bool = True,
        create_visualizations: bool = True,
        prompt_optimizations: dict[str, list[str]] | None = None,
        model_prompts: dict[str, str] | None = None,
        model_override_prompts: dict[str, str] | None = None,
        original_prompt: str | None = None,
        documents: list[Document] | None = None,
        field_config: dict | None = None,
        output_formats: list[str] | None = None,
    ) -> Path:
        """
        Generate evaluation reports in one or more formats.

        Can work in two modes:
        1. Direct mode: Pass results directly
        2. Run-dir mode: Read results from a run directory (metadata.json + models/)

        Args:
            results: List of evaluation results to include in report (optional if output_dir provided)
            output_dir: Directory to save report and visualizations, or a run directory to load from
            use_case: Description of use case for recommendation
            include_recommendation: Whether to include LLM-powered recommendation
            create_visualizations: Whether to create and embed charts
            prompt_optimizations: Optional dict mapping model names to list of applied prompt manipulations
            model_prompts: Optional dict mapping model names to the actual prompts used
            original_prompt: Optional original prompt template (before any optimizations)
            documents: Optional list of Document objects for detailed analysis display
            output_formats: List of formats to generate — any of "html", "pdf". Defaults to ["html", "pdf"].

        Returns:
            Path to generated HTML report, or PDF report if only "pdf" was requested
        """
        if output_formats is None:
            output_formats = ["html", "pdf"]

        from valtron_core.reports import ReportGenerator

        # Initialize metadata dict
        metadata = {}

        # Determine mode: direct or run-dir
        if output_dir and not results:
            output_dir = Path(output_dir)
            if not (output_dir / "metadata.json").exists():
                raise ValueError(
                    f"No metadata.json found in {output_dir}. Not a valid run directory."
                )
            results, metadata = self._load_results_from_run_dir(output_dir)
            if use_case is None:
                use_case = metadata.get("use_case", "general purpose")
            if prompt_optimizations is None:
                prompt_optimizations = metadata.get("prompt_optimizations")
            if model_prompts is None:
                model_prompts = metadata.get("model_prompts")
            if model_override_prompts is None:
                model_override_prompts = metadata.get("model_override_prompts")
            if original_prompt is None:
                original_prompt = metadata.get("original_prompt")
        elif not results:
            raise ValueError(
                "Must provide either 'results' or 'output_dir' pointing to a valid run directory"
            )

        # Set defaults
        if use_case is None:
            use_case = "general purpose"
        if output_dir is None:
            raise ValueError("output_dir must be provided")

        # Try to extract original_prompt from results if not provided
        if original_prompt is None and results:
            # Use prompt_template from first result
            original_prompt = results[0].prompt_template

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold blue]Generating report...[/bold blue]")

        report_generator = ReportGenerator(client=self.client)

        # Use passed documents, or try to extract from metadata if available
        if documents is None:
            documents = metadata.get("documents")

        # Use passed field_config, fall back to JSON metadata, then to what's stored on the results
        if field_config is None:
            field_config = metadata.get("field_config")
        if field_config is None and results:
            field_config = next(
                (r.field_config for r in results if r.field_config is not None), None
            )

        report_path = None
        recommendation = None

        if "html" in output_formats:
            console.print("[cyan]Generating HTML report...[/cyan]")
            report_path, recommendation = report_generator.generate_html_report(
                results=results,
                output_path=output_dir,
                use_case=use_case,
                include_recommendation=include_recommendation,
                prompt_optimizations=prompt_optimizations,
                model_prompts=model_prompts,
                model_override_prompts=model_override_prompts,
                original_prompt=original_prompt,
                documents=documents,
                field_config=field_config,
            )
            console.print(f"[bold green]HTML report generated: {report_path}[/bold green]")

        if "pdf" in output_formats:
            console.print("[cyan]Generating PDF report...[/cyan]")
            pdf_path = report_generator.generate_pdf_report(
                results=results,
                output_path=output_dir,
                recommendation=recommendation,
                original_prompt=original_prompt,
                prompt_optimizations=prompt_optimizations,
                model_override_prompts=model_override_prompts,
            )
            console.print(f"[bold green]PDF report generated: {pdf_path}[/bold green]")
            if report_path is None:
                report_path = pdf_path

        return report_path

    async def train_and_compare_bert(
        self,
        documents: list[Document],
        labels: list[Label],
        llm_models: list[str],
        prompt_template: str,
        bert_model_name: str = "bert-base-uncased",
        bert_output_dir: str | Path = "./bert_models",
        bert_epochs: int = 3,
        bert_batch_size: int = 8,
        temperature: float = 0.0,
        max_concurrent: int = 5,
    ) -> tuple[list[EvaluationResult], EvaluationResult]:
        """
        Train a BERT model and compare it with LLM models.

        Args:
            documents: Training/evaluation documents
            labels: Document labels
            llm_models: List of LLM models to compare against
            prompt_template: Prompt template for LLM models
            bert_model_name: Pretrained BERT model to use
            bert_output_dir: Directory to save BERT model
            bert_epochs: Number of training epochs
            bert_batch_size: Training batch size
            temperature: Temperature for LLM calls
            max_concurrent: Max concurrent LLM calls

        Returns:
            Tuple of (llm_results, bert_result)
        """
        from valtron_core.bert_evaluator import BERTEvaluator, create_bert_model_for_comparison

        console.print("\n[bold magenta]Training BERT model...[/bold magenta]")

        # Train BERT model
        trainer = create_bert_model_for_comparison(
            documents=documents,
            labels=labels,
            model_name=bert_model_name,
            output_dir=bert_output_dir,
            num_epochs=bert_epochs,
            batch_size=bert_batch_size,
        )

        console.print("[green]BERT model training complete![/green]")

        # Evaluate BERT
        console.print("\n[bold blue]Evaluating BERT model...[/bold blue]")
        bert_evaluator = BERTEvaluator(trainer=trainer)

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template="BERT local inference",
            model="bert-local",
        )

        bert_result = await bert_evaluator.evaluate(eval_input)
        self._print_result(bert_result)

        # Evaluate LLM models concurrently
        console.print(f"\n[bold blue]Comparing with {len(llm_models)} LLM models...[/bold blue]")

        async def _run_llm(m: str | dict[str, Any]) -> EvaluationResult:
            model_arg: str | dict[str, Any] = (
                {"model": m, "temperature": temperature} if isinstance(m, str) else m
            )
            return await self.evaluate(
                documents=documents,
                labels=labels,
                prompt_template=prompt_template,
                model=model_arg,
                max_concurrent=max_concurrent,
            )

        llm_results = list(await asyncio.gather(*[_run_llm(m) for m in llm_models]))

        # Print combined comparison
        all_results = llm_results + [bert_result]
        self._print_comparison(all_results)

        return llm_results, bert_result

    async def train_and_compare_bert_from_files(
        self,
        data_file: str | Path,
        llm_models: list[str],
        prompt_template: str,
        file_format: str = "json",
        bert_model_name: str = "bert-base-uncased",
        bert_output_dir: str | Path = "./bert_models",
        bert_epochs: int = 3,
        bert_batch_size: int = 8,
        temperature: float = 0.0,
        max_concurrent: int = 5,
    ) -> tuple[list[EvaluationResult], EvaluationResult]:
        """
        Train BERT and compare with LLMs from a data file.

        Args:
            data_file: Path to data file
            llm_models: List of LLM models to compare
            prompt_template: Prompt template for LLMs
            file_format: File format ('json' or 'csv')
            bert_model_name: Pretrained BERT model
            bert_output_dir: Directory to save BERT model
            bert_epochs: Training epochs
            bert_batch_size: Training batch size
            temperature: LLM temperature
            max_concurrent: Max concurrent LLM calls

        Returns:
            Tuple of (llm_results, bert_result)
        """
        # Load data
        if file_format == "json":
            documents, labels = self.loader.load_combined_from_json(data_file)
        elif file_format == "csv":
            documents, labels = self.loader.load_combined_from_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        console.print(f"[green]Loaded {len(documents)} documents from {data_file}[/green]")

        return await self.train_and_compare_bert(
            documents=documents,
            labels=labels,
            llm_models=llm_models,
            prompt_template=prompt_template,
            bert_model_name=bert_model_name,
            bert_output_dir=bert_output_dir,
            bert_epochs=bert_epochs,
            bert_batch_size=bert_batch_size,
            temperature=temperature,
            max_concurrent=max_concurrent,
        )

    async def test_prompt_decomposition(
        self,
        documents: list[Document],
        labels: list[Label],
        original_prompt: str,
        test_model: str,
        optimizer_model: str = "gemini-pro",
        num_sub_prompts: int = 5,
        temperature: float = 0.0,
    ) -> tuple[EvaluationResult, EvaluationResult, dict]:
        """
        Test prompt decomposition to see if it improves a cheaper model.

        Args:
            documents: Documents to evaluate
            labels: Expected labels
            original_prompt: Original prompt to test
            test_model: Model to test with (e.g., a cheaper model)
            optimizer_model: Model to use for decomposition (default: gemini-pro)
            num_sub_prompts: Max number of sub-prompts
            temperature: Temperature for generation

        Returns:
            Tuple of (original_result, decomposed_result, decomposition_info)
        """
        from valtron_core.optimized_evaluator import compare_original_vs_decomposed

        console.print(f"\n[bold magenta]Testing Prompt Decomposition[/bold magenta]")
        console.print(f"Original prompt will be decomposed by {optimizer_model}")
        console.print(f"Testing both versions with {test_model}")

        original_result, decomposed_result, decomposition = await compare_original_vs_decomposed(
            documents=documents,
            labels=labels,
            original_prompt=original_prompt,
            model=test_model,
            optimizer_model=optimizer_model,
            num_sub_prompts=num_sub_prompts,
            temperature=temperature,
        )

        # Print decomposition details
        console.print(f"\n[bold cyan]Decomposition Details:[/bold cyan]")
        console.print(f"Number of steps: {decomposition.get('num_steps', 0)}")
        console.print(f"Execution flow: {decomposition.get('execution_flow', 'N/A')}")
        console.print(f"\nBenefits: {decomposition.get('benefits', 'N/A')}")

        console.print(f"\n[bold cyan]Sub-prompts:[/bold cyan]")
        for sub_prompt in decomposition.get("sub_prompts", []):
            console.print(f"\n  Step {sub_prompt['step']}: {sub_prompt['description']}")
            console.print(f"  Prompt: {sub_prompt['prompt'][:100]}...")

        # Compare results
        console.print(f"\n[bold green]Comparison Results:[/bold green]")
        self._print_comparison([original_result, decomposed_result])

        # Analyze improvement
        if (
            original_result.metrics
            and decomposed_result.metrics
            and decomposed_result.metrics.accuracy > original_result.metrics.accuracy
        ):
            improvement = (
                decomposed_result.metrics.accuracy - original_result.metrics.accuracy
            ) * 100
            console.print(
                f"\n[bold green]✓ Decomposition IMPROVED accuracy by {improvement:.1f} percentage points![/bold green]"
            )
        elif (
            original_result.metrics
            and decomposed_result.metrics
            and decomposed_result.metrics.accuracy < original_result.metrics.accuracy
        ):
            console.print(
                f"\n[bold yellow]⚠ Decomposition decreased accuracy. Original prompt may be better for this model.[/bold yellow]"
            )
        else:
            console.print(
                f"\n[bold blue]Results are similar. Decomposition may not provide significant benefit.[/bold blue]"
            )

        return original_result, decomposed_result, decomposition

    async def test_prompt_decomposition_from_files(
        self,
        data_file: str | Path,
        original_prompt: str,
        test_model: str,
        file_format: str = "json",
        optimizer_model: str = "gemini-pro",
        num_sub_prompts: int = 5,
        temperature: float = 0.0,
    ) -> tuple[EvaluationResult, EvaluationResult, dict]:
        """
        Test prompt decomposition from a data file.

        Args:
            data_file: Path to data file
            original_prompt: Original prompt to test
            test_model: Model to test with
            file_format: File format ('json' or 'csv')
            optimizer_model: Model for decomposition
            num_sub_prompts: Max number of sub-prompts
            temperature: Temperature for generation

        Returns:
            Tuple of (original_result, decomposed_result, decomposition_info)
        """
        # Load data
        if file_format == "json":
            documents, labels = self.loader.load_combined_from_json(data_file)
        elif file_format == "csv":
            documents, labels = self.loader.load_combined_from_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        console.print(f"[green]Loaded {len(documents)} documents from {data_file}[/green]")

        return await self.test_prompt_decomposition(
            documents=documents,
            labels=labels,
            original_prompt=original_prompt,
            test_model=test_model,
            optimizer_model=optimizer_model,
            num_sub_prompts=num_sub_prompts,
            temperature=temperature,
        )

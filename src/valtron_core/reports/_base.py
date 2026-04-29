"""Shared base class and module-level setup for report generators."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from valtron_core.client import LLMClient
from valtron_core.models import EvaluationResult

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


class _ReportBase:
    """Shared utilities used by both HTML and PDF report generators."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def _compute_performance_best_values(
        self, results: list[EvaluationResult]
    ) -> dict[str, float]:
        """
        Compute the best values for performance metrics across all models.
        Best = highest accuracy, lowest cost, lowest time.
        """
        best_accuracy = -1.0
        best_total_cost = float('inf')
        best_avg_cost = float('inf')
        best_total_time = float('inf')
        best_avg_time = float('inf')

        for result in results:
            if result.metrics:
                if result.metrics.accuracy > best_accuracy:
                    best_accuracy = result.metrics.accuracy
                if result.metrics.total_cost < best_total_cost:
                    best_total_cost = result.metrics.total_cost
                if result.metrics.average_cost_per_document < best_avg_cost:
                    best_avg_cost = result.metrics.average_cost_per_document
                if result.metrics.total_time < best_total_time:
                    best_total_time = result.metrics.total_time
                if result.metrics.average_time_per_document < best_avg_time:
                    best_avg_time = result.metrics.average_time_per_document

        return {
            "best_accuracy": best_accuracy,
            "best_total_cost": best_total_cost,
            "best_avg_cost": best_avg_cost,
            "best_total_time": best_total_time,
            "best_avg_time": best_avg_time,
        }

    def _compute_performance_ranks(
        self, results: list[EvaluationResult]
    ) -> dict[str, dict]:
        """
        Compute dense performance ranks based on average_example_score (descending).
        Ties receive the same rank. Each entry includes the delta in percentage
        points from the top score, so near-ties are visually distinguishable.
        """
        model_scores = [
            (r.model, r.metrics.average_example_score)
            for r in results
            if r.metrics
        ]
        if not model_scores:
            return {}

        sorted_models = sorted(model_scores, key=lambda x: x[1], reverse=True)
        best_score = sorted_models[0][1]

        ranks: dict[str, dict] = {}
        current_rank = 1
        for i, (model, score) in enumerate(sorted_models):
            if i > 0 and score < sorted_models[i - 1][1]:
                current_rank += 1
            ranks[model] = {
                "rank": current_rank,
                "delta_pct": round((score - best_score) * 100, 2),
            }

        return ranks

    def _prepare_chart_data(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Prepare data for ECharts visualizations."""
        models = []
        accuracy = []
        avg_time = []
        total_time = []
        avg_cost = []
        total_cost = []

        all_doc_data = {}
        num_documents = 0

        for result in results:
            if result.metrics:
                models.append(result.model)
                accuracy.append(round(result.metrics.accuracy * 100, 2))
                avg_time.append(round(result.metrics.average_time_per_document, 3))
                total_time.append(round(result.metrics.total_time, 2))
                avg_cost.append(round(result.metrics.average_cost_per_document, 6))
                total_cost.append(round(result.metrics.total_cost, 6))

                doc_data = []
                for i, pred in enumerate(result.predictions):
                    doc_id = pred.document_id if pred.document_id else f"Doc {i + 1}"
                    if pred.example_score is not None:
                        score = round(pred.example_score * 100, 1)
                    else:
                        score = 100 if pred.is_correct else 0
                    doc_data.append({
                        'id': doc_id,
                        'cost': round(pred.cost, 6),
                        'time': round(pred.response_time, 3),
                        'score': score
                    })

                if len(result.predictions) > num_documents:
                    num_documents = len(result.predictions)

                all_doc_data[result.model] = doc_data

        histogram_data = self._prepare_histogram_data(all_doc_data, models)

        return {
            "models": models,
            "accuracy": accuracy,
            "avg_time": avg_time,
            "total_time": total_time,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "histogram_cost": histogram_data["cost"],
            "histogram_time": histogram_data["time"],
            "histogram_score": histogram_data["score"],
        }

    def _prepare_histogram_data(
        self, all_doc_data: dict[str, list[dict]], models: list[str], num_bins: int = 10
    ) -> dict[str, Any]:
        """Prepare histogram bin data for cost, time, and score distributions."""
        import math

        result = {"cost": {}, "time": {}, "score": {}}

        all_costs = []
        all_times = []
        all_scores = []

        for model, docs in all_doc_data.items():
            for doc in docs:
                all_costs.append(doc['cost'])
                all_times.append(doc['time'])
                all_scores.append(doc['score'])

        if not all_costs:
            return result

        def nice_number(x: float, round_up: bool = True) -> float:
            if x == 0:
                return 0
            sign = 1 if x >= 0 else -1
            x = abs(x)
            exponent = math.floor(math.log10(x))
            fraction = x / (10 ** exponent)

            if round_up:
                if fraction <= 1:
                    nice_fraction = 1
                elif fraction <= 2:
                    nice_fraction = 2
                elif fraction <= 5:
                    nice_fraction = 5
                else:
                    nice_fraction = 10
            else:
                if fraction < 1.5:
                    nice_fraction = 1
                elif fraction < 3:
                    nice_fraction = 2
                elif fraction < 7:
                    nice_fraction = 5
                else:
                    nice_fraction = 10

            return sign * nice_fraction * (10 ** exponent)

        def create_nice_bins(values: list[float], target_bins: int) -> list[float]:
            min_val = min(values)
            max_val = max(values)

            if min_val == max_val:
                return [min_val - 0.5, max_val + 0.5]

            data_range = max_val - min_val
            raw_bin_width = data_range / target_bins
            bin_width = nice_number(raw_bin_width, round_up=True)

            nice_min = math.floor(min_val / bin_width) * bin_width
            nice_max = math.ceil(max_val / bin_width) * bin_width

            bins = []
            current = nice_min
            while current <= nice_max + bin_width / 2:
                bins.append(round(current, 10))
                current += bin_width

            if len(bins) < 2:
                bins = [nice_min, nice_max]

            return bins

        cost_bins = create_nice_bins(all_costs, num_bins)
        time_bins = create_nice_bins(all_times, num_bins)
        score_bins = create_nice_bins(all_scores, num_bins)

        def get_decimal_places(bins: list[float]) -> int:
            if len(bins) < 2:
                return 2
            bin_width = bins[1] - bins[0]
            if bin_width == 0:
                return 2
            if bin_width >= 1:
                return 0
            elif bin_width >= 0.1:
                return 1
            elif bin_width >= 0.01:
                return 2
            elif bin_width >= 0.001:
                return 3
            elif bin_width >= 0.0001:
                return 4
            elif bin_width >= 0.00001:
                return 5
            else:
                return 6

        def create_bin_labels(bins: list[float], prefix: str = "", suffix: str = "") -> list[str]:
            decimals = get_decimal_places(bins)
            labels = []
            for i in range(len(bins) - 1):
                labels.append(f"{prefix}{bins[i]:.{decimals}f}{suffix} - {prefix}{bins[i+1]:.{decimals}f}{suffix}")
            return labels

        result["cost"]["bin_labels"] = create_bin_labels(cost_bins, prefix="$")
        result["time"]["bin_labels"] = create_bin_labels(time_bins, suffix="s")
        result["score"]["bin_labels"] = create_bin_labels(score_bins, suffix="%")

        result["cost"]["bins"] = cost_bins
        result["time"]["bins"] = time_bins
        result["score"]["bins"] = score_bins

        all_doc_ids = set()
        for model, docs in all_doc_data.items():
            for doc in docs:
                all_doc_ids.add(doc['id'])
        all_doc_ids = sorted(all_doc_ids)

        raw_cost_data = []
        raw_time_data = []
        raw_score_data = []

        for doc_id in all_doc_ids:
            cost_row = {"document": doc_id}
            time_row = {"document": doc_id}
            score_row = {"document": doc_id}

            for model in models:
                docs = all_doc_data.get(model, [])
                doc_entry = next((d for d in docs if d['id'] == doc_id), None)
                if doc_entry:
                    cost_row[model] = doc_entry['cost']
                    time_row[model] = doc_entry['time']
                    score_row[model] = doc_entry['score']
                else:
                    cost_row[model] = None
                    time_row[model] = None
                    score_row[model] = None

            raw_cost_data.append(cost_row)
            raw_time_data.append(time_row)
            raw_score_data.append(score_row)

        result["cost"]["raw_data"] = raw_cost_data
        result["time"]["raw_data"] = raw_time_data
        result["score"]["raw_data"] = raw_score_data

        for model in models:
            docs = all_doc_data.get(model, [])

            cost_counts = [0] * (len(cost_bins) - 1)
            time_counts = [0] * (len(time_bins) - 1)
            score_counts = [0] * (len(score_bins) - 1)

            for doc in docs:
                cost_bin_idx = self._find_bin_index(doc['cost'], cost_bins)
                if cost_bin_idx is not None:
                    cost_counts[cost_bin_idx] += 1

                time_bin_idx = self._find_bin_index(doc['time'], time_bins)
                if time_bin_idx is not None:
                    time_counts[time_bin_idx] += 1

                score_bin_idx = self._find_bin_index(doc['score'], score_bins)
                if score_bin_idx is not None:
                    score_counts[score_bin_idx] += 1

            if "models" not in result["cost"]:
                result["cost"]["models"] = {}
                result["time"]["models"] = {}
                result["score"]["models"] = {}

            result["cost"]["models"][model] = {"counts": cost_counts}
            result["time"]["models"][model] = {"counts": time_counts}
            result["score"]["models"][model] = {"counts": score_counts}

        return result

    def _find_bin_index(self, value: float, bins: list[float]) -> int | None:
        """Find the index of the bin that contains the given value."""
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                if bins[i] <= value <= bins[i + 1]:
                    return i
            else:
                if bins[i] <= value < bins[i + 1]:
                    return i
        return None

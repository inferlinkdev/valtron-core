"""ExperimentRegistry: registration, lookup, and sniff_best_match.

Also cross-checks sniff_best_match against utilities/config_wizard.py's own
still-independent (not yet wired to this registry; that's a later commit)
task_type dispatch logic, for a range of inputs, to confirm the two would
actually agree once wired together rather than assuming it.
"""

import json

from valtron_core.evaluation.classification import ClassificationExperiment
from valtron_core.evaluation.extraction import ExtractionExperiment
from valtron_core.evaluation.registry import ExperimentRegistry, label_looks_like_json
from valtron_core.evaluation.summarization import SummarizationExperiment


def _config_wizard_task_type(data_list: list[dict]) -> str | None:
    """Mirrors api_analyze_data's own dispatch (config_wizard.py), lines
    ~396-411, without needing a Flask request context to call the real route."""
    if not data_list:
        return None
    first_item = data_list[0]
    if "label" not in first_item:
        return "no_label"
    first_label = first_item.get("label", "")
    if isinstance(first_label, (dict, list)):
        first_label = json.dumps(first_label)
    try:
        label_value = json.loads(first_label)
        is_json = isinstance(label_value, (dict, list))
    except (json.JSONDecodeError, TypeError):
        is_json = False
    return "extraction" if is_json else "classification"


class TestRegistration:
    def test_the_three_existing_recipes_are_registered(self):
        assert ExperimentRegistry.get("classification") is ClassificationExperiment
        assert ExperimentRegistry.get("extraction") is ExtractionExperiment
        assert ExperimentRegistry.get("no_label") is SummarizationExperiment

    def test_unknown_task_type_returns_none(self):
        assert ExperimentRegistry.get("does_not_exist") is None

    def test_task_types_lists_at_least_the_three_existing_ones(self):
        assert {"classification", "extraction", "no_label"} <= set(ExperimentRegistry.task_types())

    def test_reregistering_the_same_class_is_a_no_op(self):
        ExperimentRegistry.register("classification", ClassificationExperiment)
        assert ExperimentRegistry.get("classification") is ClassificationExperiment

    def test_reregistering_a_different_class_under_a_taken_task_type_raises(self):
        try:
            ExperimentRegistry.register("classification", ExtractionExperiment)
            raised = False
        except ValueError:
            raised = True
        assert raised
        # Unchanged: the failed attempt didn't clobber the real registration.
        assert ExperimentRegistry.get("classification") is ClassificationExperiment


class TestLabelLooksLikeJson:
    def test_plain_string_is_not_json(self):
        assert label_looks_like_json("positive") is False

    def test_dict_is_json(self):
        assert label_looks_like_json({"a": 1}) is True

    def test_json_string_is_json(self):
        assert label_looks_like_json('{"a": 1}') is True

    def test_list_is_json(self):
        assert label_looks_like_json([1, 2]) is True


class TestSniffBestMatch:
    def test_empty_data_matches_nothing(self):
        assert ExperimentRegistry.sniff_best_match([]) is None

    def _cases(self):
        return [
            [{"id": "1", "content": "x", "label": "positive"}],
            [{"id": "1", "content": "x", "label": {"a": 1}}],
            [{"id": "1", "content": "x", "label": [1, 2]}],
            [{"id": "1", "content": "x", "label": '{"a": 1}'}],
            [{"id": "1", "content": "x"}],
        ]

    def test_matches_the_wizards_own_dispatch_for_every_case(self):
        for data in self._cases():
            assert ExperimentRegistry.sniff_best_match(data) == _config_wizard_task_type(data)

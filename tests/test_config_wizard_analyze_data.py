"""utilities/config_wizard.py's /api/analyze-data, before and after pointing
its task_type dispatch at ExperimentRegistry.sniff_best_match().

api_analyze_data's own dispatch used to compute "no_label"/"classification"/
"extraction" inline (a "label" membership check plus a JSON-parse check);
it now asks the registry instead. _analyze_no_label/_analyze_classification/
_analyze_extraction themselves are untouched. This locks in that the live
endpoint's actual JSON responses are unaffected by that swap: for each
sample below, the response from a real POST to /api/analyze-data must equal
calling the appropriate (unchanged) _analyze_* function directly with the
inputs the old inline dispatch logic would have computed, not merely that
the two dispatch decisions agree in isolation (see test_experiment_registry.py
for that narrower check).
"""

import json

import pytest

from valtron_core.utilities.config_wizard import (
    _analyze_classification,
    _analyze_extraction,
    _analyze_no_label,
    app,
)


def _old_dispatch_response(data_list: list[dict]) -> dict:
    """What api_analyze_data returned before this commit, computed the same way
    its old inline dispatch did, calling the same (still unchanged) _analyze_*
    functions it always called."""
    first_item = data_list[0]
    first_content = first_item.get("content", "")
    content_keys = list(first_content.keys()) if isinstance(first_content, dict) else ["content"]

    if "label" not in first_item:
        return _analyze_no_label(data_list, content_keys)

    first_label = first_item.get("label", "")
    if isinstance(first_label, (dict, list)):
        first_label = json.dumps(first_label)

    try:
        label_value = json.loads(first_label)
        is_json = isinstance(label_value, (dict, list))
    except (json.JSONDecodeError, TypeError):
        is_json = False

    if is_json:
        return _analyze_extraction(data_list, first_label, content_keys)
    return _analyze_classification(data_list, first_label, content_keys)


SAMPLES = [
    [{"id": "1", "content": "great product", "label": "positive"}],
    [{"id": "1", "content": "doc text", "label": {"name": "Alice", "age": 30}}],
    [{"id": "1", "content": "doc text", "label": [1, 2, 3]}],
    [{"id": "1", "content": "doc text", "label": '{"name": "Alice"}'}],
    [{"id": "1", "content": "doc text"}],
    [{"id": "1", "content": {"title": "t", "body": "b"}, "label": "yes"}],
]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAnalyzeDataDispatchUnaffectedByRegistrySwap:
    @pytest.mark.parametrize("data_list", SAMPLES)
    def test_live_endpoint_matches_old_dispatch_logic(self, client, data_list):
        response = client.post("/api/analyze-data", json={"data": data_list})

        # app.json.sort_keys is True (Flask's default), so both the request
        # body the test client builds and jsonify()'s own response sort dict
        # keys, independent of anything this commit changed. Round-trip the
        # comparison input through the app's own JSON provider so this test
        # isolates the dispatch change, not that pre-existing detail.
        normalized = app.json.loads(app.json.dumps(data_list))

        assert response.status_code == 200
        assert response.get_json() == _old_dispatch_response(normalized)

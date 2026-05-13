"""Configuration wizard web UI for creating recipe configs."""

import json
from pathlib import Path

import litellm

litellm.suppress_debug_info = True
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, template_folder="../templates")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

_LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
_all_models_cache: list[str] | None = None
_model_data_cache: dict | None = None

_POPULAR_MODELS = [
    "gpt-5.3-chat-latest",
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5",
    "gemini/gemini-3-flash",
    "gemini/gemini-3.1-flash-lite-preview",
    "gemini/gemini-2.5-pro",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "ollama/llama3.3",
    "ollama/mistral",
    "groq/llama-3.3-70b-versatile",
    "deepseek/deepseek-chat",
]


def _load_model_data() -> dict:
    """Fetch and cache the litellm prices JSON, falling back to an empty dict."""
    global _model_data_cache, _all_models_cache
    if _model_data_cache is not None:
        return _model_data_cache
    try:
        resp = requests.get(_LITELLM_PRICES_URL, timeout=10)
        resp.raise_for_status()
        data: dict = resp.json()
        data.pop("sample_spec", None)
    except Exception:
        data = {}
    _model_data_cache = data
    _all_models_cache = list(data.keys()) if data else list(litellm.model_list)
    return _model_data_cache


def _build_model_entry(name: str) -> dict:
    data = _load_model_data()
    info = data.get(name, {})
    vision = bool(info.get("supports_vision", False))
    pdf = bool(info.get("supports_pdf_input", False))
    return {"name": name, "supports_vision": vision, "supports_pdf": pdf}


def get_all_models() -> list[str]:
    """Return all model names (cached), fetched from the litellm prices JSON."""
    global _all_models_cache
    if _all_models_cache is not None:
        return _all_models_cache
    _load_model_data()
    return _all_models_cache or []


def _search_rank(name: str, query: str) -> int:
    """Lower rank = higher priority in search results."""
    lower = name.lower()
    suffix = lower.split("/")[-1] if "/" in lower else lower
    if lower == query:
        return 0
    if lower.startswith(query):
        return 1
    if suffix == query:
        return 2
    if suffix.startswith(query):
        return 3
    return 4


def search_models(query: str, exclude: str = "", limit: int = 20) -> list[dict]:
    """Return up to `limit` models matching `query`."""
    query = query.strip().lower()
    excluded = {e.strip() for e in exclude.split(",") if e.strip()}
    if not query:
        results = []
        for name in _POPULAR_MODELS:
            if name not in excluded:
                results.append(_build_model_entry(name))
        return results[:limit]

    all_names = get_all_models()
    matches = [n for n in all_names if query in n.lower() and n not in excluded]
    matches.sort(key=lambda n: (_search_rank(n, query), n))
    return [_build_model_entry(n) for n in matches[:limit]]


def suggest_models(current_model: str) -> list[dict]:
    """Suggest 3 models to test against, excluding the current model."""
    all_suggestions = [
        {"name": "gpt-5-mini", "is_simple": True},
        {"name": "gemini/gemini-3-flash", "is_simple": True},
        {"name": "gpt-5.2", "is_simple": False},
        {"name": "gemini/gemini-2.5-pro", "is_simple": False},
    ]

    result = []
    for s in all_suggestions:
        if s["name"] == current_model:
            continue
        entry = _build_model_entry(s["name"])
        entry["type"] = "llm"
        entry["is_simple"] = s["is_simple"]
        result.append(entry)

    return result[:3]


@app.route("/favicon.svg")
def favicon() -> object:
    """Serve the Valtron favicon."""
    template_dir = Path(__file__).parent.parent / "templates"
    return send_from_directory(template_dir, "favicon.svg", mimetype="image/svg+xml")


@app.route("/")
def index():
    """Render the configuration wizard."""
    return render_template("config_wizard.html")


@app.route("/api/search-models", methods=["GET"])
def api_search_models():
    """Search litellm models by query string. Returns up to 20 results."""
    query = request.args.get("q", "")
    exclude = request.args.get("exclude", "")
    results = search_models(query, exclude=exclude)
    return jsonify({"models": results})


@app.route("/api/suggest-models", methods=["POST"])
def api_suggest_models():
    """API endpoint to get model suggestions."""
    data = request.json
    current_model = data.get("current_model", "")

    suggestions = suggest_models(current_model)

    return jsonify({"suggestions": suggestions})


@app.route("/api/download-data", methods=["POST"])
def api_download_data():
    """Download training data from URL and return it in memory."""
    data = request.json
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        json_data = response.json()
        return jsonify(
            {
                "success": True,
                "data": json_data,
                "num_examples": len(json_data) if isinstance(json_data, list) else 1,
            }
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download file: {str(e)}"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Downloaded file is not valid JSON"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/api/upload-data", methods=["POST"])
def api_upload_data() -> tuple:
    """Accept a multipart file upload, validate it is JSON, and return it in memory."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        json_data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return jsonify({"error": "File is not valid JSON"}), 400

    return jsonify(
        {
            "success": True,
            "data": json_data,
            "num_examples": len(json_data) if isinstance(json_data, list) else 1,
        }
    )


@app.route("/api/analyze-data", methods=["POST"])
def api_analyze_data():
    """Analyze training data to detect JSON labels and infer field metrics config."""
    payload = request.json
    inline_data = payload.get("data")
    data_path = payload.get("data_path", "")

    try:
        if inline_data is not None:
            data_list = inline_data
        elif data_path:
            path = Path(data_path)
            if not path.is_absolute():
                path = Path.cwd() / data_path
            if not path.exists():
                return jsonify({"error": f"File not found: {data_path}"}), 404
            with open(path) as f:
                data_list = json.load(f)
        else:
            return jsonify({"error": "No data or data path provided"}), 400

        if not isinstance(data_list, list) or not data_list:
            return jsonify({"is_json": False, "reason": "Empty or non-list data"})

        first_item = data_list[0]
        first_content = first_item.get("content", "")
        content_keys: list[str] = list(first_content.keys()) if isinstance(first_content, dict) else ["content"]

        first_label = first_item.get("label", "")
        if isinstance(first_label, (dict, list)):
            first_label = json.dumps(first_label)

        try:
            label_value = json.loads(first_label)
            is_json = isinstance(label_value, (dict, list))
        except (json.JSONDecodeError, TypeError):
            is_json = False

        if not is_json:
            enum_values = sorted(
                {str(d.get("label", "")) for d in data_list if str(d.get("label", "")) != ""}
            )
            response_format_preview = ""
            if enum_values:
                if len(enum_values) <= 50:
                    args = ", ".join(repr(v) for v in enum_values)
                    response_format_preview = (
                        f"class ResponseModel(BaseModel):\n    label: Literal[{args}]"
                    )
                else:
                    response_format_preview = "class ResponseModel(BaseModel):\n    label: str"
                    enum_values = []
            label_property: dict = {"type": "string", "description": "Predicted class label"}
            if enum_values:
                label_property["enum"] = enum_values
            string_label_schema: dict = {
                "type": "object",
                "title": "ResponseModel",
                "properties": {"label": label_property},
                "required": ["label"],
                "additionalProperties": False,
            }
            response_format_schema = {
                "type": "json_schema",
                "json_schema": {"name": "ResponseModel", "strict": True, "schema": string_label_schema},
            }
            return jsonify(
                {
                    "is_json": False,
                    "sample_label": first_label,
                    "num_examples": len(data_list),
                    "enum_values": enum_values,
                    "response_format_preview": response_format_preview,
                    "response_format_schema": response_format_schema,
                    "content_keys": content_keys,
                }
            )

        from valtron_core.utilities.field_config_generator import infer_field_config

        field_config = infer_field_config(first_label)

        def _collect_classes(
            data: object, class_name: str, out: list[str]
        ) -> str:
            if isinstance(data, bool):
                return "bool"
            if isinstance(data, int):
                return "int"
            if isinstance(data, float):
                return "float"
            if isinstance(data, dict):
                lines = [f"class {class_name}(BaseModel):"]
                for k, v in data.items():
                    field_type = _collect_classes(v, k.rstrip("s").capitalize(), out)
                    lines.append(f"    {k}: {field_type}")
                out.append("\n".join(lines))
                return class_name
            if isinstance(data, list) and data:
                item_name = class_name.rstrip("s").capitalize()
                item_type = _collect_classes(data[0], item_name, out)
                return f"list[{item_type}]"
            if isinstance(data, list):
                return "list[str]"
            return "str"

        def _json_schema_from_value(value: object, title: str) -> dict:
            if isinstance(value, bool):
                return {"type": "boolean"}
            if isinstance(value, int):
                return {"type": "integer"}
            if isinstance(value, float):
                return {"type": "number"}
            if isinstance(value, dict):
                props = {k: _json_schema_from_value(v, k) for k, v in value.items()}
                return {
                    "type": "object",
                    "title": title,
                    "properties": props,
                    "required": list(value.keys()),
                    "additionalProperties": False,
                }
            if isinstance(value, list) and value:
                return {"type": "array", "items": _json_schema_from_value(value[0], title)}
            if isinstance(value, list):
                return {"type": "array", "items": {"type": "string"}}
            return {"type": "string"}

        label_json = json.loads(first_label)
        if isinstance(label_json, dict):
            extra_classes: list[str] = []
            main_lines = ["class ResponseModel(BaseModel):"]
            for k, v in label_json.items():
                field_type = _collect_classes(v, k.rstrip("s").capitalize(), extra_classes)
                main_lines.append(f"    {k}: {field_type}")
            all_parts = extra_classes + ["\n".join(main_lines)]
            response_format_preview = "\n\n".join(all_parts)
            json_label_schema = _json_schema_from_value(label_json, "ResponseModel")
            json_label_schema["title"] = "ResponseModel"
            response_format_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ResponseModel",
                    "strict": True,
                    "schema": json_label_schema,
                },
            }
        else:
            response_format_preview = ""
            response_format_schema = None

        return jsonify(
            {
                "is_json": True,
                "sample_label": first_label,
                "num_examples": len(data_list),
                "field_config": field_config.model_dump(),
                "enum_values": [],
                "response_format_preview": response_format_preview,
                "response_format_schema": response_format_schema,
                "content_keys": content_keys,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cwd", methods=["GET"])
def api_cwd() -> tuple:
    """Return the server's current working directory."""
    return jsonify({"cwd": str(Path.cwd())})


@app.route("/api/save-config", methods=["POST"])
def api_save_config():
    """Save the generated configuration to a file."""
    data = request.json
    config = data.get("config", {})
    filename = data.get("filename", "custom_config.json")

    # Save to ./configs/ relative to the user's working directory
    config_dir = Path.cwd() / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / filename
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return jsonify({"success": True, "path": str(config_path)})


def start_wizard(host="0.0.0.0", port=5000):
    """Start the configuration wizard server."""
    print(f"\n{'=' * 80}")
    print("CONFIGURATION WIZARD")
    print(f"{'=' * 80}\n")
    print(f"Open your browser to: http://localhost:{port}")
    print(f"(Or http://127.0.0.1:{port} if running in Docker)")
    print("\nThis wizard will help you create a configuration file for the")
    print("label classification recipe.\n")
    print(f"{'=' * 80}\n")

    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    start_wizard()

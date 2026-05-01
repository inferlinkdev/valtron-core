"""Configuration wizard web UI for creating recipe configs."""

import json
import secrets
from pathlib import Path

import litellm
from litellm.utils import supports_pdf_input
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__, template_folder="../templates")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

_all_models_cache: list[dict] | None = None

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


def _build_model_entry(name: str) -> dict:
    return {
        "name": name,
        "supports_vision": litellm.supports_vision(name),
        "supports_pdf": supports_pdf_input(name),
    }


def get_all_models() -> list[dict]:
    """Return all litellm models with vision/pdf support flags (cached)."""
    global _all_models_cache
    if _all_models_cache is not None:
        return _all_models_cache

    models = [_build_model_entry(name) for name in litellm.model_list]
    _all_models_cache = models
    return models


def search_models(query: str, exclude: str = "", limit: int = 20) -> list[dict]:
    """Return up to `limit` models matching `query`, excluding `exclude`."""
    query = query.strip().lower()
    if not query:
        results = []
        for name in _POPULAR_MODELS:
            if name != exclude:
                results.append(_build_model_entry(name))
        return results[:limit]

    all_models = get_all_models()
    matches = [m for m in all_models if query in m["name"].lower() and m["name"] != exclude]
    return matches[:limit]


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
    """Download training data from URL and save to examples directory."""
    data = request.json
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Download the file
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Validate it's JSON
        json_data = response.json()

        # Generate random filename
        random_suffix = secrets.token_hex(4)
        filename = f"training_data_{random_suffix}.json"

        # Save to examples directory
        examples_dir = Path(__file__).parent.parent.parent / "examples" / "example_data"
        examples_dir.mkdir(parents=True, exist_ok=True)

        file_path = examples_dir / filename
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Return relative path for config
        relative_path = f"examples/example_data/{filename}"

        return jsonify(
            {
                "success": True,
                "path": relative_path,
                "filename": filename,
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
    """Accept a multipart file upload, validate it is JSON, and save to examples directory."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        json_data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return jsonify({"error": "File is not valid JSON"}), 400

    random_suffix = secrets.token_hex(4)
    filename = f"training_data_{random_suffix}.json"

    examples_dir = Path(__file__).parent.parent.parent / "examples" / "example_data"
    examples_dir.mkdir(parents=True, exist_ok=True)

    file_path = examples_dir / filename
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=2)

    relative_path = f"examples/example_data/{filename}"
    return jsonify(
        {
            "success": True,
            "path": relative_path,
            "filename": filename,
            "num_examples": len(json_data) if isinstance(json_data, list) else 1,
        }
    )


@app.route("/api/analyze-data", methods=["POST"])
def api_analyze_data():
    """Analyze training data to detect JSON labels and infer field metrics config."""
    data = request.json
    data_path = data.get("data_path", "")

    if not data_path:
        return jsonify({"error": "No data path provided"}), 400

    path = Path(data_path)
    if not path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        path = project_root / data_path

    if not path.exists():
        return jsonify({"error": f"File not found: {data_path}"}), 404

    try:
        with open(path) as f:
            data_list = json.load(f)

        if not isinstance(data_list, list) or not data_list:
            return jsonify({"is_json": False, "reason": "Empty or non-list data"})

        first_label = data_list[0].get("label", "")
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
            return jsonify(
                {
                    "is_json": False,
                    "sample_label": first_label,
                    "num_examples": len(data_list),
                    "enum_values": enum_values,
                    "response_format_preview": response_format_preview,
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

        label_json = json.loads(first_label)
        if isinstance(label_json, dict):
            extra_classes: list[str] = []
            main_lines = ["class ResponseModel(BaseModel):"]
            for k, v in label_json.items():
                field_type = _collect_classes(v, k.rstrip("s").capitalize(), extra_classes)
                main_lines.append(f"    {k}: {field_type}")
            all_parts = extra_classes + ["\n".join(main_lines)]
            response_format_preview = "\n\n".join(all_parts)
        else:
            response_format_preview = ""

        return jsonify(
            {
                "is_json": True,
                "sample_label": first_label,
                "num_examples": len(data_list),
                "field_config": field_config.model_dump(),
                "enum_values": [],
                "response_format_preview": response_format_preview,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

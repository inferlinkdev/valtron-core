---
sidebar_position: 3
---

# Configuration Wizard

The configuration wizard is a browser-based UI that walks you through building a Valtron config file step by step. It is the fastest way to get a working config without writing JSON by hand. For fine-grained control over every available option, see the [Config Format](../config-format) reference.

The wizard covers:
- Selecting your current model and the models you want to test against
- Writing your prompt template
- Linking your labeled dataset
- Reviewing the auto-generated response format
- Optionally configuring few-shot learning and per-field metrics

At the end it generates a ready-to-use `custom_config.json` that you can download or copy to clipboard.

---

## Starting the wizard

```python
from valtron_core.utilities.config_wizard import start_wizard

start_wizard()
```

This starts a local Flask server. Open [http://localhost:5000](http://localhost:5000) in your browser.

You can also run it directly from the command line:

```bash
python -m valtron_core.utilities.config_wizard
```

The wizard runs entirely on your machine. No data leaves your environment.

---

## Step 1: Current Model

![Step 1: Current Model](/img/1_wizard_current_model.png)

Enter the model you are currently using for this task. The field uses autocomplete backed by the full LiteLLM model list, so you can type a partial name (e.g. `gpt`, `claude`, `gemini`) and select from the dropdown.

**Example inputs:** `gpt-4o-mini`, `claude-haiku-4-5-20251001`, `ollama/llama3.3`

---

## Step 2: Test Models

![Step 2: Select Models](/img/2_wizard_select_models.png)

The wizard automatically suggests three comparison models based on your current model: a mix of simpler (cheaper) and more capable options.

You can:
- Remove any suggested model with the **×** button
- Search and add any LiteLLM-compatible model using the search field
- Enable **"Try to improve prompts for simpler models"** to apply `explanation` and `few_shot` manipulations to smaller models automatically. This often meaningfully closes the accuracy gap between cheap and expensive models.

Use the **"Add local model (Ollama)"** or **"Add trained transformer"** help links at the bottom for guidance on adding local models.

---

## Step 3: Prompt

![Step 3: Prompt](/img/3_wizard_prompt.png)

Enter the prompt template you use for this task. The prompt **must contain `{content}`** as a placeholder. Each document's content is inserted there during evaluation.

**Example:**
```
List all institutions in the following affiliation string.

{content}
```

If you enabled prompt improvement in Step 2, a notice will appear confirming that few-shot examples will be configured in the next step.

---

## Step 4: Training Data

![Step 4: Training Data](/img/4_wizard_training_data.png)

Provide your labeled dataset. You can enter:
- A **URL** pointing to a JSON file (the wizard downloads it automatically)
- A **local file path** relative to your working directory (e.g. `./examples/my_data.json`)
- A **file upload** by clicking the browse button — files up to 500 MB are supported; for larger files, enter the local file path instead

Your data must be a JSON array in this format:

```json
[
  {
    "id": "1",
    "content": "John Smith, Department of Computer Science, Stanford University, Stanford, CA, USA; Google Research, Mountain View, CA, USA",
    "label": {
      "institutions": [
        { "name": "Stanford University", "city": "Stanford",      "state": "CA", "country": "USA" },
        { "name": "Google Research",     "city": "Mountain View", "state": "CA", "country": "USA" }
      ]
    }
  }
]
```

Also enter a **use case description** (e.g. `"academic affiliation extraction"`, `"geographic entity extraction"`). This is used as the report header and passed to the AI recommendation engine.

### Few-shot configuration (optional)

If you enabled prompt improvement in Step 2, a **Few-Shot Configuration** section appears here:

| Field | Default | Description |
|---|---|---|
| Number of examples to generate | 20 | Synthetic examples the generator produces |
| Max seed examples | 10 | Real data rows used to seed generation |

Click **Next**. The wizard downloads (if needed) and analyzes your data before moving to Step 5.

---

## Step 5: Response Format

![Step 5: Response Format](/img/5_wizard_response_format.png)

The wizard displays the Pydantic model inferred from your label values and shows a note that the schema will be saved to your config file. This is a best-effort inference from your data — if you need a more precise schema, pass your own Pydantic model directly as `response_format` when constructing `ModelEval`. See [Structured extraction mode](https://valtron.ai/docs/evaluation-results#structured-extraction-mode) for details.

The inferred schema is saved to the config file as `response_format_schema` in litellm format, which is what the system uses to constrain LLM output. For example, a plain-text label dataset with three classes produces:

```json
{
  "response_format_schema": {
    "type": "json_schema",
    "json_schema": {
      "name": "ResponseModel",
      "strict": true,
      "schema": {
        "type": "object",
        "title": "ResponseModel",
        "properties": {
          "label": {
            "type": "string",
            "description": "Predicted class label",
            "enum": ["negative", "neutral", "positive"]
          }
        },
        "required": ["label"],
        "additionalProperties": false
      }
    }
  }
}
```

For datasets with more than 50 distinct label values the `enum` constraint is omitted and the field is typed as a plain string. For JSON-structured labels, the schema is recursively inferred from the structure of your first label example. The list of detected enum values (for plain-text labels) is shown below the Pydantic class preview.

---

## Step 6: Field-Level Metrics

![Step 6: Field-Level Metrics](/img/6_wizard_field_metrics.png)

The wizard inspects your data labels to decide what grading options are available.

### Plain-text labels

You can configure how the `label` field is graded (default: exact match). Select **"Yes — configure field grading"** to change the metric.

### JSON-structured labels

If labels are JSON objects (e.g. `{"institutions": [{"name": "Stanford University", "city": "Stanford", "state": "CA", "country": "USA"}]}`), you can choose between:

- **Overall accuracy only** — exact-match on the full JSON string
- **Per-field grading** — configure how each field is compared individually

When you select **"Yes — configure field grading"**, an interactive tree editor appears with one row per field detected in your label schema. For each field you can set:

| Control | Description |
|---|---|
| **Metric** | How the field is compared: `Exact match`, `Fuzzy match`, `Semantic similarity`, or `LLM graded` |
| **Weight** | How much this field contributes to the overall score (default 1) |
| **Optional** | Whether a missing field counts as correct |

For **list fields**, additional controls appear:

| Control | Description |
|---|---|
| **Ordered** | Whether list item order must match |
| **Match threshold** | Minimum similarity to count a pair as matched (0–1) |

See [Field Metrics](../field-metrics) for a full reference.

---

## Step 7: Review and Download

![Step 7: Review and Download](/img/7_wizard_review.png)

The wizard displays the generated configuration as editable JSON. You can tweak any value directly before saving.

Use:
- **Download Config** — saves `custom_config.json` to your browser's downloads folder
- **Copy to Clipboard** — copies the JSON for pasting into your project

The file is also saved automatically to `./configs/custom_config.json` relative to your working directory.

---

## Using the generated config

Pass the downloaded file directly to `ModelEval`:

```python
from valtron_core.recipes import ModelEval

experiment = ModelEval(
    config="./configs/custom_config.json",
    data="./my_data.json",
)
report_path = experiment.run()
print(f"Report: {report_path}")
```

See [Config Format](../config-format) for a full reference of every config field, and [Quick Start](./quick-start) for a minimal end-to-end example.

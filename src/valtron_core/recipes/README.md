# Valtron Core Recipes

Pre-built pipelines for common ML evaluation tasks.

## ModelEval Recipe

A complete pipeline for evaluating multiple models on a task — supports both label classification and structured extraction.

### Features

- **Dual Mode**: Label/classification mode (no `response_format`) or structured extraction mode (with `response_format`)
- **Automatic Prompt Optimization**: Applies `ExplanationEnhancer` to simple models for better accuracy
- **Optional Few-Shot Generation**: Generate additional training data from seed examples
- **Multi-Model Evaluation**: Test multiple models simultaneously
- **Comprehensive Reporting**: HTML reports with visualizations and AI recommendations

### Two-Line Usage

```python
from valtron_core.recipes import ModelEval

# Label classification mode
evaluator = ModelEval(config=config, data=data)
report_path = evaluator.run()

# Structured extraction mode
from mymodels import MySchema
evaluator = ModelEval(config=config, data=data, response_format=MySchema)
report_path = evaluator.run()

# Async contexts (e.g. Jupyter notebooks)
report_path = await evaluator.arun()
```

### Configuration Format

Create a JSON configuration file:

```json
{
  "models": [
    {
      "name": "gpt-4o-mini",
      "prompt_manipulation": ["explanation"],
      "params": {
        "temperature": 0.0
      }
    },
    {
      "name": "gpt-4o",
      "prompt_manipulation": [],
      "params": {
        "temperature": 0.0
      }
    }
  ],
  "prompt": "Your prompt with {document} placeholder",
  "few_shot": {
    "enabled": true,
    "max_seed_examples": 10,
    "generator_model": "gpt-4o-mini",
    "num_examples": 20
  },
  "output_dir": "./results",
  "use_case": "Description of your use case",
  "temperature": 0.0
}
```

### Data Format

Provide labeled documents in JSON format:

```json
[
  {
    "content": "Your document text...",
    "label": "expected_label",
    "metadata": {
      "optional": "metadata"
    }
  }
]
```

### What Happens

1. **Few-Shot Generation** (optional): Generates additional training examples
2. **Prompt Optimization**: Enhances prompts for simple models with ExplanationEnhancer
3. **Evaluation**: Tests all models on the dataset
4. **Reporting**: Generates HTML report with:
   - Accuracy metrics
   - Cost analysis
   - Performance comparison
   - Model recommendation
   - Interactive visualizations

### Prompt Manipulations

Some manipulations require `response_format` to be provided (structured extraction mode only):

```python
from valtron_core.recipes import Manipulation, STRUCTURED_MANIPULATIONS

# Check which manipulations require response_format
print(STRUCTURED_MANIPULATIONS)
# frozenset({Manipulation.decompose, Manipulation.hallucination_filter, Manipulation.multi_pass})

# Check a single manipulation
Manipulation.decompose.requires_response_format   # True
Manipulation.few_shot.requires_response_format    # False
```

### Output

The recipe generates:
- `evaluation_data.json` - Processed dataset
- `report.html` - Main report
- `accuracy_comparison.png` - Accuracy chart
- `cost_comparison.png` - Cost chart
- `dashboard.html` - Interactive dashboard
- And more visualizations

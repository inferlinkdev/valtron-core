# API Reference

```{toctree}
:hidden:

evaluation
models
scoring
analysis
training
reports
```

Generated directly from docstrings in `../src/valtron_core/` via Sphinx `autodoc` + `napoleon`. For prose, walkthroughs, and code snippets, see the [User Guide](../user-guide/index). Browse by module in the sidebar, or search across every class below.

<input type="text" id="api-search" placeholder="Search classes... (e.g. &quot;transformer&quot;, &quot;config&quot;, &quot;field&quot;)" onkeyup="valtronFilterApiTable()">

<table id="api-table" class="api-table">
<thead><tr><th>Class</th><th>Description</th></tr></thead>
<tbody>
<tr><td><a href="model_eval.html"><code>ModelEval</code></a></td><td>The primary entry point: config + data in, evaluation report out.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="classification_experiment.html"><code>ClassificationExperiment</code></a></td><td>Classification-shaped data with plain string labels.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="extraction_experiment.html"><code>ExtractionExperiment</code></a></td><td>Structured extraction with a required schema.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="model_eval_config.html"><code>ModelEvalConfig</code></a></td><td>Top-level config schema: models, prompt, output settings.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="classification_config.html"><code>ClassificationConfig</code></a></td><td>Config for <code>ClassificationExperiment</code>: adds <code>infer_schema</code>.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="llm_model_config.html"><code>LLMModelConfig</code></a></td><td>A single hosted or self-hosted LLM model entry.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="transformer_model_config.html"><code>TransformerModelConfig</code></a></td><td>A single local transformer model entry.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="few_shot_config.html"><code>FewShotConfig</code></a></td><td>Few-shot example generation settings.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="decompose_config.html"><code>DecomposeConfig</code></a></td><td>Per-field decomposition settings for the <code>decompose</code> manipulation.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="manipulation.html"><code>Manipulation</code></a></td><td>Enum of built-in prompt manipulations.<a class="api-module-link" href="evaluation.html">valtron_core.evaluation</a></td></tr>
<tr><td><a href="document.html"><code>Document</code></a></td><td>One input record: id, content, attachments.<a class="api-module-link" href="models.html">valtron_core.models</a></td></tr>
<tr><td><a href="label.html"><code>Label</code></a></td><td>One expected/ground-truth value.<a class="api-module-link" href="models.html">valtron_core.models</a></td></tr>
<tr><td><a href="evaluation_result.html"><code>EvaluationResult</code></a></td><td>Everything <code>ModelEval</code> produces for one model.<a class="api-module-link" href="models.html">valtron_core.models</a></td></tr>
<tr><td><a href="evaluation_metrics.html"><code>EvaluationMetrics</code></a></td><td>Accuracy, cost, timing, and field metrics for one model.<a class="api-module-link" href="models.html">valtron_core.models</a></td></tr>
<tr><td><a href="prediction_result.html"><code>PredictionResult</code></a></td><td>One document's prediction within a result.<a class="api-module-link" href="models.html">valtron_core.models</a></td></tr>
<tr><td><a href="field_metrics_config.html"><code>FieldMetricsConfig</code></a></td><td>Per-field scoring config for structured extraction.<a class="api-module-link" href="models.html">valtron_core.models</a></td></tr>
<tr><td><a href="field_config.html"><code>FieldConfig</code></a></td><td>One node (leaf/object/list) in a field-metrics tree.<a class="api-module-link" href="scoring.html">valtron_core.scoring.json_eval</a></td></tr>
<tr><td><a href="eval_result.html"><code>EvalResult</code></a></td><td>The scored output of one field.<a class="api-module-link" href="scoring.html">valtron_core.scoring.json_eval</a></td></tr>
<tr><td><a href="tradeoff_analyzer.html"><code>TradeoffAnalyzer</code></a></td><td>Cost/accuracy routing between a transformer and one or more LLMs.<a class="api-module-link" href="analysis.html">valtron_core.analysis</a></td></tr>
<tr><td><a href="transformer_classifier.html"><code>TransformerClassifier</code></a></td><td>Train and run a local DistilBERT classifier.<a class="api-module-link" href="training.html">valtron_core.training</a></td></tr>
<tr><td><a href="bert_trainer.html"><code>BERTTrainer</code></a></td><td>The underlying training loop <code>TransformerClassifier</code> wraps.<a class="api-module-link" href="training.html">valtron_core.training</a></td></tr>
<tr><td><a href="bert_evaluator.html"><code>BERTEvaluator</code></a></td><td>Standalone evaluation of a trained transformer.<a class="api-module-link" href="training.html">valtron_core.training</a></td></tr>
<tr><td><a href="report_generator.html"><code>ReportGenerator</code></a></td><td>HTML and PDF report generation.<a class="api-module-link" href="reports.html">valtron_core.reports</a></td></tr>
</tbody>
</table>

<p id="api-search-empty" style="display:none">No classes match your search.</p>

<script>
function valtronFilterApiTable() {
  var query = document.getElementById('api-search').value.toLowerCase();
  var rows = document.querySelectorAll('#api-table tbody tr');
  var visible = 0;
  rows.forEach(function (row) {
    var match = row.textContent.toLowerCase().indexOf(query) !== -1;
    row.style.display = match ? '' : 'none';
    if (match) visible++;
  });
  document.getElementById('api-search-empty').style.display = visible === 0 ? '' : 'none';
}
</script>

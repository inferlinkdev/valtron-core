# `valtron_core.models`

```{toctree}
:hidden:

document
label
evaluation_result
evaluation_metrics
prediction_result
field_metrics_config
```

Input and output data models shared across the whole package.

<table class="api-table">
<thead><tr><th>Class</th><th>Description</th></tr></thead>
<tbody>
<tr><td><a href="document.html"><code>Document</code></a></td><td>One input record (id, content, attachments, ...)</td></tr>
<tr><td><a href="label.html"><code>Label</code></a></td><td>One expected/ground-truth value</td></tr>
<tr><td><a href="evaluation_result.html"><code>EvaluationResult</code></a></td><td>Everything <code>ModelEval</code> produces for one model</td></tr>
<tr><td><a href="evaluation_metrics.html"><code>EvaluationMetrics</code></a></td><td>Accuracy, cost, timing, field metrics for one model</td></tr>
<tr><td><a href="prediction_result.html"><code>PredictionResult</code></a></td><td>One document's prediction within a result</td></tr>
<tr><td><a href="field_metrics_config.html"><code>FieldMetricsConfig</code></a></td><td>Per-field scoring config, see <a href="../user-guide/extraction/field-metrics/index.html">Field Metrics</a></td></tr>
</tbody>
</table>

# `valtron_core.scoring.json_eval`

```{toctree}
:hidden:

field_config
leaf_metric_config
object_metric_config
list_metric_config
eval_result
```

The field-level scoring tree underneath [Field Metrics](../user-guide/extraction/field-metrics/index).

<table class="api-table">
<thead><tr><th>Class</th><th>Description</th></tr></thead>
<tbody>
<tr><td><a href="field_config.html"><code>FieldConfig</code></a></td><td>One node (leaf/object/list) in a <code>field_metrics_config</code> tree</td></tr>
<tr><td><a href="leaf_metric_config.html"><code>LeafMetricConfig</code></a></td><td>Scoring config for a leaf node, and the built-in metrics available to it</td></tr>
<tr><td><a href="object_metric_config.html"><code>ObjectMetricConfig</code></a></td><td>Scoring config for an object node, and the built-in propagation strategies available to it</td></tr>
<tr><td><a href="list_metric_config.html"><code>ListMetricConfig</code></a></td><td>Scoring config for a list node: matching, thresholds, and expensive-comparison guardrails</td></tr>
<tr><td><a href="eval_result.html"><code>EvalResult</code></a></td><td>The scored output of one field</td></tr>
</tbody>
</table>

# `valtron_core.evaluation`

```{toctree}
:hidden:

model_eval
classification_experiment
extraction_experiment
summarization_experiment
model_eval_config
classification_config
summarization_config
summarization_ranking
summarization_score
llm_model_config
transformer_model_config
few_shot_config
decompose_config
manipulation
```

Recipe classes and their config schemas, used throughout the [User Guide](../user-guide/classification/config-format).

<table class="api-table">
<thead><tr><th>Class</th><th>Description</th></tr></thead>
<tbody>
<tr><td><a href="model_eval.html"><code>ModelEval</code></a></td><td>The base recipe: config + data in, evaluation report out</td></tr>
<tr><td><a href="classification_experiment.html"><code>ClassificationExperiment</code></a></td><td>Classification-shaped data with plain string labels</td></tr>
<tr><td><a href="extraction_experiment.html"><code>ExtractionExperiment</code></a></td><td>Structured extraction with a required schema</td></tr>
<tr><td><a href="summarization_experiment.html"><code>SummarizationExperiment</code></a></td><td>Reference-free summarization quality, judged rather than labeled</td></tr>
<tr><td><a href="model_eval_config.html"><code>ModelEvalConfig</code></a></td><td>Top-level config schema</td></tr>
<tr><td><a href="classification_config.html"><code>ClassificationConfig</code></a></td><td>Config for <code>ClassificationExperiment</code>: adds <code>infer_schema</code></td></tr>
<tr><td><a href="summarization_config.html"><code>SummarizationConfig</code></a></td><td>Config for <code>SummarizationExperiment</code>: judge model, checklist, and scoring knobs</td></tr>
<tr><td><a href="summarization_ranking.html"><code>SummarizationRanking</code></a></td><td>The corpus-level ranking a <code>SummarizationExperiment</code> run produces</td></tr>
<tr><td><a href="summarization_score.html"><code>SummarizationScore</code></a></td><td>One model's score and axes within a <code>SummarizationRanking</code></td></tr>
<tr><td><a href="llm_model_config.html"><code>LLMModelConfig</code></a></td><td>A hosted/self-hosted LLM model entry</td></tr>
<tr><td><a href="transformer_model_config.html"><code>TransformerModelConfig</code></a></td><td>A local transformer model entry</td></tr>
<tr><td><a href="few_shot_config.html"><code>FewShotConfig</code></a></td><td>Few-shot generation settings</td></tr>
<tr><td><a href="decompose_config.html"><code>DecomposeConfig</code></a></td><td>Per-field decomposition settings</td></tr>
<tr><td><a href="manipulation.html"><code>Manipulation</code></a></td><td>Enum of built-in prompt manipulations</td></tr>
</tbody>
</table>

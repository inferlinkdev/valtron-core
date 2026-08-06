ModelEval
=========

.. raw:: html

   <table class="api-table">
   <thead><tr><th>Method</th><th>Description</th></tr></thead>
   <tbody>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.run"><code>run()</code></a></td><td>Run the complete pipeline and save outputs (synchronous)</td></tr>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.arun"><code>arun()</code></a></td><td>Run the complete pipeline and save outputs (async)</td></tr>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.evaluate"><code>evaluate()</code></a></td><td>Run the evaluation pipeline (synchronous)</td></tr>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.aevaluate"><code>aevaluate()</code></a></td><td>Run the evaluation pipeline and store results (async)</td></tr>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.add_models"><code>add_models()</code></a></td><td>Add new models to the experiment</td></tr>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.reevaluate"><code>reevaluate()</code></a></td><td>Re-score stored predictions with a new field_metrics_config or ground truth</td></tr>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.save_experiment_results"><code>save_experiment_results()</code></a></td><td>Write the run directory (metadata.json + models/*.json)</td></tr>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.save_html_report"><code>save_html_report()</code></a></td><td>Generate the HTML report directly from in-memory results</td></tr>
   <tr><td><a href="#valtron_core.evaluation.model_eval.ModelEval.save_pdf_report"><code>save_pdf_report()</code></a></td><td>Generate the PDF report (and HTML) directly from in-memory results</td></tr>
   </tbody>
   </table>

.. autoclass:: valtron_core.evaluation.model_eval.ModelEval
   :members:
   :show-inheritance:

TradeoffAnalyzer
================

.. raw:: html

   <table class="api-table">
   <thead><tr><th>Method</th><th>Description</th></tr></thead>
   <tbody>
   <tr><td><a href="#valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer.from_model_eval"><code>from_model_eval()</code></a></td><td>Build a TradeoffAnalyzer from a completed ModelEval run (classmethod)</td></tr>
   <tr><td><a href="#valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer.from_data"><code>from_data()</code></a></td><td>Build a TradeoffAnalyzer from raw data and a transformer path (classmethod)</td></tr>
   <tr><td><a href="#valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer.analyze"><code>analyze()</code></a></td><td>Compute the tradeoff sweep and store results in memory</td></tr>
   <tr><td><a href="#valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer.aanalyze"><code>aanalyze()</code></a></td><td>Async variant of analyze()</td></tr>
   <tr><td><a href="#valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer.run"><code>run()</code></a></td><td>Analyze and write the tradeoff HTML report in one call</td></tr>
   <tr><td><a href="#valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer.arun"><code>arun()</code></a></td><td>Async variant of run()</td></tr>
   <tr><td><a href="#valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer.save_html_report"><code>save_html_report()</code></a></td><td>Write the tradeoff HTML report from in-memory sweep results</td></tr>
   <tr><td><a href="#valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer.save_json_report"><code>save_json_report()</code></a></td><td>Write the full sweep results as JSON</td></tr>
   </tbody>
   </table>

.. autoclass:: valtron_core.analysis.tradeoff_analyzer.TradeoffAnalyzer
   :members:
   :show-inheritance:

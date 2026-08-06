ObjectMetricConfig
===================

.. autoclass:: valtron_core.scoring.json_eval.ObjectMetricConfig
   :members:
   :show-inheritance:

Propagation Strategies
-----------------------

.. list-table::
   :header-rows: 1

   * - ``propagation``
     - Description
   * - ``"weighted_avg"`` (default)
     - Weighted mean of child scores
   * - ``"min"``
     - Score of the worst-scoring child
   * - ``"max"``
     - Score of the best-scoring child

``propagation`` also accepts any name registered via ``FieldMetricsConfig.custom_aggs``. See
:doc:`../user-guide/extraction/field-metrics/custom-evaluation`.

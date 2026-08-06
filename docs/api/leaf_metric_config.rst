LeafMetricConfig
================

.. autoclass:: valtron_core.scoring.json_eval.LeafMetricConfig
   :members:
   :show-inheritance:

Built-in Metrics
----------------

.. list-table::
   :header-rows: 1

   * - ``metric``
     - ``params``
     - Notes
   * - ``"exact"``
     - none
     - ``predicted == expected``, no normalization
   * - ``"threshold"``
     - - ``min`` (float)
     - Passes if ``actual >= min``; for numeric/confidence fields
   * - ``"exact_compare"``
     - - ``case_sensitive`` (default ``false``)
       - ``ignore_spaces`` (default ``false``)
     - Normalized string equality
   * - ``"text_similarity"``
     - - ``metric`` (``"fuzz_ratio"`` default, ``"bleu"``, ``"gleu"``, ``"cosine"``)
       - ``threshold`` (float or ``null`` for raw score)
       - ``case_sensitive``
       - ``ignore_spaces``
       - ``embedding_model`` (for ``"cosine"``)
     - Fuzzy match; ``"cosine"`` calls an embedding API
   * - ``"llm"``
     - - ``model`` (default ``"gpt-4o-mini"``)
       - ``prompt_template`` (optional)
     - One LLM call per field per document
   * - ``"embedding"``
     - - ``model`` (default ``"text-embedding-3-small"``)
       - ``threshold`` (float or ``null``)
     - Cosine similarity between embedding vectors; one API call per field per document

``metric`` also accepts any name registered via ``FieldMetricsConfig.custom_metrics``. See
:doc:`../user-guide/extraction/field-metrics/custom-evaluation`.

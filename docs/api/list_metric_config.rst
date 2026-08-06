ListMetricConfig
==================

.. autoclass:: valtron_core.scoring.json_eval.ListMetricConfig
   :members:
   :show-inheritance:

Alignment Settings
-------------------

Passed as flat keys directly inside ``metric_config``, alongside ``ordered`` and
``match_threshold``. See
:doc:`../user-guide/extraction/field-metrics/list-fields`.

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``match_key_fields``
     - ``null`` (auto)
     - Explicit identity fields to embed; auto-selected via LLM if unset
   * - ``match_key_model``
     - ``"gpt-5.4-mini"``
     - Model used for automatic match-key field selection
   * - ``embed_model``
     - ``"text-embedding-3-small"``
     - Model used to embed items for alignment
   * - ``lo``
     - ``0.35``
     - Minimum cosine similarity for a pair to be eligible

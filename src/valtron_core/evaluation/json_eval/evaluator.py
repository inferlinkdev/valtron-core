from __future__ import annotations
from typing import Any, Callable, cast
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import logging
import threading
from litellm import completion, completion_cost, embedding, ModelResponse
import numpy as np
from scipy.optimize import linear_sum_assignment

from valtron_core.evaluation.json_eval.alignment import (
    _cosine_matrix,
    _match_key_text,
    _truncate_for_prompt,
)
from valtron_core.evaluation.json_eval.schema import (
    AlignmentConfig,
    AlignmentItem,
    EvalResult,
    FieldConfig,
    ListMetricConfig,
    ObjectMetricConfig,
    _MatchKeyFields,
    _leaf_mc,
    _list_mc,
    _object_mc,
)
from valtron_core.evaluation.json_eval.registries import (
    DEFAULT_AGG_REGISTRY,
    DEFAULT_METRIC_REGISTRY,
    _run_comparator,
    _weighted_avg_field,
)
from valtron_core.evaluation.json_eval.validation import (
    _BUILTIN_METRIC_NAMES,
    _item_logic_uses_expensive_api,
    _scan_item_logic_for_expensive_metrics,
)

logger = logging.getLogger(__name__)


class JsonEvaluator:
    def __init__(
        self,
        custom_metrics: dict[str, Callable[..., Any]] | None = None,
        custom_aggs: dict[str, Callable[..., Any]] | None = None,
    ):
        self._template_vars: dict[str, Any] = {}
        self._evaluation_cost_usd: float = 0.0
        self._evaluation_cost_lock = threading.Lock()
        # Per-path cache of match-key field selections (one LLM call per list, reused per run).
        self._match_key_cache: dict[str, list[str] | None] = {}

        self.metric_registry: dict[str, Any] = {**DEFAULT_METRIC_REGISTRY}
        if custom_metrics:
            self.metric_registry.update(custom_metrics)

        self.agg_registry: dict[str, Any] = {**DEFAULT_AGG_REGISTRY}
        if custom_aggs:
            self.agg_registry.update(custom_aggs)

    def _record_evaluation_cost(self, cost_usd: float) -> None:
        with self._evaluation_cost_lock:
            self._evaluation_cost_usd += float(cost_usd or 0.0)

    @property
    def evaluation_cost(self) -> float:
        with self._evaluation_cost_lock:
            return self._evaluation_cost_usd

    def _comparator_metric(self, expected: Any, actual: Any, params: dict[str, Any]) -> tuple[float, bool]:
        score, is_correct, cost_usd, call_count = _run_comparator(expected, actual, params)
        if cost_usd:
            self._record_evaluation_cost(cost_usd)
        return score, is_correct

    def evaluate(
        self,
        config_dict: dict[str, Any] | str,
        expected: dict[str, Any] | str,
        actual: dict[str, Any] | str,
        extra_template_vars: dict[str, Any] | None = None,
    ) -> EvalResult:
        if isinstance(config_dict, str):
            config_dict = json.loads(config_dict)
        if isinstance(expected, str):
            expected = json.loads(expected)
        if isinstance(actual, str):
            actual = json.loads(actual)

        self._template_vars: dict[str, Any] = extra_template_vars or {}
        config = FieldConfig.model_validate(config_dict)
        return self._recurse(config, expected, actual, "root")

    def _recurse(self, config: FieldConfig, exp: Any, act: Any, path: str) -> EvalResult:
        if config.type == "object":
            return self._eval_object(config, _object_mc(config), exp, act, path)
        elif config.type == "list":
            return self._eval_list(config, _list_mc(config), exp, act, path)
        return self._eval_leaf(config, exp, act, path)

    def _eval_leaf(self, config: FieldConfig, exp: Any, act: Any, path: str) -> EvalResult:
        missing_exp = exp is None
        missing_act = act is None

        # Both missing -> neutral
        if missing_exp and missing_act:
            return EvalResult(
                path=path,
                score=1.0,
                weight=0.0,
                metric="optional_missing_both",
                tp=0.0 if config.optional else 1.0,
                is_correct=True,
            )

        # Missing expected
        if missing_exp:
            return EvalResult(
                path=path,
                score=1.0 if config.optional else 0.0,
                weight=0.0 if config.optional else config.weight,
                metric="unexpected_field",
                fp=0.0 if config.optional else 1.0,
                is_correct=config.optional,
            )

        # Missing actual
        if missing_act:
            return EvalResult(
                path=path,
                score=1.0 if config.optional else 0.0,
                weight=0.0 if config.optional else config.weight,
                metric="missing_field",
                fn=0.0 if config.optional else 1.0,
                is_correct=False,
            )

        # Normal metric evaluation
        m_cfg = _leaf_mc(config)
        metric_fn = self.metric_registry.get(m_cfg.metric, self.metric_registry["exact"])
        effective_params = {**m_cfg.params, "_template_vars": self._template_vars}
        result = metric_fn(exp, act, effective_params)
        extra_details: dict[str, Any] = {}
        if isinstance(result, tuple):
            if len(result) == 3:
                score, is_correct, extra_details = result
                cost = float(extra_details.get("cost") or 0.0)
                if cost:
                    self._record_evaluation_cost(cost)
            else:
                score, is_correct = result
        else:
            score = result
            is_correct = score == 1.0

        return EvalResult(
            path=path,
            score=score,
            weight=config.weight,
            metric=m_cfg.metric,
            tp=1.0 if is_correct else 0.0,
            fp=0.0 if is_correct else 1.0,
            fn=0.0 if is_correct else 1.0,
            precision=1.0 if is_correct else 0.0,
            recall=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            params=m_cfg.params,
            details=extra_details,
        )

    def _eval_object(self, config: FieldConfig, o_cfg: ObjectMetricConfig, exp: dict[str, Any], act: dict[str, Any], path: str) -> EvalResult:
        exp, act = (exp or {}), (act or {})
        fields = config.fields or {}
        child_results = {}
        eval_results = []
        for key in exp.keys():
            field_cfg = fields.get(key, FieldConfig())
            res = self._recurse(field_cfg, exp.get(key), act.get(key), f"{path}.{key}")
            child_results[key] = res
            eval_results.append(copy.deepcopy(res))

        agg_fn = self.agg_registry.get(o_cfg.propagation, lambda items: _weighted_avg_field(items, "score"))
        return EvalResult(
            path=path,
            score=agg_fn(eval_results),
            weight=config.weight,
            metric=f"agg:{o_cfg.propagation}",
            children=child_results,
            is_correct=all(res.is_correct for res in eval_results) and (bool(exp) or not bool(act)),
            precision=_weighted_avg_field(eval_results, "precision"),
            recall=_weighted_avg_field(eval_results, "recall"),
        )

    def _eval_list(self, config: FieldConfig, m_cfg: ListMetricConfig, exp: list[Any], act: list[Any], path: str) -> EvalResult:
        exp, act = (exp or []), (act or [])

        if not m_cfg.ordered and m_cfg.item_logic:
            custom_names = frozenset(set(self.metric_registry.keys()) - _BUILTIN_METRIC_NAMES)
            allowed = set(m_cfg.allow_expensive_comparisons_for or [])
            unallowed_expensive = [
                issue for issue in _scan_item_logic_for_expensive_metrics(
                    m_cfg.item_logic, path, custom_names
                )
                if issue["relative_path"] not in allowed
            ]
            if unallowed_expensive:
                fields = ", ".join(f'"{i["relative_path"]}"' for i in unallowed_expensive)
                raise ValueError(
                    f"Unordered list at '{path}' uses 3rd-party metric(s) on [{fields}] without "
                    f"explicit opt-in. These comparisons call an external API and add cost per document. "
                    f"Add these paths to allow_expensive_comparisons_for on the list's metric_config if "
                    f"you accept the extra cost, or replace the metric(s) with ones that don't call "
                    f"3rd-party APIs.\n\n"
                )

        if m_cfg.ordered:
            return self._eval_list_ordered(config, exp, act, path, m_cfg)

        if _item_logic_uses_expensive_api(m_cfg.item_logic):
            logger.info(
                "Unordered list at '%s': aligning items by embedding + Hungarian assignment "
                "because at least one leaf below calls a 3rd-party API (LLM or embedding).",
                path,
            )
            return self._eval_list_unordered_with_alignment(config, exp, act, path, m_cfg)

        return self._eval_list_unordered(config, exp, act, path, m_cfg)

    def _eval_list_ordered(self, config: FieldConfig, exp: list[Any], act: list[Any], path: str, m_cfg: ListMetricConfig) -> EvalResult:
        item_logic = m_cfg.item_logic
        assert item_logic is not None
        alignments: list[AlignmentItem] = []
        min_len = min(len(exp), len(act))
        for i in range(min_len):
            res = self._recurse(item_logic, exp[i], act[i], f"{path}[{i}]")
            alignments.append(AlignmentItem(e_idx=i, a_idx=i, score=res.score, result=res))

        matched = sum(1 for a in alignments if a.result.is_correct)

        # Add unmatched expected items so leaf-level fn is counted
        for i in range(min_len, len(exp)):
            res = self._recurse(item_logic, exp[i], None, f"{path}[{i}]")
            alignments.append(AlignmentItem(e_idx=i, a_idx=-1, score=0.0, result=res))

        precision = matched / len(act) if act else 0
        recall = matched / len(exp) if exp else 0

        soft_tp = sum(a.result.score for a in alignments if a.a_idx >= 0)
        soft_precision = soft_tp / len(act) if act else 0.0
        soft_recall = soft_tp / len(exp) if exp else 0.0
        soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0

        return EvalResult(
            path=path,
            score=soft_f1,
            weight=config.weight,
            metric="list_ordered_f1",
            alignment=alignments,
            precision=precision,
            recall=recall,
            fp=len(act) - matched,
            fn=len(exp) - matched,
            tp=matched,
            details={"matched_items": min_len},
            is_correct=(len(exp) == len(act) and all(a.result.is_correct for a in alignments)),
        )

    def _eval_list_unordered(self, config: FieldConfig, exp: list[Any], act: list[Any], path: str, m_cfg: ListMetricConfig) -> EvalResult:
        item_logic = m_cfg.item_logic
        assert item_logic is not None
        potential_matches: list[AlignmentItem] = []
        for i, e_item in enumerate(exp):
            for j, a_item in enumerate(act):
                # Pre-filter: if required_fields_to_match is set, check those fields
                # first. If any required field doesn't match, skip this pair -- the
                # actual item can't correspond to this expected item (counts as FP).
                if (
                    m_cfg.required_fields_to_match
                    and item_logic.fields
                    and isinstance(e_item, dict)
                    and isinstance(a_item, dict)
                ):
                    required_ok = True
                    for rf in m_cfg.required_fields_to_match:
                        rf_config = item_logic.fields.get(rf, FieldConfig())
                        rf_res = self._recurse(
                            rf_config,
                            e_item.get(rf),
                            a_item.get(rf),
                            f"{path}[{i}].{rf}",
                        )
                        if not rf_res.is_correct:
                            required_ok = False
                            break
                    if not required_ok:
                        continue

                res = self._recurse(item_logic, e_item, a_item, f"{path}[{i}]")
                potential_matches.append(AlignmentItem(e_idx=i, a_idx=j, score=res.score, result=res))

        potential_matches.sort(key=lambda x: x.score, reverse=True)
        matched_e: set[int] = set()
        matched_a: set[int] = set()
        alignments: list[AlignmentItem] = []

        for m in potential_matches:
            if m.e_idx not in matched_e and m.a_idx not in matched_a:
                if m.score >= m_cfg.match_threshold:
                    matched_e.add(m.e_idx)
                    matched_a.add(m.a_idx)
                    alignments.append(m)

        # Add unmatched expected items so leaf-level fn is counted
        for i in range(len(exp)):
            if i not in matched_e:
                res = self._recurse(item_logic, exp[i], None, f"{path}[{i}]")
                alignments.append(AlignmentItem(e_idx=i, a_idx=-1, score=0.0, result=res))

        precision = len(matched_a) / len(act) if act else 0
        recall = len(matched_e) / len(exp) if exp else 0

        soft_tp = sum(a.result.score for a in alignments if a.a_idx >= 0)
        soft_precision = soft_tp / len(act) if act else 0.0
        soft_recall = soft_tp / len(exp) if exp else 0.0
        soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0

        all_correct = len(matched_e) == len(exp) and len(matched_a) == len(act) and all(a.result.is_correct for a in alignments)
        return EvalResult(
            path=path,
            score=1.0 if all_correct else soft_f1,
            weight=config.weight,
            metric="list_greedy_f1",
            alignment=alignments,
            precision=precision,
            recall=recall,
            fp=len(act) - len(matched_a),
            fn=len(exp) - len(matched_e),
            tp=len(matched_e),
            details={"matched_items": len(matched_e)},
            is_correct=all_correct,
        )

    def _select_match_key_fields(
        self,
        m_cfg: ListMetricConfig,
        exp: list[Any],
        act: list[Any],
        path: str,
    ) -> tuple[list[str] | None, float]:
        """Decide which item fields to embed when aligning candidates for this list.

        Resolution order: explicit ``match_key_fields`` on the config wins; otherwise a single
        small-LLM call picks the identity-bearing fields from a few sample items. The result is
        cached per evaluation path -- the item schema is constant across a run, so this is
        roughly one LLM call per list for the whole run. Any failure (non-dict items, API
        error, empty/invalid selection) caches None, which tells :func:`_match_key_text` to
        fall back to the item's top-level scalar fields.

        :param m_cfg: The list's metric config.
        :param exp: Expected list of items.
        :param act: Actual list of items.
        :param path: Evaluation path, used as the cache key and in log messages.
        :return: ``(fields, cost_usd)`` -- field names to embed (or None), and the LLM cost incurred.
        """
        align_cfg = m_cfg.alignment or AlignmentConfig()
        if align_cfg.match_key_fields:
            return align_cfg.match_key_fields, 0.0

        if path in self._match_key_cache:
            return self._match_key_cache[path], 0.0

        samples = [it for it in (exp + act) if isinstance(it, dict)][:5]
        if not samples:
            self._match_key_cache[path] = None
            return None, 0.0

        # Nested items: skip the LLM field selection and return None, which tells
        # _match_key_text to characterize the item by its top-level scalar fields only (no
        # recursion into nested lists/dicts). The LLM selection only earns its keep on flat
        # items, where it can drop low-information scalar fields (enums, flags) that would
        # otherwise dilute the cosine signal.
        if any(isinstance(v, (dict, list)) for it in samples for v in it.values()):
            self._match_key_cache[path] = None
            return None, 0.0

        field_names = sorted({k for it in samples for k in it.keys()})
        if len(field_names) <= 1:
            selected = field_names or None
            self._match_key_cache[path] = selected
            return selected, 0.0

        samples_repr = "\n\n".join(
            "ITEM:\n"
            + "\n".join(f"  {k}: {_truncate_for_prompt(it[k])}" for k in field_names if k in it)
            for it in samples[:3]
        )
        prompt = (
            "You are configuring an automated evaluation pipeline. Below are sample items from "
            "a list. Pick the subset of fields that together IDENTIFY an item -- the fields a "
            "human would use to tell one item apart from another (e.g. a title, name, or "
            "description). Exclude low-information fields whose values repeat across items "
            "(enums, booleans, status flags, priorities) and bookkeeping fields (ids, indices) "
            "unless they are the only identifier.\n\n"
            f"Available fields: {field_names}\n\n"
            f"Sample items:\n{samples_repr}\n\n"
            "Return JSON only, matching the schema (a list of field names drawn from the "
            "available fields)."
        )

        selected: list[str] | None = None
        match_key_cost = 0.0
        try:
            response = cast(ModelResponse, completion(
                model=align_cfg.match_key_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format=_MatchKeyFields,
            ))
            try:
                match_key_cost = float(completion_cost(completion_response=response) or 0.0)
                self._record_evaluation_cost(match_key_cost)
            except Exception:  # noqa: BLE001
                logger.warning("match_key_cost_tracking_failed at '%s'", path)

            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty match-key selection response")
            parsed = _MatchKeyFields.model_validate_json(content)
            valid = [f for f in parsed.fields if f in field_names]
            selected = valid or None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Match-key field selection failed at '%s' (model=%s); embedding whole item: %s",
                path, m_cfg.alignment.match_key_model, e,
            )
            selected = None

        logger.info("Match-key fields for '%s': %s", path, selected if selected else "<whole item>")
        self._match_key_cache[path] = selected
        return selected, match_key_cost

    def _embed_texts(self, texts: list[str], model: str, path: str) -> tuple[list[list[float]], float]:
        """Embed a batch of texts in a single API call and record the spend.

        :param texts: Texts to embed (expected items followed by actual items).
        :param model: Embedding model name.
        :param path: Evaluation path, used in log messages.
        :return: ``(vectors, cost_usd)`` -- one embedding vector per input text, in input order.
        """
        response = embedding(model=model, input=texts)
        cost_usd = 0.0
        try:
            cost_usd = float(completion_cost(completion_response=response) or 0.0)
            self._record_evaluation_cost(cost_usd)
        except Exception:  # noqa: BLE001
            logger.warning("embed_cost_tracking_failed at '%s'", path)

        return [d["embedding"] for d in response.data], cost_usd

    def _align_by_hungarian(
        self,
        sims: np.ndarray,
        candidates_per_e: list[list[int]],
        lo: float,
    ) -> dict[int, int | None]:
        """Globally optimal one-to-one assignment over the cosine matrix (no LLM calls).

        Builds a cost matrix (cost = 1 - cosine) over only the pairs that pass the
        ``required_fields_to_match`` pre-filter and clear the ``lo`` floor; every other pair is
        forbidden with a prohibitively large cost. ``scipy.optimize.linear_sum_assignment`` then
        finds the assignment minimizing total cost (i.e. maximizing total cosine), and any
        forbidden pair it is forced to pick on a rectangular matrix is dropped to unmatched.

        :param sims: Expected-by-actual cosine similarity matrix.
        :param candidates_per_e: Per expected item, the allowed actual indices (pre-filter).
        :param lo: Minimum cosine for a pair to be eligible.
        :return: Mapping of each expected index to a chosen actual index or None.
        """
        n_exp = sims.shape[0]
        n_act = sims.shape[1] if sims.ndim == 2 else 0
        e_assignment: dict[int, int | None] = {i: None for i in range(n_exp)}
        if n_exp == 0 or n_act == 0:
            return e_assignment

        forbid = 1e6
        # Eligible pairs (cosine >= lo) cost 1 - cosine; everything else is prohibitively
        # expensive so the solver avoids it.
        cost = np.where(sims >= lo, 1.0 - sims, forbid)
        # Apply the required_fields_to_match pre-filter, only for rows it actually narrowed.
        for i, cands in enumerate(candidates_per_e):
            if len(cands) != n_act:
                disallowed = np.ones(n_act, dtype=bool)
                disallowed[list(cands)] = False
                cost[i, disallowed] = forbid

        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] < forbid:  # a real, eligible pair (not a forced forbidden slot)
                e_assignment[int(i)] = int(j)

        return e_assignment

    def _align_by_embedding(
        self,
        exp: list[Any],
        act: list[Any],
        candidates_per_e: list[list[int]],
        path: str,
        m_cfg: ListMetricConfig,
    ) -> tuple[dict[int, int | None], dict[str, Any]]:
        """Align expected->actual items by optimal one-to-one assignment over cosine similarity.

        One batched embedding call scores every expected item against every actual item by
        cosine similarity (over the match-key rendering of each item), then
        :meth:`_align_by_hungarian` finds the globally optimal assignment, leaving pairs below
        ``align_lo`` unmatched. No LLM aligner calls are made. If the embedding call fails at
        runtime, scoring degrades to all-unmatched (logged) rather than blocking the run.

        :param exp: Expected list of items.
        :param act: Actual list of items.
        :param candidates_per_e: Per expected item, the actual indices passing the
            ``required_fields_to_match`` pre-filter.
        :param path: Current evaluation path.
        :param m_cfg: The list's metric config (supplies the embedding model and align_lo).
        :return: A ``(e_assignment, stats)`` tuple, where ``e_assignment`` maps each expected
            index to a chosen actual index or None, and ``stats`` carries diagnostic counts.
        """
        n_exp, n_act = len(exp), len(act)
        e_assignment: dict[int, int | None] = {i: None for i in range(n_exp)}
        stats: dict[str, Any] = {
            "n_matched": 0, "n_embed_calls": 0,
            "embedding_ok": False, "match_key_fields": None,
            "match_key_cost": 0.0, "embedding_cost": 0.0,
        }

        if n_exp == 0 or n_act == 0:
            return e_assignment, stats

        align_cfg = m_cfg.alignment or AlignmentConfig()

        # One batched embedding call -> cosine matrix. On failure, leave everything unmatched
        # rather than blocking the run.
        try:
            fields, match_key_cost = self._select_match_key_fields(m_cfg, exp, act, path)
            texts = [_match_key_text(it, fields) for it in exp] + [
                _match_key_text(it, fields) for it in act
            ]
            vecs, embedding_cost = self._embed_texts(texts, align_cfg.embed_model, path)
            sims = _cosine_matrix(vecs[:n_exp], vecs[n_exp:])
            stats["n_embed_calls"] = 1
            stats["embedding_ok"] = True
            stats["match_key_fields"] = fields
            stats["match_key_cost"] = match_key_cost
            stats["embedding_cost"] = embedding_cost
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Embedding alignment unavailable at '%s'; leaving items unmatched: %s", path, e,
            )
            return e_assignment, stats

        e_assignment = self._align_by_hungarian(sims, candidates_per_e, align_cfg.lo)
        stats["n_matched"] = sum(1 for v in e_assignment.values() if v is not None)
        return e_assignment, stats

    def _eval_list_unordered_with_alignment(
        self,
        config: FieldConfig,
        exp: list[Any],
        act: list[Any],
        path: str,
        m_cfg: ListMetricConfig,
    ) -> EvalResult:
        """Evaluate an unordered list whose items contain LLM-judge leaves, aligning items by
        optimal assignment over embedding cosine similarity, then judging each matched pair.

        Alignment is delegated to :meth:`_align_by_embedding`: one batched embedding call scores
        all expected x actual pairs by cosine similarity and the Hungarian algorithm finds the
        globally optimal one-to-one assignment -- no LLM aligner calls. Once alignment is decided,
        each matched pair is evaluated by recursing ``item_logic`` so every leaf (LLM-judge,
        embedding, or local) runs through its own configured metric with its own prompt template.

        :param config: The list's :class:`FieldConfig`.
        :param exp: Expected list of items.
        :param act: Actual (predicted) list of items.
        :param path: Current evaluation path.
        :param m_cfg: The list's metric config.
        :return: An :class:`EvalResult` for the list.
        """
        if not exp and not act:
            return EvalResult(
                path=path, score=1.0, weight=config.weight,
                metric="list_embed_hungarian_f1",
                alignment=[], precision=0.0, recall=0.0,
                tp=0, fp=0, fn=0,
                details={"matched_items": 0},
                is_correct=True,
            )

        if not exp:
            return EvalResult(
                path=path, score=0.0, weight=config.weight,
                metric="list_embed_hungarian_f1",
                alignment=[], precision=0.0, recall=0.0,
                fp=len(act), fn=0, tp=0,
                details={"matched_items": 0},
                is_correct=False,
            )

        item_logic = m_cfg.item_logic
        assert item_logic is not None
        required_fields = m_cfg.required_fields_to_match or []

        # Per-E candidate filtering by required_fields_to_match. Skipping a candidate
        # here is functionally identical to the LLM returning matched_a_idx=null for it,
        # but cheaper and deterministic.
        candidates_per_e: list[list[int]] = []
        for i, e_item in enumerate(exp):
            if (
                required_fields
                and item_logic
                and item_logic.fields
                and isinstance(e_item, dict)
            ):
                cands: list[int] = []
                for j, a_item in enumerate(act):
                    if not isinstance(a_item, dict):
                        continue
                    ok = True
                    for rf in required_fields:
                        rf_cfg = item_logic.fields.get(rf, FieldConfig())
                        rf_res = self._recurse(
                            rf_cfg, e_item.get(rf), a_item.get(rf),
                            f"{path}[{i}].{rf}",
                        )
                        if not rf_res.is_correct:
                            ok = False
                            break
                    if ok:
                        cands.append(j)
                candidates_per_e.append(cands)
            else:
                candidates_per_e.append(list(range(len(act))))

        # Optimal one-to-one assignment over embedding cosine decides the alignment
        # (no LLM aligner calls; see _align_by_embedding).
        max_workers = min(32, max(1, len(exp)))
        e_assignment, align_stats = self._align_by_embedding(
            exp, act, candidates_per_e, path, m_cfg
        )

        # Run the leaf-judging recursion for every expected item in parallel. Each
        # _recurse call makes its own LLM-judge / embedding / local-metric calls
        # internally; running 32 of them concurrently lets the API round-trips overlap
        # rather than blocking serially on the main thread. Output order is preserved
        # by ex.map() so the alignments list stays e_idx-ordered.
        def _judge_one(i: int) -> AlignmentItem:
            a_idx = e_assignment[i]
            if a_idx is not None:
                res = self._recurse(item_logic, exp[i], act[a_idx], f"{path}[{i}]")
                return AlignmentItem(e_idx=i, a_idx=a_idx, score=res.score, result=res)
            res = self._recurse(item_logic, exp[i], None, f"{path}[{i}]")
            return AlignmentItem(e_idx=i, a_idx=-1, score=0.0, result=res)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            alignments: list[AlignmentItem] = list(ex.map(_judge_one, range(len(exp))))

        matched_e: set[int] = {a.e_idx for a in alignments if a.a_idx >= 0}
        matched_a: set[int] = {a.a_idx for a in alignments if a.a_idx >= 0}

        precision = len(matched_a) / len(act) if act else 0
        recall = len(matched_e) / len(exp) if exp else 0

        soft_tp = sum(a.result.score for a in alignments if a.a_idx >= 0)
        soft_precision = soft_tp / len(act) if act else 0.0
        soft_recall = soft_tp / len(exp) if exp else 0.0
        soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0

        all_correct = (
            len(matched_e) == len(exp)
            and len(matched_a) == len(act)
            and all(a.result.is_correct for a in alignments)
        )

        return EvalResult(
            path=path,
            score=1.0 if all_correct else soft_f1,
            weight=config.weight,
            metric="list_embed_hungarian_f1",
            alignment=alignments,
            precision=precision,
            recall=recall,
            fp=len(act) - len(matched_a),
            fn=len(exp) - len(matched_e),
            tp=len(matched_e),
            details={
                "matched_items": len(matched_e),
                "align_method": "embed_hungarian" if align_stats["embedding_ok"] else "embedding_unavailable",
                "embedding_model": m_cfg.alignment.embed_model if m_cfg.alignment else None,
                "match_key_fields": align_stats["match_key_fields"],
                "n_matched": align_stats["n_matched"],
                "n_embed_calls": align_stats["n_embed_calls"],
                "match_key_cost": align_stats.get("match_key_cost", 0.0),
                "embedding_cost": align_stats.get("embedding_cost", 0.0),
            },
            is_correct=all_correct,
        )

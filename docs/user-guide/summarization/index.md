# Summarization

```{toctree}
:hidden:

data-format
config-format
evaluation-api
evaluation-results
report-formats
```

A summarization task is one where the model turns a large piece of text into a smaller one that still captures what matters in it: condensing a contract into its key terms, a meeting transcript into its decisions, or a news article into its lead. Unlike classification or extraction, the output isn't a label or a set of fields; it is free-form prose.

That shape is what makes summarization hard to score. There is no single correct answer to check a model's output against. Two summaries of the same document can differ in wording, length, and structure and both be excellent, so there's nothing to diff a candidate's output against the way classification checks a label or extraction checks a field.

Summarization experiments are expressed in Valtron by giving each document just an `id` and its `content`, no `label` required, and running it through `SummarizationExperiment`. In place of a label, you supply a `judge_model`: an LLM that reads each source document, decides which facts a good summary would have to convey, and grades every candidate's summary against that judgment. This is what makes the method reference-free. The target a summary is scored against comes from the source document itself, not from a human-written reference summary or a panel of other models.

Every candidate summary is graded on four axes:

- **correctness**: what fraction of the summary's own claims are supported by the source document
- **salient_coverage**: of the facts a good summary must convey, how many did the candidate summary convey
- **salient_precision**: of what the summary said, how much of it landed on that must-convey material rather than padding
- **requirements_met**: optional, how much of a per-document-class checklist the summary satisfied

These four axes combine into a single score under the `salience-f+reqs` scheme:

$$
\text{score} =
\begin{cases}
0 & \text{if correctness} < \text{gate} \\[4pt]
(1 - w) \cdot F_\beta(\text{salient\_coverage}, \text{salient\_precision}) + w \cdot \text{requirements\_met} & \text{with a checklist} \\[4pt]
F_\beta(\text{salient\_coverage}, \text{salient\_precision}) & \text{without one}
\end{cases}
$$

where `w` is `requirement_weight` and $F_\beta$ is the harmonic mean of the two:

$$
F_\beta(\text{salient\_coverage}, \text{salient\_precision}) = (1 + \beta^2) \cdot \frac{\text{salient\_precision} \cdot \text{salient\_coverage}}{\beta^2 \cdot \text{salient\_precision} + \text{salient\_coverage}}
$$

If a summary's `correctness` score is below `gate` (`0.5` by default), the overall score is zero, no matter how well the summary covers the source document's must-convey facts. `correctness` is not averaged together with the other three axes; falling below the gate overrides them completely. In practice, this means a summary that reads well but states things the source document doesn't support cannot outscore a summary that is accurate but plain.

Above that gate, `salient_coverage` and `salient_precision` are combined using the harmonic mean rather than a plain average, so a summary can't raise its score by padding in extra content at the cost of precision.

If a `requirements` checklist is configured, `requirements_met` is added to the score as a separate quantity, weighted by `requirement_weight`, rather than being folded into how coverage or precision are computed. If no checklist is configured, `requirements_met` doesn't factor into the score at all, and the score is just the harmonic mean of coverage and precision described above.

See [Evaluation Results](./evaluation-results.md#reading-the-ranking) for how these numbers come off an actual run, and [`EvaluationMetrics`](../../api/evaluation_metrics) for where each document's axes land.

```{rubric} Meeting Summary Example
```

```python
from valtron_core.evaluation import SummarizationExperiment

data = [
    {
        "id": "1",
        "content": (
            "The city council voted 6-3 on Tuesday to approve a $40 million bond "
            "measure funding two new elementary schools and a renovation of the "
            "downtown library. Construction is expected to begin in March 2027 and "
            "finish before the 2029 school year. Opponents argued the bond would "
            "raise property taxes by roughly 0.2%, while supporters pointed to "
            "overcrowding at the district's existing schools."
        ),
    },
]

config = {
    "models": [{"name": "gpt-4o-mini"}, {"name": "gpt-4o"}],
    "judge_model": "gpt-4o",
}

experiment = SummarizationExperiment(config=config, data=data)
report_path = experiment.run("./results")

for candidate in experiment.ranking.scores:
    axes = ", ".join(
        f"{name}={value:.0%}" if value is not None else f"{name}=n/a"
        for name, value in candidate.axes().items()
    )
    print(f"{candidate.model:<12} score={candidate.score:.0%}  {axes}")
```

```text
gpt-4o        score=91%  correctness=100%, salient_coverage=88%, salient_precision=92%, requirements_met=n/a
gpt-4o-mini   score=78%  correctness=100%, salient_coverage=71%, salient_precision=85%, requirements_met=n/a
```

`run("./results")` runs the same pipeline shape as the other recipes: it validates the data, extracts each document's facts once through `judge_model`, has every candidate model write a summary, grades it on the four axes above, and writes results and reports to `./results`. `experiment.ranking` is where the cross-model comparison lives, built from those same four axes averaged over the whole document set, in place of the per-document `is_correct` and `accuracy` fields classification and extraction report. By default, `config` uses the default prompt [`SALIENCE_SUMMARY_PROMPT`](./config-format). You can overwrite it with your own.

```{rubric} In this chapter
```

Beyond the two-model, single-document example above, a summarization run can score a per-document-class checklist alongside the four axes, tune how faithfulness and coverage trade off, and be reweighted or regraded after the fact without re-generating a single summary. Each of those is covered in this chapter:

- **[3.1 Data Format](./data-format)**: how documents are structured with no `label` involved.
- **[3.2 Config Format](./config-format)**: the full [`SummarizationConfig`](../../api/summarization_config) schema, including the judge model, the requirements checklist, and the scoring parameters.
- **[3.3 Evaluation API](./evaluation-api)**: [`SummarizationExperiment`](../../api/summarization_experiment) itself, incremental evaluation, and reweighting or regrading a run with `reevaluate()`.
- **[3.4 Evaluation Results](./evaluation-results)**: the [`SummarizationRanking`](../../api/summarization_ranking) a run produces, and where the four axes land on `EvaluationMetrics`.
- **[3.5 Report Formats](./report-formats)**: the HTML and PDF reports, including per-document judge decisions.

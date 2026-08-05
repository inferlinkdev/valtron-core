# Examples

```{toctree}
:hidden:

sentiment-classification
affiliation-extraction
transformer-comparison
tradeoff-analysis
multimodal-molecules
incremental-evaluation
```

Each example is a self-contained Python file in the [`examples/`](https://github.com/your-org/valtron-core/tree/main/examples) directory of the repository. Install the package, set your API keys, and run any script directly.

| Example | Task type | Key concept |
|---|---|---|
| [Sentiment Classification](./sentiment-classification) | Classification | Minimal setup, multiple LLMs |
| [Affiliation Extraction](./affiliation-extraction) | Structured extraction | Multi-institution grading, field metrics |
| [Transformer Comparison](./transformer-comparison) | Classification | Train DistilBERT, compare to cloud LLMs |
| [Tradeoff Analysis](./tradeoff-analysis) | Classification | Train, evaluate, then find cost/accuracy routing thresholds |
| [Multimodal Molecules](./multimodal-molecules) | Multimodal classification | Image attachments, vision-capable models |
| [Incremental Evaluation](./incremental-evaluation) | Any | Load a prior run, add models, regenerate report |

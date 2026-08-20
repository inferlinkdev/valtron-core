# Summarization test fixtures

## billsum/

Three US Congressional bill texts, used by `test_experiment.py` to exercise the
summarization recipe against real prose rather than invented sentences.

- **Source:** BillSum (Kornilova & Eidelman, 2019), `FiscalNote/billsum` on
  HuggingFace, `test` split.
- **License:** US Government works — public domain. Bill text only; the
  dataset's reference summaries are not included here and are not needed, since
  this method is reference free.
- **Why these three:** they are among the shortest documents in the corpus
  (738–914 words), which keeps the fixtures small enough to read.

Other corpora used in the research that produced this method are deliberately
absent. They carry third-party copyright — BBC (XSum), CNN/Daily Mail, critic
reviews (Rotten Tomatoes), per-paper licenses (arXiv, PubMed) — and one,
SAMSum, is CC BY-NC-ND, which is incompatible with redistribution in an
Apache-2.0 package.

The requirements checklist in `test_experiment.py` is the one authored for the
billsum document class; it is not part of the upstream benchmark.

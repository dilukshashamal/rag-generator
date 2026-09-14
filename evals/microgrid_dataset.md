# Microgrid evaluation: 10 cases

`microgrid_dataset.json` contains exactly eight answerable cases and two unanswerable
cases, based only on `microgrid_operations_and_response_guide.pdf`. The first two
questions reproduce the reported failures. Reference answers preserve operating
conditions and safety exceptions; PDF layout whitespace is normalized for readability.

| Case | Coverage | Reference pages |
| --- | --- | --- |
| 01 | Normal battery state-of-charge range | 1–2 |
| 02 | Low and critical battery reserve responses | 2–3 |
| 03 | Generator automatic start and earlier-start condition | 2, 4 |
| 04 | Critical-alarm acknowledgement and escalation | 1, 4 |
| 05 | Island-mode monitoring interval | 2 |
| 06 | Battery thermal-alarm precautions | 3–4 |
| 07 | Stale telemetry and verification before control decisions | 4 |
| 08 | Quarterly generator maintenance and owner | 4 |
| 09 | Generator purchase price: unspecified | No answer |
| 10 | Battery warranty duration: unspecified | No answer |

To use the app's Evaluation button, set these environment values and recreate the
app services so the worker loads the selected dataset:

```dotenv
EVAL_DATASET_PATH=./evals/microgrid_dataset.json
EVAL_MAX_QUESTIONS=10
```

Select the indexed microgrid collection, open **Evaluation**, and click **Run
evaluation**. Dataset preparation does not queue an evaluation. The separate security
fixture checks remain part of the existing evaluator; they are not extra RAG questions.
Other evaluation settings, including the optional RAGAS judge, are unchanged.

## Interpreting results

The current evaluator labels sources as `filename|heading`. These PDF pages have
no extracted Markdown-style heading, so their compatible label is
`microgrid_operations_and_response_guide.pdf|None`. Retrieval scores therefore measure
document-level relevance, not page-level relevance. Review citation pages using the
table above. The original HR dataset remains available in `evals/dataset.json`.

The existing lexical answer metric checks an exact reference substring; it is not a
semantic correctness score. Extracted line breaks, multiple cited passages, or an
equivalent supported wording can score zero. Its citation/faithfulness checks also
use exact whitespace, unlike the answer validator's whitespace-tolerant fallback.
Review those results alongside actual answers and any enabled semantic judge metrics.

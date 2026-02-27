#!/usr/bin/env python3
"""Evaluate all runs against qrels."""
import pyterrier as pt
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from ir_measures import MAP, nDCG, P, RR

ensure_pyterrier_is_loaded(is_offline=False)

ds = pt.datasets.get_dataset('irds:ir-lab-wise-2025/radboud-validation-20251114-training')
qrels = ds.get_qrels()
topics = ds.get_topics('title')

bm25 = pt.io.read_results('output/runs/radboud-validation-20251114-training/pyterrier-BM25-on-default_text/run.txt.gz')
rm3 = pt.io.read_results('/Users/justin/Desktop/Repositories/InfoRet/wows-code/ecir26/je-alkonume-huehner-01-rm3/template-new-approach/output/runs/radboud-validation-20251114-training/pyterrier-BM25-rm3-on-default_text/run.txt.gz')
llm_70b = pt.io.read_results('output/runs/radboud-validation-20251114-training/pyterrier-BM25-llm-llama3.1-70b-d50-on-default_text/run.txt.gz')
gemma2 = pt.io.read_results('output/runs/radboud-validation-20251114-training/pyterrier-BM25-llm-gemma2-9b-d200-on-default_text/run.txt.gz')

metrics = [MAP, nDCG, nDCG@10, P@10, RR@10]

results = pt.Experiment(
    [bm25, rm3, llm_70b, gemma2],
    topics,
    qrels,
    eval_metrics=metrics,
    names=['BM25 (Baseline)', 'BM25 + RM3', 'BM25 + LLM Llama3.1:70b (d50)', 'BM25 + LLM Gemma2:9b (d200)'],
)

print(results.to_string(index=False))

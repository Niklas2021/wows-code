#!/usr/bin/env python3
"""
Two-stage retrieval: BM25 → LLM Reranking via Ollama.

Stage 1: BM25 retrieves top-K candidate documents.
Stage 2: A local LLM (via Ollama) scores each query–document pair for relevance.

Usage example:
    python rerank_llm.py \
        --dataset radboud-validation-20251114-training \
        --output output \
        --rerank-depth 50 \
        --ollama-model llama3.1:8b \
        --text-field-to-retrieve default_text
"""

import click
import json
import re
import pyterrier as pt
from pathlib import Path
from tirex_tracker import tracking, ExportFormat
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm
import requests
import pandas as pd


# ---------------------------------------------------------------------------
# Indexing (same as 00-bm25, but with text in meta for reranking)
# ---------------------------------------------------------------------------

def extract_text_of_document(doc, field: str) -> str:
    if field == "default_text":
        return doc.default_text()
    elif field == "title":
        return doc.title
    elif field == "description":
        return doc.description
    return ""


def get_index(dataset_id: str, field: str, output_path: Path):
    index_dir = output_path / "indexes" / f"{dataset_id}-on-{field}"
    if not index_dir.is_dir():
        print("Building index ...")
        index_dir.mkdir(parents=True, exist_ok=True)
        dataset_obj = ir_datasets.load(f"ir-lab-wise-2025/{dataset_id}")

        def docs_iter():
            for doc in tqdm(dataset_obj.docs_iter(), "Pre-Process Documents"):
                yield {
                    "docno": doc.doc_id,
                    "text": extract_text_of_document(doc, field) or "",
                }

        with tracking(
            export_file_path=index_dir / "index-metadata.yml",
            export_format=ExportFormat.IR_METADATA,
        ):
            pt.IterDictIndexer(
                str(index_dir.absolute()),
                meta={"docno": 100, "text": 8192},
                verbose=True,
            ).index(docs_iter())

    return pt.IndexFactory.of(str(index_dir.absolute()))


# ---------------------------------------------------------------------------
# LLM Reranking via Ollama
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = """You are a search relevance expert. Given a search query and a document, rate how relevant the document is to the query.

Query: {query}

Document (first 1500 chars):
{document}

Rate the relevance on a scale from 0 to 100, where:
- 0 = completely irrelevant
- 50 = somewhat relevant
- 100 = perfectly relevant

Be precise — use the full range and avoid round numbers when possible.

Respond with ONLY a JSON object: {{"score": <number>}}"""


def score_with_ollama(
    query: str,
    document: str,
    model: str,
    ollama_url: str,
    max_doc_chars: int = 1500,
) -> float:
    """Ask Ollama to score a single query-document pair."""
    doc_text = document[:max_doc_chars]
    prompt = RELEVANCE_PROMPT.format(query=query, document=doc_text)

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 50},
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json()["response"].strip()

        # Try to parse JSON from the response
        # Handle cases where LLM wraps in markdown code blocks
        answer = re.sub(r"```json\s*", "", answer)
        answer = re.sub(r"```\s*", "", answer)

        parsed = json.loads(answer)
        return float(parsed["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: try to find a number in the response
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", answer)
        if numbers:
            return min(float(numbers[0]), 100.0)
        return 0.0
    except requests.RequestException as e:
        print(f"  [WARN] Ollama request failed: {e}")
        return 0.0


def rerank_with_llm(
    run_df: pd.DataFrame,
    index,
    model: str,
    ollama_url: str,
    max_doc_chars: int,
) -> pd.DataFrame:
    """Rerank a PyTerrier run DataFrame using an Ollama LLM."""
    # Load document texts from the index meta
    text_loader = pt.text.get_text(index, "text")
    run_with_text = text_loader(run_df)

    results = []
    queries = run_with_text.groupby("qid")
    
    for qid, group in tqdm(queries, desc="LLM Reranking queries"):
        query_text = group.iloc[0]["query"]
        
        for _, row in tqdm(
            group.iterrows(),
            total=len(group),
            desc=f"  Query {qid}",
            leave=False,
        ):
            llm_score = score_with_ollama(
                query=query_text,
                document=row.get("text", ""),
                model=model,
                ollama_url=ollama_url,
                max_doc_chars=max_doc_chars,
            )
            # Combine LLM score with normalized BM25 score as tiebreaker
            # LLM score dominates (0-100), BM25 adds tiny fraction (0-0.99)
            bm25_score = row.get("score", 0.0)
            results.append({
                "qid": row["qid"],
                "query": query_text,
                "docno": row["docno"],
                "llm_score": llm_score,
                "bm25_score": bm25_score,
            })

    reranked = pd.DataFrame(results)
    # Normalize BM25 scores per query to 0-0.99 range as tiebreaker
    for qid in reranked["qid"].unique():
        mask = reranked["qid"] == qid
        bm25_vals = reranked.loc[mask, "bm25_score"]
        bm25_min, bm25_max = bm25_vals.min(), bm25_vals.max()
        if bm25_max > bm25_min:
            reranked.loc[mask, "bm25_norm"] = (bm25_vals - bm25_min) / (bm25_max - bm25_min) * 0.99
        else:
            reranked.loc[mask, "bm25_norm"] = 0.0
    reranked["score"] = reranked["llm_score"] + reranked.get("bm25_norm", 0.0)
    reranked = reranked.sort_values(["qid", "score"], ascending=[True, False])
    reranked["rank"] = reranked.groupby("qid").cumcount()
    return reranked


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    output: Path,
    index,
    dataset_id: str,
    retrieval_model: str,
    text_field: str,
    rerank_depth: int,
    ollama_model: str,
    ollama_url: str,
    max_doc_chars: int,
):
    tag = f"pyterrier-{retrieval_model}-llm-{ollama_model.replace(':', '-').replace('/', '-')}-d{rerank_depth}-on-{text_field}"
    target_dir = output / "runs" / dataset_id / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "run.txt.gz"

    if target_file.exists():
        print(f"Run already exists: {target_file}")
        return

    # Stage 1: BM25 retrieval
    topics = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset_id}").get_topics("title")
    retriever = pt.terrier.Retriever(index, wmodel=retrieval_model, num_results=rerank_depth)

    print(f"Stage 1: {retrieval_model} retrieval (top {rerank_depth}) ...")
    bm25_run = retriever(topics)

    # Stage 2: LLM reranking
    print(f"Stage 2: LLM reranking with {ollama_model} ...")
    description = (
        f"Two-stage retrieval. "
        f"Stage 1: {retrieval_model} on {text_field} (top {rerank_depth}). "
        f"Stage 2: LLM reranking via Ollama model={ollama_model}, "
        f"max_doc_chars={max_doc_chars}."
    )

    with tracking(
        export_file_path=target_dir / "ir-metadata.yml",
        export_format=ExportFormat.IR_METADATA,
        system_description=description,
        system_name=tag,
    ):
        reranked = rerank_with_llm(
            bm25_run, index, ollama_model, ollama_url, max_doc_chars
        )

    reranked["run_id"] = tag
    pt.io.write_results(reranked, target_file)
    print(f"Done! Run saved to {target_file}")


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["radboud-validation-20251114-training", "spot-check-20251122-training"]),
    required=True,
    help="The dataset.",
)
@click.option("--output", type=Path, default=Path("output"), help="The output directory.")
@click.option("--retrieval-model", type=str, default="BM25", help="Stage 1 retrieval model.")
@click.option(
    "--text-field-to-retrieve",
    type=click.Choice(["default_text", "title", "description"]),
    default="default_text",
    help="Which text field to index and retrieve on.",
)
@click.option("--rerank-depth", type=int, default=200, help="How many top docs to rerank with the LLM.")
@click.option("--ollama-model", type=str, default="llama3.1:8b", help="Ollama model name (e.g. llama3.1:8b, mistral, phi3:14b, llama3.1:70b).")
@click.option("--ollama-url", type=str, default="http://localhost:11434", help="Ollama API URL.")
@click.option("--max-doc-chars", type=int, default=1500, help="Max chars of document text to send to the LLM.")
def main(dataset, output, retrieval_model, text_field_to_retrieve, rerank_depth, ollama_model, ollama_url, max_doc_chars):
    ensure_pyterrier_is_loaded(is_offline=False)
    index = get_index(dataset, text_field_to_retrieve, output)
    run_pipeline(
        output, index, dataset, retrieval_model, text_field_to_retrieve,
        rerank_depth, ollama_model, ollama_url, max_doc_chars,
    )


if __name__ == "__main__":
    main()

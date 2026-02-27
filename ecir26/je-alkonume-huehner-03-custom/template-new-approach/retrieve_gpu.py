#!/usr/bin/env python3
"""
MonoT5 GPU Reranking: BM25 Top-K → MonoT5 Reranking with CUDA.

Designed to run inside a Docker container with --gpus all.
Can also be run natively if you have CUDA + all dependencies.

Usage (Docker):
    docker run --gpus all -v ./output:/app/output monot5-gpu \
        --dataset radboud-validation-20251114-training \
        --rerank-depth 1000 \
        --monot5-batch-size 64

Usage (native):
    python retrieve_gpu.py \
        --dataset radboud-validation-20251114-training \
        --output output \
        --rerank-depth 1000 \
        --monot5-batch-size 64
"""

import click
import torch
import pyterrier as pt
from pathlib import Path
from tirex_tracker import tracking, ExportFormat
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm
import shutil


def extract_text_of_document(doc, field: str) -> str:
    if field == "default_text":
        return doc.default_text()
    if field == "title":
        return doc.title
    if field == "description":
        return doc.description
    return ""


def get_index(dataset_id: str, field: str, output_path: Path, force_reindex: bool):
    index_dir = output_path / "indexes" / f"{dataset_id}-on-{field}"

    if force_reindex and index_dir.is_dir():
        shutil.rmtree(index_dir)

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


def run_retrieval(
    output: Path,
    index,
    dataset_id: str,
    retrieval_model: str,
    text_field_to_retrieve: str,
    rerank_depth: int,
    monot5_model: str,
    monot5_batch_size: int,
):
    # Detect GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Topics/queries
    topics = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset_id}").get_topics("title")

    # Stage 1: BM25 retrieval
    retriever = pt.terrier.Retriever(index, wmodel=retrieval_model, num_results=rerank_depth)

    # Stage 2: MonoT5 reranking
    from pyterrier_t5 import MonoT5ReRanker

    text_loader = pt.text.get_text(index, "text")
    reranker = MonoT5ReRanker(
        model=monot5_model,
        batch_size=monot5_batch_size,
        verbose=True,
        text_field="text",
    )

    pipeline = retriever >> text_loader >> reranker

    tag = f"pyterrier-{retrieval_model}-monot5d{rerank_depth}-gpu-on-{text_field_to_retrieve}"
    target_dir = output / "runs" / dataset_id / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "run.txt.gz"

    if target_file.exists():
        print(f"Run already exists: {target_file}")
        return

    description = (
        f"Initial retrieval: {retrieval_model} on {text_field_to_retrieve} (top {rerank_depth}); "
        f"Reranking: MonoT5 (model={monot5_model}, batch_size={monot5_batch_size}, device={device})."
    )

    print(f"Stage 1: {retrieval_model} retrieval (top {rerank_depth}) ...")
    print(f"Stage 2: MonoT5 reranking (batch_size={monot5_batch_size}, device={device}) ...")

    with tracking(
        export_file_path=target_dir / "ir-metadata.yml",
        export_format=ExportFormat.IR_METADATA,
        system_description=description,
        system_name=tag,
    ):
        run = pipeline(topics)

    # Fix rank column (MonoT5 updates scores but not ranks)
    run = run.sort_values(["qid", "score"], ascending=[True, False])
    run["rank"] = run.groupby("qid").cumcount()
    run["run_id"] = tag
    pt.io.write_results(run, target_file)
    print(f"Done! Run saved to {target_file}")
    print(f"  Total results: {len(run)} rows")


@click.command()
@click.option(
    "--dataset",
    type=click.Choice([
        "radboud-validation-20251114-training",
        "spot-check-20251122-training",
    ]),
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
@click.option("--rerank-depth", type=int, default=1000, help="How many top docs to rerank with MonoT5.")
@click.option("--monot5-model", type=str, default="castorini/monot5-base-msmarco", help="HuggingFace model name.")
@click.option("--monot5-batch-size", type=int, default=64, help="Batch size for MonoT5 (increase for GPU).")
@click.option("--force-reindex/--no-force-reindex", default=False, help="Rebuild index from scratch.")
def main(
    dataset,
    output,
    retrieval_model,
    text_field_to_retrieve,
    rerank_depth,
    monot5_model,
    monot5_batch_size,
    force_reindex,
):
    ensure_pyterrier_is_loaded(is_offline=False)
    index = get_index(dataset, text_field_to_retrieve, output, force_reindex)
    run_retrieval(
        output,
        index,
        dataset,
        retrieval_model,
        text_field_to_retrieve,
        rerank_depth,
        monot5_model,
        monot5_batch_size,
    )


if __name__ == "__main__":
    main()

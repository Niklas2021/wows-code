#!/usr/bin/env python3
import click
import pyterrier as pt
from pathlib import Path
from tirex_tracker import tracking, ExportFormat
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm
import shutil


def extract_text_of_document(doc, field: str) -> str:
    # Here you can modify the document representation
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
        print("Build new index")
        index_dir.mkdir(parents=True, exist_ok=True)

        dataset_obj = ir_datasets.load(f"ir-lab-wise-2025/{dataset_id}")

        def docs_iter():
            for doc in tqdm(dataset_obj.docs_iter(), "Pre-Process Documents"):
                yield {
                    "docno": doc.doc_id,
                    # store the chosen field as text for reranking later
                    "text": extract_text_of_document(doc, field) or "",
                }

        # IMPORTANT for reranking: we store "text" as meta so we can retrieve it later
        with tracking(export_file_path=index_dir / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
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
    monot5_verbose: bool,
):
    # Topics/queries
    topics = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset_id}").get_topics("title")

    # Stage 1: initial retrieval (top-K)
    retriever = pt.terrier.Retriever(index, wmodel=retrieval_model, num_results=rerank_depth)

    # Stage 2: load text + MonoT5 reranking
    try:
        from pyterrier_t5 import MonoT5ReRanker
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for reranking. Install with:\n"
            "  python3.14 -m pip install -U pyterrier-t5\n"
            f"Original error: {e}"
        )

    text_loader = pt.text.get_text(index, "text")
    reranker = MonoT5ReRanker(
        model=monot5_model,
        batch_size=monot5_batch_size,
        verbose=monot5_verbose,
        text_field="text",
    )

    pipeline = retriever >> text_loader >> reranker

    tag = f"pyterrier-{retrieval_model}-monot5d{rerank_depth}-on-{text_field_to_retrieve}"
    target_dir = output / "runs" / dataset_id / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "run.txt.gz"

    if target_file.exists():
        return

    description = (
        f"Initial retrieval: {retrieval_model} on {text_field_to_retrieve} (top {rerank_depth}); "
        f"Reranking: MonoT5 (model={monot5_model}, batch_size={monot5_batch_size})."
    )

    with tracking(
        export_file_path=target_dir / "ir-metadata.yml",
        export_format=ExportFormat.IR_METADATA,
        system_description=description,
        system_name=tag,
    ):
        run = pipeline(topics)

    run["run_id"] = tag
    pt.io.write_results(run, target_file)


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["radboud-validation-20251114-training", "spot-check-20251122-training"]),
    required=True,
    help="The dataset.",
)
@click.option("--output", type=Path, required=False, default=Path("output"), help="The output directory.")
@click.option("--retrieval-model", type=str, default="BM25", required=False, help="The retrieval model (e.g., BM25, PL2, DirichletLM).")
@click.option("--text-field-to-retrieve", type=click.Choice(["default_text", "title", "description"]), required=False, default="default_text", help="The text field of the documents on which to retrieve.")
@click.option("--rerank-depth", type=int, default=100, required=False, help="How many top documents to rerank with MonoT5.")
@click.option("--monot5-model", type=str, default="castorini/monot5-base-msmarco", required=False, help="HF model name for MonoT5.")
@click.option("--monot5-batch-size", type=int, default=4, required=False, help="Batch size for MonoT5 reranker.")
@click.option("--monot5-verbose/--no-monot5-verbose", default=False, required=False, help="Show MonoT5 progress bar.")
@click.option("--force-reindex/--no-force-reindex", default=False, required=False, help="Delete and rebuild index (useful if you changed meta fields).")
def main(
    dataset,
    text_field_to_retrieve,
    retrieval_model,
    output,
    rerank_depth,
    monot5_model,
    monot5_batch_size,
    monot5_verbose,
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
        monot5_verbose,
    )


if __name__ == "__main__":
    main()
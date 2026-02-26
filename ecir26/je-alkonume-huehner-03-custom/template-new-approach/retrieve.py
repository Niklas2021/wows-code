#!/usr/bin/env python3
import click
import pyterrier as pt
from pathlib import Path
from tirex_tracker import tracking, ExportFormat
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm
import inspect
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
        print("Build new index")
        index_dir.mkdir(parents=True, exist_ok=True)

        dataset_obj = ir_datasets.load(f"ir-lab-wise-2025/{dataset_id}")

        def docs_iter():
            for doc in tqdm(dataset_obj.docs_iter(), "Pre-Process Documents"):
                yield {"docno": doc.doc_id, "text": extract_text_of_document(doc, field) or ""}

        with tracking(export_file_path=index_dir / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
            pt.IterDictIndexer(
                str(index_dir.absolute()),
                meta={"docno": 100, "text": 8192},
                verbose=True,
            ).index(docs_iter())

    return pt.IndexFactory.of(str(index_dir.absolute()))


def build_rm3(index, fb_docs: int, fb_terms: int, orig_weight: float):
    """
    Build an RM3 operator in a version-tolerant way:
    some PyTerrier versions accept orig_query_weight, others orig_weight, some neither.
    """
    rm3_kwargs = {"fb_docs": fb_docs, "fb_terms": fb_terms}
    sig = inspect.signature(pt.rewrite.RM3.__init__).parameters

    applied = False
    if "orig_query_weight" in sig:
        rm3_kwargs["orig_query_weight"] = orig_weight
        applied = True
    elif "orig_weight" in sig:
        rm3_kwargs["orig_weight"] = orig_weight
        applied = True

    return pt.rewrite.RM3(index, **rm3_kwargs), applied


def run_retrieval(
    output: Path,
    index,
    dataset_id: str,
    retrieval_model: str,
    text_field_to_retrieve: str,
    topic_field: str,
    num_results: int,
    bm25_k1: float,
    bm25_b: float,
    rm3: bool,
    rm3_fb_docs: int,
    rm3_fb_terms: int,
    rm3_orig_weight: float,
):
    # topics/queries
    topics = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset_id}").get_topics(topic_field)

    # controls/properties for Terrier
    controls = {}
    properties = {}

    # BM25 tuning (only relevant if BM25)
    if retrieval_model.upper() == "BM25":
        controls["bm25.b"] = bm25_b
        # most Terrier setups accept this as a property
        properties["bm25.k_1"] = str(bm25_k1)

    base_retriever = pt.terrier.Retriever(
        index,
        wmodel=retrieval_model,
        controls=controls,
        properties=properties,
        num_results=num_results,
    )

    pipeline = base_retriever
    rm3_applied = False
    if rm3:
        rm3_op, rm3_applied = build_rm3(index, rm3_fb_docs, rm3_fb_terms, rm3_orig_weight)
        pipeline = base_retriever >> rm3_op >> base_retriever

    tag = (
        f"pyterrier-{retrieval_model}"
        f"-k1{bm25_k1}-b{bm25_b}"
        f"-q{topic_field}"
        f"-rm3{int(rm3)}"
        f"-on-{text_field_to_retrieve}"
    )

    target_dir = output / "runs" / dataset_id / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "run.txt.gz"

    if target_file.exists():
        return

    description = (
        f"Retriever={retrieval_model} field={text_field_to_retrieve} topics={topic_field} "
        f"num_results={num_results}. "
        f"BM25(k1={bm25_k1}, b={bm25_b}) if applicable. "
        f"RM3={rm3} fb_docs={rm3_fb_docs} fb_terms={rm3_fb_terms} "
        f"orig_w={rm3_orig_weight} applied={rm3_applied}."
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
@click.option("--topic-field", type=click.Choice(["title", "description", "narrative"]), required=False, default="title", help="Which topic field to use as query text.")
@click.option("--num-results", type=int, required=False, default=1000, help="How many results per query to return.")
@click.option("--bm25-k1", type=float, required=False, default=1.2, help="BM25 k1 parameter (only used if retrieval-model=BM25).")
@click.option("--bm25-b", type=float, required=False, default=0.75, help="BM25 b parameter (only used if retrieval-model=BM25).")
@click.option("--rm3/--no-rm3", default=False, required=False, help="Enable RM3 pseudo relevance feedback.")
@click.option("--rm3-fb-docs", type=int, default=10, required=False, help="RM3: number of feedback documents.")
@click.option("--rm3-fb-terms", type=int, default=20, required=False, help="RM3: number of feedback terms.")
@click.option("--rm3-orig-weight", type=float, default=0.5, required=False, help="RM3: original query weight (if supported by your PyTerrier version).")
@click.option("--force-reindex/--no-force-reindex", default=False, required=False, help="Delete and rebuild index.")
def main(
    dataset,
    output,
    retrieval_model,
    text_field_to_retrieve,
    topic_field,
    num_results,
    bm25_k1,
    bm25_b,
    rm3,
    rm3_fb_docs,
    rm3_fb_terms,
    rm3_orig_weight,
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
        topic_field,
        num_results,
        bm25_k1,
        bm25_b,
        rm3,
        rm3_fb_docs,
        rm3_fb_terms,
        rm3_orig_weight,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import click
import pyterrier as pt
from pathlib import Path
from tirex_tracker import tracking, ExportFormat
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm
import inspect


def extract_text_of_document(doc, field):
    # ToDo: here one can make modifications to the document representations
    if field == "default_text":
        return doc.default_text()
    elif field == "title":
        return doc.title
    elif field == "description":
        return doc.description


def get_index(dataset, field, output_path):
    index_dir = output_path / "indexes" / f"{dataset}-on-{field}"
    if not index_dir.is_dir():
        print("Build new index")
        docs = []
        dataset = ir_datasets.load(f"ir-lab-wise-2025/{dataset}")

        for doc in tqdm(dataset.docs_iter(), "Pre-Process Documents"):
            docs.append({"docno": doc.doc_id, "text": extract_text_of_document(doc, field)})

        with tracking(export_file_path=index_dir / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
            pt.IterDictIndexer(str(index_dir.absolute()), meta={'docno' : 100}, verbose=True).index(docs)

    return pt.IndexFactory.of(str(index_dir.absolute()))


def run_retrieval(output, index, dataset, retrieval_model, text_field_to_retrieve,
                  rm3, rm3_fb_docs, rm3_fb_terms, rm3_orig_weight):    
    tag = f"pyterrier-{retrieval_model}-on-{text_field_to_retrieve}"
    target_dir = output / "runs" / dataset / tag
    target_file = target_dir / "run.txt.gz"

    if target_file.exists():
        return

    topics = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset}").get_topics("title")
    retriever = pt.terrier.Retriever(index, wmodel=retrieval_model)

    pipeline = retriever
    tag_suffix = "rm3"

    if rm3:
        rm3_kwargs = {
            "fb_docs": rm3_fb_docs,
            "fb_terms": rm3_fb_terms,
        }

        rm3_sig = inspect.signature(pt.rewrite.RM3.__init__).parameters
        orig_weight_applied = False

        if "orig_query_weight" in rm3_sig:
            rm3_kwargs["orig_query_weight"] = rm3_orig_weight
            orig_weight_applied = True
        elif "orig_weight" in rm3_sig:
            rm3_kwargs["orig_weight"] = rm3_orig_weight
            orig_weight_applied = True
        # falls beides nicht existiert: wir ignorieren rm3_orig_weight (sonst crash)

        rm3_op = pt.rewrite.RM3(index, **rm3_kwargs)

        pipeline = retriever >> rm3_op >> retriever
        tag_suffix = "rm3"
    else:
        orig_weight_applied = False
        tag_suffix = "no-rm3"

    tag = f"pyterrier-{retrieval_model}-{tag_suffix}-on-{text_field_to_retrieve}"
    target_dir = output / "runs" / dataset / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "run.txt.gz"

    description = (
        f"PyTerrier {retrieval_model} on {text_field_to_retrieve}. "
        f"RM3={rm3} fb_docs={rm3_fb_docs} fb_terms={rm3_fb_terms} "
        f"orig_w={rm3_orig_weight} applied={orig_weight_applied}."
    )
    
    with tracking(export_file_path=target_dir / "ir-metadata.yml",
              export_format=ExportFormat.IR_METADATA,
              system_description=description,
              system_name=tag):
        run = pipeline(topics)

    run["run_id"] = tag
    pt.io.write_results(run, target_file)


@click.command()
@click.option("--dataset", type=click.Choice(["radboud-validation-20251114-training", "spot-check-20251122-training"]), required=True, help="The dataset.")
@click.option("--output", type=Path, required=False, default=Path("output"), help="The output directory.")
@click.option("--retrieval-model", type=str, default="BM25", required=False, help="The retrieval model (e.g., BM25, PL2, DirichletLM).")
@click.option("--text-field-to-retrieve", type=click.Choice(["default_text", "title", "description"]), required=False, default="default_text", help="The text field of the documents on which to retrieve.")
@click.option("--rm3/--no-rm3", default=True, required=False, help="Enable RM3 pseudo relevance feedback.")
@click.option("--rm3-fb-docs", type=int, default=10, required=False, help="RM3: number of feedback documents.")
@click.option("--rm3-fb-terms", type=int, default=20, required=False, help="RM3: number of feedback terms.")
@click.option("--rm3-orig-weight", type=float, default=0.5, required=False, help="RM3: original query weight.")
def main(dataset, text_field_to_retrieve, retrieval_model, output, rm3, rm3_fb_docs, rm3_fb_terms, rm3_orig_weight):
    ensure_pyterrier_is_loaded(is_offline=False)

    index = get_index(dataset, text_field_to_retrieve, output)
    run_retrieval(output, index, dataset, retrieval_model, text_field_to_retrieve,
                rm3, rm3_fb_docs, rm3_fb_terms, rm3_orig_weight)    

if __name__ == '__main__':
    main()


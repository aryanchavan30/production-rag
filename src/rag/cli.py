from pathlib import Path
import typer
from rich.console import Console
from rag import setup_logging
setup_logging()
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from langchain_openai import ChatOpenAI
from rag.config import settings
from rag.ingestion.pipeline import IngestionPipeline
from rag.graph.graph import AdaptiveRAGGraph
from rag.generation.generator import generate
from rag.ingestion.embedder import Embedder
from rag.ingestion.indexer import BM25Indexer, QdrantIndexer
from rag.retrieval.reranker import BGEReranker
from rag.retrieval.retriever import HybridRetriever

app = typer.Typer(help="Production-ready local RAG system CLI", no_args_is_help=True)
console = Console()


@app.callback()
def _():
    """RAG CLI — ingest documents, query the knowledge base."""


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),
    collection: str = typer.Option(settings.qdrant_collection, help="Qdrant collection name"),
    context: bool = typer.Option(False, "--context/--no-context", help="Add LLM context prefixes (slow on CPU)"),
    llm: str = typer.Option(settings.llm_model, help="LLM model name for context generation"),
    llm_base_url: str = typer.Option(settings.llm_base_url, help="OpenAI-compatible base URL (Ollama or vLLM)"),
    chunk_size: int = typer.Option(settings.chunk_size, help="Max tokens per chunk"),
):
    """Ingest documents into the RAG knowledge base."""
    if not path.exists():
        console.print(f"[red]Error:[/red] Path '{path}' does not exist.")
        raise typer.Exit(1)

    if context:
        console.print(f"[yellow]Context mode ON[/yellow] — calling {llm} at {llm_base_url}. This is slow on CPU.")

    pipeline = IngestionPipeline(
        qdrant_url=settings.qdrant_url,
        collection_name=collection,
        embed_dim=settings.embed_dim,
        bm25_index_path=settings.bm25_index_path,
        chunk_size=chunk_size,
        semantic_threshold=settings.semantic_threshold,
        add_context=context,
        llm_model=llm,
        llm_base_url=llm_base_url,
        embed_model_path=settings.embed_model_path,
        chunker_embed_model_path=settings.chunker_embed_model_path,
    )

    console.print(f"[cyan]Ingesting:[/cyan] {path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing documents...", total=None)

        if path.is_dir():
            result = pipeline.ingest_directory(path)
        else:
            result = pipeline.ingest_file(path)

        progress.update(task, completed=True)

    table = Table(title="Ingestion Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files processed", str(result.files_processed))
    table.add_row("Chunks indexed", str(result.chunks_indexed))
    table.add_row("Files skipped", str(result.files_skipped))
    table.add_row("Errors", str(len(result.errors)))
    console.print(table)

    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for err in result.errors:
            console.print(f"  • {err}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the knowledge base"),
    collection: str = typer.Option(settings.qdrant_collection, help="Qdrant collection name"),
    top_k: int = typer.Option(5, help="Number of documents to retrieve and rerank"),
    show_sources: bool = typer.Option(False, "--sources/--no-sources", help="Show retrieved source chunks"),
):
    """Query the RAG knowledge base and get an answer."""
    console.print(f"[cyan]Question:[/cyan] {question}\n")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading pipeline...", total=None)

        llm = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key="ollama",
            temperature=0,
        )
        embedder = Embedder(model_path=settings.embed_model_path)
        qdrant = QdrantIndexer(
            url=settings.qdrant_url,
            collection_name=collection,
            embed_dim=settings.embed_dim,
        )
        bm25 = BM25Indexer.load(settings.bm25_index_path)
        retriever = HybridRetriever(qdrant=qdrant, bm25=bm25, embedder=embedder)
        reranker = BGEReranker(model_path=settings.reranker_model_path)
        rag_graph = AdaptiveRAGGraph(retriever=retriever, reranker=reranker, llm=llm, rerank_top_k=top_k)

        progress.update(task, description="Retrieving documents...")
        state = rag_graph.run_retrieval(question)

        progress.update(task, description="Generating answer...")
        answer = generate(state["question"], state["documents"], llm)

        progress.update(task, completed=True)

    console.print(f"[bold green]Answer:[/bold green]\n{answer}\n")

    if show_sources:
        console.print("[bold]Sources:[/bold]")
        for i, doc in enumerate(state["documents"]):
            console.print(f"\n[cyan][{i+1}] {doc.get('filename', doc['source'])}[/cyan]")
            console.print(doc["text"][:300] + "...")


if __name__ == "__main__":
    app()

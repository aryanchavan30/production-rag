from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
PLAIN_TEXT_EXTENSIONS = {".txt", ".md"}
DOCLING_EXTENSIONS = {".pdf", ".docx"}


@dataclass
class ParsedDocument:
    text: str
    source: str
    filename: str


def parse_document(file_path: Path) -> ParsedDocument:
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    if ext in PLAIN_TEXT_EXTENSIONS:
        text = file_path.read_text(encoding="utf-8")
    else:
        text = _parse_with_docling(file_path)

    return ParsedDocument(
        text=text.strip(),
        source=str(file_path),
        filename=file_path.name,
    )


def parse_directory(directory: Path) -> list[ParsedDocument]:
    directory = Path(directory)
    documents = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(parse_document(file_path))
    return documents


def _parse_with_docling(file_path: Path) -> str:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    return result.document.export_to_markdown()

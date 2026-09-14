"""Validate container signatures and parser output before accepting an upload."""

import io
import re
import zipfile
from pathlib import PurePath

from defusedxml import ElementTree
from pypdf import PdfReader

from src.config import Settings
from src.models.schemas import Section
from src.security.questions import ValidationError

MIME_TYPES = {
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
}
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def safe_filename(filename: str) -> str:
    name = PurePath(filename.replace("\\", "/")).name
    name = re.sub(r"[^\w .()-]", "_", name, flags=re.UNICODE)[:255]
    if not name or name in {".", ".."}:
        raise ValidationError("Provide a valid filename.")
    return name


def _text_sections(text: str, *, page: int | None = None) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            if "\n".join(lines).strip():
                sections.append(Section(text="\n".join(lines).strip(), page=page, heading=heading))
            heading = re.sub(r"^#{1,6}\s+", "", line)[:500]
            lines = [line]
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append(Section(text="\n".join(lines).strip(), page=page, heading=heading))
    return sections


def extract(filename: str, mime: str, data: bytes, settings: Settings) -> list[Section]:
    extension = PurePath(filename).suffix.lower()
    if extension not in MIME_TYPES:
        raise ValidationError("Supported files are PDF, DOCX, TXT and Markdown.")
    if mime.split(";")[0].lower() not in MIME_TYPES[extension]:
        raise ValidationError("The file MIME type does not match its extension.")
    if not data:
        raise ValidationError("Empty files cannot be uploaded.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ValidationError(f"Files must be at most {settings.max_upload_mb} MB.")
    try:
        if extension == ".pdf":
            if not data.startswith(b"%PDF-"):
                raise ValidationError("The file is not a valid PDF.")
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise ValidationError("Password-protected PDFs are unsupported. Upload an unlocked copy.")
            if len(reader.pages) > settings.max_pages:
                raise ValidationError(f"Documents must contain at most {settings.max_pages} pages.")
            sections = []
            extracted_size = 0
            for number, page in enumerate(reader.pages, 1):
                content = page.get_contents()
                if content and len(content.get_data()) > settings.max_extracted_characters * 4:
                    raise ValidationError("This PDF page is too complex to extract safely.")
                text = page.extract_text() or ""
                extracted_size += len(text)
                if extracted_size > settings.max_extracted_characters:
                    raise ValidationError("The document contains too much extracted text.")
                sections.extend(_text_sections(text, page=number))
        elif extension == ".docx":
            if not data.startswith(b"PK\x03\x04"):
                raise ValidationError("The file is not a valid DOCX archive.")
            sections = _docx_sections(data, settings)
        else:
            if b"\x00" in data or data.startswith((b"MZ", b"PK\x03\x04", b"%PDF-")):
                raise ValidationError("The file is not a plain text document.")
            text = data.decode("utf-8-sig")
            if any(ord(c) < 32 and c not in "\n\r\t\f" for c in text):
                raise ValidationError("The document contains binary control characters.")
            pages = text.split("\f")
            if len(pages) > settings.max_pages:
                raise ValidationError(f"Documents must contain at most {settings.max_pages} pages.")
            sections = [s for i, page in enumerate(pages, 1)
                        for s in _text_sections(page, page=i if len(pages) > 1 else None)]
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("The file is invalid or unreadable. Export a fresh, unencrypted copy.") from None
    total = sum(len(s.text) for s in sections)
    if total < 10 or not any(any(c.isalnum() for c in s.text) for s in sections):
        raise ValidationError("No usable text was found. Scanned PDFs need OCR before upload.")
    if total > settings.max_extracted_characters:
        raise ValidationError("The document contains too much extracted text.")
    return sections


def _docx_sections(data: bytes, settings: Settings) -> list[Section]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = archive.infolist()
        if len(files) > 2000 or sum(f.file_size for f in files) > settings.max_upload_mb * 1024 * 1024 * 5:
            raise ValidationError("The DOCX archive exceeds extraction limits.")
        names = {f.filename for f in files}
        if not {"[Content_Types].xml", "word/document.xml"}.issubset(names):
            raise ValidationError("The archive is not a DOCX document.")
        if any("vbaProject" in name or "embeddings/" in name for name in names):
            raise ValidationError("Documents containing macros or embedded objects are unsupported.")
        if "docProps/app.xml" in names:
            props = ElementTree.fromstring(archive.read("docProps/app.xml"))
            for node in props.iter():
                if node.tag.endswith("}Pages") and int(node.text or "0") > settings.max_pages:
                    raise ValidationError("The DOCX exceeds the page limit.")
        root = ElementTree.fromstring(archive.read("word/document.xml"))
        output: list[Section] = []
        heading: str | None = None
        page = 1
        buffer: list[str] = []
        for paragraph in root.iter(f"{WORD_NS}p"):
            text = "".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t"))
            style = paragraph.find(f"{WORD_NS}pPr/{WORD_NS}pStyle")
            is_heading = style is not None and "heading" in style.get(f"{WORD_NS}val", "").lower()
            breaks = sum(1 for node in paragraph.iter()
                         if node.tag == f"{WORD_NS}lastRenderedPageBreak"
                         or (node.tag == f"{WORD_NS}br" and node.get(f"{WORD_NS}type") == "page"))
            if (is_heading or breaks) and buffer:
                output.append(Section(text="\n".join(buffer), page=page, heading=heading))
                buffer = []
            if is_heading:
                heading = text[:500]
            if text.strip():
                buffer.append(text)
            if breaks:
                if buffer:
                    output.append(Section(text="\n".join(buffer), page=page, heading=heading))
                    buffer = []
                page += breaks
                if page > settings.max_pages:
                    raise ValidationError("The DOCX exceeds the page limit.")
        if buffer:
            output.append(Section(text="\n".join(buffer), page=page, heading=heading))
        return output

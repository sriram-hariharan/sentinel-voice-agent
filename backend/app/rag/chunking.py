import hashlib
import re
from pathlib import Path

from backend.app.rag.models import PolicyChunk, PolicyDocument, PolicySection

_REQUIRED_METADATA = {
    "policy_id",
    "title",
    "version",
    "effective_date",
}
_SECTION_HEADING = re.compile(r"^##\s+(.+?)\s*$")
_SLUG_CHARACTER = re.compile(r"[^a-z0-9]+")


class PolicyDocumentError(ValueError):
    """Raised when a version-controlled policy document is malformed."""


def content_hash(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise PolicyDocumentError("Policy document requires frontmatter")

    end = normalized.find("\n---\n", 4)
    if end < 0:
        raise PolicyDocumentError("Policy frontmatter is not terminated")

    metadata: dict[str, str] = {}
    for line in normalized[4:end].splitlines():
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise PolicyDocumentError(
                f"Malformed policy frontmatter line: {line!r}"
            )
        metadata[key.strip()] = value.strip().strip('"').strip("'")

    missing = _REQUIRED_METADATA - metadata.keys()
    if missing:
        raise PolicyDocumentError(
            f"Missing policy metadata: {', '.join(sorted(missing))}"
        )

    return metadata, normalized[end + 5 :].strip()


def _parse_sections(body: str) -> tuple[PolicySection, ...]:
    sections: list[PolicySection] = []
    heading = "Overview"
    lines: list[str] = []

    def flush() -> None:
        content = "\n".join(lines).strip()
        if content:
            sections.append(PolicySection(heading=heading, content=content))

    for line in body.splitlines():
        match = _SECTION_HEADING.match(line)
        if match:
            flush()
            heading = match.group(1).strip()
            lines = []
            continue
        if line.startswith("# "):
            continue
        lines.append(line)

    flush()
    if not sections:
        raise PolicyDocumentError("Policy document contains no section content")
    return tuple(sections)


def parse_policy_document(path: Path) -> PolicyDocument:
    try:
        raw = path.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(raw)
        return PolicyDocument(
            policy_id=metadata["policy_id"],
            title=metadata["title"],
            version=metadata["version"],
            effective_date=metadata["effective_date"],
            source_path=str(path),
            content_hash=content_hash(raw),
            sections=_parse_sections(body),
        )
    except PolicyDocumentError:
        raise
    except Exception as exc:
        raise PolicyDocumentError(
            f"Could not parse policy document {path}"
        ) from exc


def load_policy_documents(directory: Path) -> list[PolicyDocument]:
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise PolicyDocumentError(
            f"No Markdown policy documents found in {directory}"
        )

    documents = [parse_policy_document(path) for path in paths]
    identifiers = [document.policy_id for document in documents]
    if len(identifiers) != len(set(identifiers)):
        raise PolicyDocumentError("Policy IDs must be unique")
    return documents


def _section_slug(value: str) -> str:
    slug = _SLUG_CHARACTER.sub("-", value.lower()).strip("-")
    return slug[:60] or "section"


def _split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    words = paragraph.split()
    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        added = len(word) + (1 if current else 0)
        if current and length + added > max_chars:
            pieces.append(" ".join(current))
            current = []
            length = 0
        current.append(word)
        length += len(word) + (1 if length else 0)

    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_policy_document(
    document: PolicyDocument,
    *,
    max_chars: int = 1200,
) -> list[PolicyChunk]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")

    chunks: list[PolicyChunk] = []
    document_index = 0
    section_slug_counts: dict[str, int] = {}

    for section in document.sections:
        paragraphs: list[str] = []
        for paragraph in re.split(r"\n\s*\n", section.content):
            normalized = " ".join(paragraph.split())
            if not normalized:
                continue
            if len(normalized) <= max_chars:
                paragraphs.append(normalized)
            else:
                paragraphs.extend(
                    _split_oversized_paragraph(normalized, max_chars)
                )

        grouped: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if current and len(candidate) > max_chars:
                grouped.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            grouped.append(current)

        base_section_slug = _section_slug(section.heading)
        section_occurrence = section_slug_counts.get(base_section_slug, 0)
        section_slug_counts[base_section_slug] = section_occurrence + 1
        section_slug = (
            base_section_slug
            if section_occurrence == 0
            else f"{base_section_slug}-{section_occurrence:02d}"
        )
        for section_index, text in enumerate(grouped):
            chunks.append(
                PolicyChunk(
                    chunk_id=(
                        f"{document.policy_id}:{section_slug}:{section_index:02d}"
                    ),
                    policy_id=document.policy_id,
                    title=document.title,
                    version=document.version,
                    effective_date=document.effective_date,
                    section=section.heading,
                    chunk_index=document_index,
                    content=text,
                )
            )
            document_index += 1

    return chunks

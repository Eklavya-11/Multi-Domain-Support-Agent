"""
Corpus loader — reads every Markdown article shipped in data/ and splits
each one into retrieval-friendly chunks with rich metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from config import DATA_DIR, CHUNK_MAX_CHARS


@dataclass
class Chunk:
    """One retrieval unit — a section (or part of a section) of a support article."""
    text: str
    company: str          # hackerrank | claude | visa
    category: str         # top-level product area derived from path
    source_file: str      # relative path inside data/
    title: str            # article title extracted from first heading


def _extract_title(text: str) -> str:
    """Pull the first markdown heading as the article title."""
    match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"


def _category_from_path(rel_path: Path) -> str:
    """
    Derive a product-area label from the path relative to the company dir.
    e.g.  screen/managing-tests/foo.md  →  screen
          claude/account-management/bar.md  →  account_management
    """
    parts = rel_path.parts[:-1]  # drop the filename
    if not parts:
        return "general"
    # Use the first sub-folder, normalise hyphens to underscores
    return parts[0].replace("-", "_")


def _split_by_sections(text: str) -> List[str]:
    """Split markdown text at ## headings, keeping each heading with its body."""
    sections: List[str] = []
    current: List[str] = []
    for line in text.splitlines(keepends=True):
        if re.match(r"^#{1,3}\s", line) and current:
            sections.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current))
    return sections


def _hard_split(text: str, max_chars: int) -> List[str]:
    """If a section is still too long, split on paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    pieces: List[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 > max_chars and buf:
            pieces.append(buf.strip())
            buf = para
        else:
            buf = buf + "\n\n" + para if buf else para
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def _chunk_article(text: str, company: str, category: str,
                   source_file: str, title: str) -> List[Chunk]:
    """Chunk one article into retrieval-friendly pieces."""
    sections = _split_by_sections(text)
    chunks: List[Chunk] = []
    for sec in sections:
        if len(sec.strip()) < 30:
            continue  # skip trivially short sections
        if len(sec) <= CHUNK_MAX_CHARS:
            chunks.append(Chunk(
                text=sec.strip(), company=company, category=category,
                source_file=source_file, title=title,
            ))
        else:
            for piece in _hard_split(sec, CHUNK_MAX_CHARS):
                chunks.append(Chunk(
                    text=piece, company=company, category=category,
                    source_file=source_file, title=title,
                ))
    return chunks


def load_corpus() -> List[Chunk]:
    """
    Walk data/{hackerrank,claude,visa}/**/*.md, chunk every article, and
    return a flat list of Chunk objects ready for embedding.
    """
    all_chunks: List[Chunk] = []
    for company_dir in sorted(DATA_DIR.iterdir()):
        if not company_dir.is_dir():
            continue
        company = company_dir.name
        for md_file in sorted(company_dir.rglob("*.md")):
            if md_file.name == "index.md":
                continue
            rel = md_file.relative_to(company_dir)
            category = _category_from_path(rel)
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            title = _extract_title(text)
            chunks = _chunk_article(text, company, category,
                                    str(rel), title)
            all_chunks.extend(chunks)
    print(f"[corpus] Loaded {len(all_chunks)} chunks from "
          f"{sum(1 for _ in DATA_DIR.rglob('*.md'))} files")
    return all_chunks

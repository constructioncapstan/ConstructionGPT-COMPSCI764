"""Render technical PDF pages for vision experiments."""

from __future__ import annotations

from pathlib import Path

import fitz


def render_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    selected_pages: list[int] | None = None,
) -> list[Path]:
    """Render a PDF to PNG pages.

    selected_pages uses zero-based page indices. If omitted, all pages are rendered.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages = selected_pages if selected_pages is not None else list(range(len(doc)))

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    outputs: list[Path] = []

    for page_index in pages:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out = output_dir / f"page_{page_index + 1:03d}.png"
        pix.save(out)
        outputs.append(out)

    return outputs

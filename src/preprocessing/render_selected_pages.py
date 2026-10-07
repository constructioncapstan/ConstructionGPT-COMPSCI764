"""Render selected consent-plan pages for ConstructionGPT.

Raw plans are private and must remain outside version control.

Expected private layout:
data/private/raw_plans/
  P01/
    arch.pdf
    struct.pdf
  ...
  P08/
    arch.pdf
    struct.pdf
    framing.pdf

Usage:
    python -m src.preprocessing.render_selected_pages \
        --manifest data/plan_manifest_public.csv \
        --raw-root data/private/raw_plans \
        --output-root data/private/rendered_plans
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def render_page(
    pdf_path: Path,
    page_number: int,
    full_path: Path,
    retrieval_path: Path,
    dpi: int,
    crop_left: float,
    crop_top: float,
    crop_right: float,
    crop_bottom: float,
) -> tuple[int, int]:
    doc = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(
                f"{pdf_path}: requested page {page_number}, but PDF has {len(doc)} pages"
            )

        page = doc[page_number - 1]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(full_path), jpg_quality=90)
    finally:
        doc.close()

    image = Image.open(full_path).convert("RGB")
    width, height = image.size

    # Retrieval view deliberately removes a small amount from the page edges.
    # This reduces the chance that CLIP retrieves projects mainly because of
    # title blocks, logos, addresses, approval stamps, or page furniture.
    left = int(width * crop_left)
    top = int(height * crop_top)
    right = int(width * crop_right)
    bottom = int(height * crop_bottom)

    if right <= left or bottom <= top:
        raise ValueError("Invalid retrieval crop fractions")

    retrieval = image.crop((left, top, right, bottom))
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval.save(retrieval_path, quality=90, optimize=True)

    return width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--crop-left", type=float, default=0.04)
    parser.add_argument("--crop-top", type=float, default=0.04)
    parser.add_argument("--crop-right", type=float, default=0.90)
    parser.add_argument("--crop-bottom", type=float, default=0.90)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rendered_rows: list[dict[str, str | int]] = []

    for row in rows:
        project_id = row["project_id"]
        source_role = row["source_role"]
        page_number = int(row["page_number"])
        page_type = row["page_type"]

        pdf_path = args.raw_root / project_id / f"{source_role}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Missing private source PDF: {pdf_path}. "
                "Copy the audited source into the expected private layout."
            )

        project_out = args.output_root / project_id
        stem = f"{project_id}_{page_type}_{source_role}_p{page_number:02d}"

        full_path = project_out / f"{stem}_full.jpg"
        retrieval_path = project_out / f"{stem}_retrieval.jpg"

        width, height = render_page(
            pdf_path=pdf_path,
            page_number=page_number,
            full_path=full_path,
            retrieval_path=retrieval_path,
            dpi=args.dpi,
            crop_left=args.crop_left,
            crop_top=args.crop_top,
            crop_right=args.crop_right,
            crop_bottom=args.crop_bottom,
        )

        rendered_rows.append(
            {
                "project_id": project_id,
                "source_role": source_role,
                "page_number": page_number,
                "page_type": page_type,
                "full_image": str(full_path.relative_to(args.output_root)),
                "retrieval_image": str(retrieval_path.relative_to(args.output_root)),
                "width_px": width,
                "height_px": height,
                "sha256_full": sha256(full_path),
            }
        )

    output_manifest = args.output_root / "render_manifest.csv"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rendered_rows[0].keys())
        writer.writeheader()
        writer.writerows(rendered_rows)

    print(
        f"Rendered {len(rendered_rows)} selected pages "
        f"({len(rendered_rows) * 2} image files) to {args.output_root}"
    )
    print(f"Render manifest: {output_manifest}")


if __name__ == "__main__":
    main()

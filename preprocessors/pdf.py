"""
PDF preprocessor: text-based PDFs with embedded figures.
Final version -- uses pymupdf4llm for text + table extraction.

APPROACH:
  pymupdf4llm.to_markdown() converts the full PDF to clean Markdown.
  Tables are rendered as proper pipe-delimited Markdown tables.
  Two-column academic layouts are handled correctly.
  The Markdown output is passed directly to the LLM as text_content.
  Phase 1 reads tables from the Markdown without any programmatic parsing.

INSTALL:
  pip install pymupdf4llm

TEXT_CHAR_LIMIT -- set per hardware tier:
  CPU only  (phi3:mini,     4K context):  8000
  Low GPU   (llama3.1:8b,   8K context): 16000
  Mid GPU   (llama3.3:70b, 16K context): 50000  <-- default
  High GPU  (llama3.3:70b, 32K context): 80000
"""
import io
import re
import shutil
from pathlib import Path
from typing import Union

try:
    import pymupdf4llm
    HAS_4LLM = True
except ImportError:
    HAS_4LLM = False

try:
    import fitz  # pymupdf -- still needed for figure extraction
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from preprocessors.imagery import preprocess_imagery

TEXT_CHAR_LIMIT       = 50000   # adjust for your hardware tier (see docstring)
MIN_FIGURE_SIZE_BYTES = 5000
MIN_FIGURE_DIM        = 100
TMP_DIR               = Path('./tmp/pdf_figures')


def preprocess_pdf(
    file_path: Union[str, Path],
    keep_figures: bool = False,
    text_char_limit: int = TEXT_CHAR_LIMIT,
) -> dict:
    """
    Extract text (as clean Markdown with tables) and figures from a PDF.

    Args:
        file_path:       Path to the PDF file
        keep_figures:    Keep extracted figure PNGs in ./tmp/pdf_figures/
        text_char_limit: Max characters of Markdown to pass to the LLM

    Returns:
        Structured summary dict compatible with context_assembly.py
    """
    path    = Path(file_path)
    summary = _base_summary(path)

    if not HAS_4LLM:
        summary['notes'].append(
            'pymupdf4llm not installed. Run: pip install pymupdf4llm'
        )
        return summary

    if not HAS_PYMUPDF:
        summary['notes'].append(
            'pymupdf not installed. Run: pip install pymupdf'
        )
        return summary

    # ── Text + table extraction (pymupdf4llm) ─────────────────────────────────
    try:
        # to_markdown() returns the full document as a Markdown string.
        # Tables are rendered as pipe-delimited Markdown tables.
        # Images are replaced with placeholder text (omitted by default).
        md_text = pymupdf4llm.to_markdown(
            str(path),
            show_progress=False,
        )

        # Count pages via pymupdf for metadata
        doc = fitz.open(str(path))
        summary['text_page_count'] = len(doc)
        summary['shape']           = [len(doc), 1]

        # Count Markdown tables detected
        table_count = len(re.findall(r'^\|', md_text, re.MULTILINE))
        # table_count is an overcount (counts every pipe-starting line);
        # a better heuristic: count separator rows (|---|)
        n_table_rows = len(re.findall(r'^\|', md_text, re.MULTILINE))
        summary['notes'].append(
            f'pymupdf4llm Markdown extraction complete. '
            f'~{n_table_rows} table row(s) detected in output. '
            f'Tables are embedded in text_content as Markdown.'
        )

        full_text = md_text
        truncated = len(full_text) > text_char_limit
        summary['text_content']    = full_text[:text_char_limit]
        summary['text_word_count'] = len(full_text.split())
        summary['has_tables']      = n_table_rows > 0

        if not full_text.strip():
            summary['notes'].append(
                'No text extracted. PDF may be scanned (image-only). '
                'Convert to text-based PDF or provide a .txt transcript.'
            )
        elif truncated:
            pct = round(text_char_limit / len(full_text) * 100)
            summary['notes'].append(
                f'Text truncated: {text_char_limit:,} of {len(full_text):,} chars '
                f'(~{pct}% of document). '
                f'Increase TEXT_CHAR_LIMIT in pdf.py to capture more.'
            )
        else:
            summary['notes'].append(
                f'Full Markdown text: {len(full_text):,} chars, '
                f'{summary["text_page_count"]} pages.'
            )

    except Exception as e:
        summary['notes'].append(f'pymupdf4llm extraction error: {e}')
        doc = fitz.open(str(path))
        summary['text_page_count'] = len(doc)
        summary['shape']           = [len(doc), 1]

    # ── Figure extraction (pymupdf) ───────────────────────────────────────────
    figure_summaries = []
    fig_dir          = TMP_DIR / path.stem
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_count        = 0

    if HAS_PIL:
        try:
            for page_idx in range(len(doc)):
                page       = doc[page_idx]
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        img_bytes  = base_image['image']

                        if len(img_bytes) < MIN_FIGURE_SIZE_BYTES:
                            continue

                        pil_img = PILImage.open(io.BytesIO(img_bytes))
                        w, h    = pil_img.size
                        if w < MIN_FIGURE_DIM or h < MIN_FIGURE_DIM:
                            continue

                        fig_count += 1
                        fig_path   = (
                            fig_dir / f'fig_p{page_idx + 1}_{img_idx:02d}.png'
                        )
                        pil_img.save(str(fig_path), 'PNG')

                        img_summary = preprocess_imagery(fig_path)
                        img_summary['_source']   = (
                            f'PDF page {page_idx + 1}, figure {img_idx + 1}'
                        )
                        img_summary['_from_pdf'] = True
                        figure_summaries.append(img_summary)

                    except Exception as e:
                        summary['notes'].append(
                            f'Figure error (page {page_idx + 1}, '
                            f'image {img_idx}): {e}'
                        )
        except Exception as e:
            summary['notes'].append(f'Figure extraction error: {e}')
    else:
        summary['notes'].append(
            'Pillow not installed -- figure extraction skipped. '
            'Run: pip install Pillow'
        )

    doc.close()

    summary['figure_count']    = fig_count
    summary['figure_summaries'] = figure_summaries
    summary['table_summaries'] = []  # tables are in text_content as Markdown

    if fig_count > 0:
        summary['notes'].append(f'{fig_count} figure(s) extracted.')

    if not keep_figures:
        shutil.rmtree(fig_dir, ignore_errors=True)

    return summary


def _base_summary(path: Path) -> dict:
    return {
        'filename':            path.name,
        'file_type':           '.pdf',
        'data_type':           'pdf',
        'shape':               [0, 1],
        'text_content':        '',
        'text_word_count':     0,
        'text_page_count':     0,
        'table_summaries':     [],
        'figure_summaries':    [],
        'figure_count':        0,
        'has_tables':          False,
        'columns':             [],
        'numeric_summary':     {},
        'categorical_summary': {},
        'sample_rows':         [],
        'missing_values':      {},
        'notes':               [],
    }

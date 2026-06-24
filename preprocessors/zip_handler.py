"""
ZIP file handler for multi-file uploads (shapefiles, etc.)
preprocessors/zip_handler.py

Extracts ZIP archives to a persistent temp directory and returns paths
to all recognised data files found inside.

Primary use case: shapefile upload in the Gradio GUI.
A shapefile bundle (.shp + .dbf + .shx + .prj) must all exist in the
same directory for geopandas to read it. Gradio uploads each file to a
separate temp path, so users must ZIP the bundle first.

Also handles:
  - ZIPs containing multiple CSVs, GeoJSONs, or images
  - Nested ZIPs (one level deep)
  - ZIPs with subdirectories (files are flattened to a single temp dir)

Extracted files are NOT cleaned up automatically because geopandas needs
the sibling files on disk while reading. Call cleanup_zip_temp() when done.
"""
import zipfile
import shutil
from pathlib import Path
from typing import List, Tuple

ZIP_TEMP_BASE = Path('./tmp/zip_extracted')

SUPPORTED_TABULAR    = {'.csv', '.xlsx', '.xls', '.txt'}
SUPPORTED_GEOSPATIAL = {'.shp', '.geojson', '.json'}
SUPPORTED_IMAGERY    = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
SUPPORTED_PDF        = {'.pdf'}
SUPPORTED_ALL        = (
    SUPPORTED_TABULAR | SUPPORTED_GEOSPATIAL |
    SUPPORTED_IMAGERY | SUPPORTED_PDF
)


def extract_zip(zip_path: str) -> Tuple[List[Path], Path]:
    """
    Extract a ZIP archive and return paths to all supported data files inside.

    Files are extracted to ./tmp/zip_extracted/<zip_stem>/.
    All files (including from subdirectories) are flattened into that folder
    so shapefile siblings (.shp, .dbf, .shx, .prj) land in the same directory.

    Args:
        zip_path: Path to the uploaded .zip file

    Returns:
        (file_paths, extract_dir)
        file_paths:  List of Path objects for supported data files found
        extract_dir: The directory they were extracted to (for cleanup)
    """
    zip_path   = Path(zip_path)
    extract_dir = ZIP_TEMP_BASE / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            # Skip directories and hidden macOS metadata files
            if member.is_dir():
                continue
            fname = Path(member.filename).name
            if fname.startswith('.') or fname.startswith('__'):
                continue
            # Flatten to extract_dir (ignore any subdirectory structure)
            dest = extract_dir / fname
            with zf.open(member) as src, open(dest, 'wb') as dst:
                shutil.copyfileobj(src, dst)

    # Collect all supported files found
    found = []
    for f in sorted(extract_dir.iterdir()):
        if f.suffix.lower() in SUPPORTED_ALL:
            found.append(f)

    return found, extract_dir


def cleanup_zip_temp(extract_dir: Path) -> None:
    """Remove an extracted ZIP temp directory when no longer needed."""
    shutil.rmtree(extract_dir, ignore_errors=True)
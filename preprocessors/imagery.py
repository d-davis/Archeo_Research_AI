"""
Imagery preprocessor: TIF/TIFF (raster), PNG, JPEG.

For raster (TIF):
  - Uses rasterio: band info, resolution, CRS, nodata, per-band stats
  - Stats computed from a 512x512 downsampled read (efficient on large rasters)

For standard images (PNG, JPEG):
  - Uses Pillow: dimensions, color mode, per-channel statistics

Also exposes get_image_thumbnail_b64() which generates a base64-encoded
JPEG thumbnail used by the vision model step in main.py.

No full pixel arrays are passed to the LLM -- only derived statistics.
"""
import base64
import io
from pathlib import Path
from typing import Union, Optional

try:
    import rasterio
    from rasterio.enums import Resampling
    import numpy as np
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image, ImageStat
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

THUMBNAIL_MAX = (512, 512)


def preprocess_imagery(file_path: Union[str, Path]) -> dict:
    """Load and summarize an image file. Returns structured summary dict."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in ('.tif', '.tiff'):
        return _summarize_raster(path)
    elif ext in ('.png', '.jpg', '.jpeg'):
        return _summarize_standard_image(path)
    else:
        raise ValueError(f'Unsupported imagery format for Phase 2: {ext}')


def get_image_thumbnail_b64(file_path: Union[str, Path]) -> Optional[str]:
    """
    Return a base64-encoded JPEG thumbnail for the vision model.
    Returns None if imagery libraries are unavailable.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    try:
        if ext in ('.tif', '.tiff') and HAS_RASTERIO and HAS_PIL:
            return _raster_to_b64(path)
        elif HAS_PIL:
            return _standard_to_b64(path)
    except Exception:
        return None
    return None


# ── Raster (TIF) ──────────────────────────────────────────────────────────────

def _summarize_raster(path: Path) -> dict:
    base = _base_summary(path, 'raster_imagery')

    if not HAS_RASTERIO:
        base['notes'].append('rasterio not installed. Run: conda install -c conda-forge rasterio')
        return base

    with rasterio.open(path) as src:
        base['width']      = src.width
        base['height']     = src.height
        base['shape']      = [src.height, src.width]
        base['band_count'] = src.count
        base['driver']     = src.driver
        base['nodata']     = src.nodata

        # CRS
        if src.crs:
            base['crs'] = {
                'epsg':      src.crs.to_epsg(),
                'wkt_short': src.crs.to_string()[:100],
            }
        else:
            base['notes'].append('No CRS defined on raster.')

        # Pixel resolution
        if src.transform:
            base['pixel_resolution'] = {
                'x': round(abs(float(src.transform.a)), 8),
                'y': round(abs(float(src.transform.e)), 8),
            }

        # Geographic extent (top-left and bottom-right corners)
        try:
            bounds = src.bounds
            base['geographic_extent'] = {
                'left':   round(bounds.left,   6),
                'bottom': round(bounds.bottom, 6),
                'right':  round(bounds.right,  6),
                'top':    round(bounds.top,    6),
            }
        except Exception:
            pass

        # Per-band statistics from downsampled read (max 4 bands)
        band_stats = {}
        for band_idx in range(1, min(src.count + 1, 5)):
            try:
                out_h = min(512, src.height)
                out_w = min(512, src.width)
                data = src.read(
                    band_idx,
                    out_shape=(1, out_h, out_w),
                    resampling=Resampling.average,
                    masked=True
                )
                flat = data[0].compressed()
                if len(flat) > 0:
                    band_stats[f'band_{band_idx}'] = {
                        'dtype': str(src.dtypes[band_idx - 1]),
                        'min':   round(float(flat.min()),  4),
                        'max':   round(float(flat.max()),  4),
                        'mean':  round(float(flat.mean()), 4),
                        'std':   round(float(flat.std()),  4),
                        'nodata_pixels': int((data[0].mask).sum()) if hasattr(data[0], 'mask') else 0,
                    }
            except Exception as e:
                base['notes'].append(f'Band {band_idx} read error: {e}')
        base['band_stats'] = band_stats
        base['numeric_summary'] = band_stats  # context assembler compatibility

    size_mb = path.stat().st_size / (1024 * 1024)
    base['notes'].append(f'File size: {size_mb:.1f} MB')
    # Generate thumbnail for vision model
    try:
        base['_thumbnail_b64'] = _raster_to_b64(path)
    except Exception:
        base['_thumbnail_b64'] = None
    return base


def _raster_to_b64(path: Path) -> str:
    """Render first 3 bands to a normalized RGB thumbnail, return as base64 JPEG."""
    with rasterio.open(path) as src:
        n = min(src.count, 3)
        out_h = min(1024, src.height)
        out_w = min(1024, src.width)
        data = src.read(
            list(range(1, n + 1)),
            out_shape=(n, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        bands = []
        for i in range(n):
            b = data[i].astype('float32')
            lo, hi = b.min(), b.max()
            if hi > lo:
                b = ((b - lo) / (hi - lo) * 255).astype('uint8')
            else:
                b = np.zeros_like(b, dtype='uint8')
            bands.append(b)
        arr = np.stack(bands if n == 3 else bands * 3, axis=-1)
        img = Image.fromarray(arr, mode='RGB')
    return _img_to_b64(img)


# ── Standard images (PNG, JPEG) ───────────────────────────────────────────────

def _summarize_standard_image(path: Path) -> dict:
    base = _base_summary(path, 'standard_imagery')

    if not HAS_PIL:
        base['notes'].append('Pillow not installed. Run: pip install Pillow')
        return base

    img = Image.open(path)
    base['width']  = img.width
    base['height'] = img.height
    base['shape']  = [img.height, img.width]
    base['mode']   = img.mode

    # Per-channel statistics
    try:
        stat = ImageStat.Stat(img)
        mode_channels = {'RGB': ['R','G','B'], 'RGBA': ['R','G','B','A'],
                         'L': ['L'], 'P': ['P']}
        channels = mode_channels.get(img.mode, ['C0','C1','C2'])[:len(stat.mean)]
        ch_stats = {}
        for i, ch in enumerate(channels):
            ch_stats[ch] = {
                'mean': round(stat.mean[i],   2),
                'std':  round(stat.stddev[i], 2),
                'min':  round(stat.extrema[i][0], 2),
                'max':  round(stat.extrema[i][1], 2),
            }
        base['channel_stats']  = ch_stats
        base['numeric_summary'] = ch_stats  # context assembler compatibility
    except Exception:
        pass

    size_mb = path.stat().st_size / (1024 * 1024)
    base['notes'].append(f'File size: {size_mb:.2f} MB')
    # Generate thumbnail for vision model
    try:
        base['_thumbnail_b64'] = _standard_to_b64(path)
    except Exception:
        base['_thumbnail_b64'] = None
    return base


def _standard_to_b64(path: Path) -> str:
    img = Image.open(path).convert('RGB')
    img.thumbnail(THUMBNAIL_MAX, Image.LANCZOS)
    return _img_to_b64(img)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _img_to_b64(img: 'Image.Image') -> str:
    img.thumbnail(THUMBNAIL_MAX, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _base_summary(path: Path, data_type: str) -> dict:
    """Return a minimal summary dict with all required keys."""
    return {
        'filename':            path.name,
        'file_type':           path.suffix.lower(),
        'data_type':           data_type,
        'shape':               [0, 0],
        'width':               None,
        'height':              None,
        'band_count':          None,
        'crs':                 None,
        'pixel_resolution':    None,
        'band_stats':          {},
        'channel_stats':       {},
        'vision_description':  None,  # populated by main.py after vision model call
        '_thumbnail_b64': None,
        'columns':             [],
        'numeric_summary':     {},
        'categorical_summary': {},
        'sample_rows':         [],
        'missing_values':      {},
        'notes':               [],
    }

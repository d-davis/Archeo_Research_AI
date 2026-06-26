"""
Chart generation from preprocessed file summaries.
Phase 4 addition.

Generates PNG charts from the statistical summaries produced by the
preprocessing layer. Charts are saved to ./output/charts/<session_id>/
and displayed inline in the Gradio UI.

Chart types generated per file type:
  tabular:        numeric distribution plots (box plots), categorical
                  frequency bar charts, missing value heatmap (if any)
  geospatial:     bounding box map, top categorical attribute bar charts,
                  numeric range plots
  raster_imagery: per-band histogram
  standard_imagery: per-channel distribution
  pdf:            word count bar (placeholder; tables shown as text)

All charts use a consistent palette derived from the system accent color.
"""
import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CHART_DIR  = Path('./output/charts')
ACCENT     = '#3d9970'
GRID_COLOR = '#2a2a3a'
BG_COLOR   = '#1a1a2e'
TEXT_COLOR = '#e8e8f0'
MUTED      = '#5a5a7a'


def _apply_style(ax, title: str):
    """Apply consistent dark styling to a matplotlib axis."""
    ax.set_facecolor(BG_COLOR)
    ax.figure.patch.set_facecolor(BG_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def generate_charts(
    preprocessed_files: List[dict],
    session_id: str,
    max_charts_per_file: int = 4,
) -> List[Path]:
    """
    Generate PNG charts from preprocessed summaries.

    Args:
        preprocessed_files:    List of preprocessed summary dicts
        session_id:            Used to create a unique output subfolder
        max_charts_per_file:   Cap on charts per file to avoid overload

    Returns:
        List of paths to generated PNG files.
    """
    if not HAS_MPL:
        return []

    out_dir = CHART_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    for file_summary in preprocessed_files:
        fname     = file_summary.get('filename', 'unknown')
        data_type = file_summary.get('data_type', '')
        safe_name = fname.replace('.', '_').replace(' ', '_')

        if data_type == 'tabular':
            paths = _charts_tabular(file_summary, out_dir, safe_name, max_charts_per_file)
            chart_paths.extend(paths)

        elif data_type == 'geospatial':
            paths = _charts_geospatial(file_summary, out_dir, safe_name, max_charts_per_file)
            chart_paths.extend(paths)

        elif data_type == 'raster_imagery':
            paths = _charts_raster(file_summary, out_dir, safe_name)
            chart_paths.extend(paths)

        elif data_type == 'standard_imagery':
            paths = _charts_standard_image(file_summary, out_dir, safe_name)
            chart_paths.extend(paths)

        elif data_type == 'pdf':
            paths = _charts_pdf(file_summary, out_dir, safe_name)
            chart_paths.extend(paths)

    return chart_paths


# ── Tabular charts ─────────────────────────────────────────────────────────────

def _charts_tabular(summary: dict, out_dir: Path, safe_name: str, max_n: int) -> List[Path]:
    paths    = []
    num_sum  = summary.get('numeric_summary', {})
    cat_sum  = summary.get('categorical_summary', {})
    fname    = summary.get('filename', '')

    # Numeric box-style summary plot (min/mean/max per column)
    if num_sum and len(paths) < max_n:
        cols = list(num_sum.keys())[:12]
        means = [num_sum[c].get('mean', 0) for c in cols]
        mins  = [num_sum[c].get('min',  0) for c in cols]
        maxs  = [num_sum[c].get('max',  0) for c in cols]

        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.8), 4))
        x = range(len(cols))
        ax.bar(x, maxs, color=ACCENT, alpha=0.3, label='Max')
        ax.bar(x, means, color=ACCENT, alpha=0.85, label='Mean')
        ax.bar(x, mins, color=ACCENT, alpha=0.5, label='Min')
        ax.set_xticks(list(x))
        ax.set_xticklabels([c[:14] for c in cols], rotation=35, ha='right', fontsize=7)
        ax.legend(fontsize=7, facecolor=BG_COLOR, labelcolor=TEXT_COLOR)
        _apply_style(ax, f'{fname}: Numeric Column Ranges')
        plt.tight_layout()
        p = out_dir / f'{safe_name}_numeric.png'
        fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)
        paths.append(p)

    # Categorical frequency charts (one per column, up to max_n - 1)
    for col, vc in list(cat_sum.items())[:max_n - len(paths)]:
        items = sorted(vc.items(), key=lambda x: x[1], reverse=True)[:10]
        labels, values = zip(*items) if items else ([], [])
        fig, ax = plt.subplots(figsize=(6, max(2.5, len(labels) * 0.35)))
        colors = [ACCENT if i == 0 else '#2d7a55' for i in range(len(labels))]
        ax.barh([str(l)[:28] for l in labels], values, color=colors)
        _apply_style(ax, f'{fname}: {col} (top {len(labels)})')
        plt.tight_layout()
        p = out_dir / f'{safe_name}_cat_{col[:20]}.png'
        fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)
        paths.append(p)

    return paths


# ── Geospatial charts ──────────────────────────────────────────────────────────

def _charts_geospatial(summary: dict, out_dir: Path, safe_name: str, max_n: int) -> List[Path]:
    paths  = []
    fname  = summary.get('filename', '')
    bbox   = summary.get('bounding_box')
    cat_sum = summary.get('categorical_summary', {})
    num_sum = summary.get('numeric_summary', {})

    # Bounding box map
    if bbox and len(paths) < max_n:
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            minx, miny = bbox.get('minx', 0), bbox.get('miny', 0)
            maxx, maxy = bbox.get('maxx', 1), bbox.get('maxy', 1)
            rect = mpatches.FancyBboxPatch(
                (minx, miny), maxx - minx, maxy - miny,
                boxstyle='round,pad=0', linewidth=1.5,
                edgecolor=ACCENT, facecolor='#1d3d2a', alpha=0.8
            )
            ax.add_patch(rect)
            ax.set_xlim(minx - abs(maxx - minx) * 0.15, maxx + abs(maxx - minx) * 0.15)
            ax.set_ylim(miny - abs(maxy - miny) * 0.15, maxy + abs(maxy - miny) * 0.15)
            n_feat = summary.get('feature_count', '?')
            crs_name = summary.get('crs', {}).get('name', 'Unknown CRS') if summary.get('crs') else 'No CRS'
            ax.set_xlabel('X / Longitude', fontsize=8)
            ax.set_ylabel('Y / Latitude', fontsize=8)
            ax.text((minx + maxx) / 2, (miny + maxy) / 2,
                    f'{n_feat} features', ha='center', va='center',
                    color=ACCENT, fontsize=9, fontweight='bold')
            _apply_style(ax, f'{fname}: Spatial Extent\n{crs_name}')
            plt.tight_layout()
            p = out_dir / f'{safe_name}_bbox.png'
            fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
            plt.close(fig)
            paths.append(p)
        except Exception:
            pass

    # Top categorical attributes
    for col, vc in list(cat_sum.items())[:max_n - len(paths)]:
        items = sorted(vc.items(), key=lambda x: x[1], reverse=True)[:10]
        if not items:
            continue
        labels, values = zip(*items)
        fig, ax = plt.subplots(figsize=(6, max(2.5, len(labels) * 0.35)))
        ax.barh([str(l)[:28] for l in labels], values, color=ACCENT)
        _apply_style(ax, f'{fname}: {col}')
        plt.tight_layout()
        p = out_dir / f'{safe_name}_attr_{col[:20]}.png'
        fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)
        paths.append(p)

    return paths


# ── Raster imagery charts ──────────────────────────────────────────────────────

def _charts_raster(summary: dict, out_dir: Path, safe_name: str) -> List[Path]:
    band_stats = summary.get('band_stats', {})
    fname      = summary.get('filename', '')
    if not band_stats:
        return []

    bands  = list(band_stats.keys())
    means  = [band_stats[b].get('mean', 0) for b in bands]
    stds   = [band_stats[b].get('std',  0) for b in bands]
    mins   = [band_stats[b].get('min',  0) for b in bands]
    maxs   = [band_stats[b].get('max',  0) for b in bands]

    fig, ax = plt.subplots(figsize=(max(4, len(bands) * 1.2), 4))
    x = range(len(bands))
    ax.bar(x, means, yerr=stds, color=ACCENT, alpha=0.85,
           error_kw={'ecolor': TEXT_COLOR, 'capsize': 4})
    ax.set_xticks(list(x))
    ax.set_xticklabels([b.replace('_', ' ') for b in bands], fontsize=8)
    ax.set_ylabel('Mean value', fontsize=8)
    _apply_style(ax, f'{fname}: Per-band Statistics (mean +/- std)')
    plt.tight_layout()
    p = out_dir / f'{safe_name}_bands.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return [p]


# ── Standard image charts ──────────────────────────────────────────────────────

def _charts_standard_image(summary: dict, out_dir: Path, safe_name: str) -> List[Path]:
    ch_stats = summary.get('channel_stats', {})
    fname    = summary.get('filename', '')
    if not ch_stats:
        return []

    channels = list(ch_stats.keys())
    means    = [ch_stats[c].get('mean', 0) for c in channels]
    stds     = [ch_stats[c].get('std',  0) for c in channels]
    ch_colors = {'R': '#e05050', 'G': '#50c050', 'B': '#5080e0', 'L': ACCENT}
    colors   = [ch_colors.get(c, ACCENT) for c in channels]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(channels, means, yerr=stds, color=colors, alpha=0.85,
           error_kw={'ecolor': TEXT_COLOR, 'capsize': 4})
    ax.set_ylabel('Mean pixel value (0-255)', fontsize=8)
    _apply_style(ax, f'{fname}: Channel Statistics')
    plt.tight_layout()
    p = out_dir / f'{safe_name}_channels.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return [p]


# ── PDF charts ─────────────────────────────────────────────────────────────────

def _charts_pdf(summary: dict, out_dir: Path, safe_name: str) -> List[Path]:
    """Simple word count and page count bar for PDF summaries."""
    fname      = summary.get('filename', '')
    word_count = summary.get('text_word_count', 0)
    page_count = summary.get('text_page_count', 0)
    if not word_count:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
    axes[0].bar(['Words'], [word_count], color=ACCENT)
    _apply_style(axes[0], 'Word Count')
    axes[1].bar(['Pages'], [page_count], color='#2d7a55')
    _apply_style(axes[1], 'Page Count')
    fig.suptitle(fname, color=TEXT_COLOR, fontsize=9, y=1.02)
    plt.tight_layout()
    p = out_dir / f'{safe_name}_stats.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return [p]

# ═══════════════════════════════════════════════════════════════════════════════
# Follow-up chart generation (Option B: structured chart requests)
# ═══════════════════════════════════════════════════════════════════════════════
import re
import pandas as pd
# Add at top of file if not already present:
#   import re
#   import pandas as pd
def _load_dataframe(filename: str, session: dict):
    """
    Reload a DataFrame from the original file path stored in the session.
    Returns (df_or_gdf, kind) where kind is 'tabular' or 'geospatial'.
    Returns (None, None) if the file cannot be loaded.
    """
    file_paths = session.get('file_paths', {})
    path_str = file_paths.get(filename)
    if not path_str:
        return None, None
    path = Path(path_str)
    if not path.exists():
        return None, None
    ext = path.suffix.lower()
    try:
        if ext in ('.csv', '.txt'):
            return pd.read_csv(path, low_memory=False), 'tabular'
        elif ext in ('.xlsx', '.xls'):
            return pd.read_excel(path), 'tabular'
        elif ext in ('.geojson', '.json', '.shp'):
            import geopandas as gpd
            return gpd.read_file(path), 'geospatial'
    except Exception:
        return None, None
    return None, None
def generate_followup_charts(
    chart_requests: list,
    session: dict,
    session_id: str,
) -> list:
    """
    Generate charts from structured CHART_REQUEST dicts.
    Args:
        chart_requests: List of dicts parsed from LLM CHART_REQUEST lines
        session:        Current session object (needs 'file_paths')
        session_id:     Used for output subfolder naming
    Returns:
        List of Path objects for generated chart PNGs.
    """
    if not HAS_MPL:
        return []
    out_dir = CHART_DIR / f'{session_id}_followup'
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []
    for i, req in enumerate(chart_requests):
        chart_type = req.get('type', '').lower()
        filename   = req.get('file', '')
        title      = req.get('title', f'Chart {i+1}')
        safe       = re.sub(r'[^a-zA-Z0-9_]', '_', f'{filename}_{chart_type}_{i}')
        df, kind = _load_dataframe(filename, session)
        if df is None:
            continue
        try:
            if chart_type == 'histogram':
                col = req.get('column', '')
                p = _fu_histogram(df, col, title, out_dir, safe)
            elif chart_type == 'boxplot':
                col = req.get('column', '')
                p = _fu_boxplot(df, col, title, out_dir, safe)
            elif chart_type == 'bar':
                col = req.get('column', '')
                p = _fu_bar(df, col, title, out_dir, safe)
            elif chart_type == 'timeseries':
                p = _fu_timeseries(df, req.get('x',''), req.get('y',''), title, out_dir, safe)
            elif chart_type == 'scatter':
                p = _fu_scatter(df, req.get('x',''), req.get('y',''), title, out_dir, safe)
            elif chart_type == 'correlation_heatmap':
                p = _fu_correlation_heatmap(df, title, out_dir, safe)
            elif chart_type == 'spatial':
                p = _fu_spatial(df, req.get('lat',''), req.get('lon',''), title, out_dir, safe)
            else:
                continue
            if p:
                chart_paths.append(p)
        except Exception:
            continue  # skip failed charts, don't crash the response
    return chart_paths
# ── Individual follow-up chart functions ──────────────────────────────────────
def _fu_histogram(df, column: str, title: str, out_dir: Path, safe: str):
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors='coerce').dropna()
    if len(s) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(s, bins=min(30, max(5, len(s)//5)), color=ACCENT, alpha=0.85, edgecolor=BG_COLOR)
    ax.set_xlabel(column, fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_boxplot(df, column: str, title: str, out_dir: Path, safe: str):
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors='coerce').dropna()
    if len(s) < 4:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    bp = ax.boxplot(s, patch_artist=True, medianprops={'color': TEXT_COLOR, 'linewidth': 2})
    bp['boxes'][0].set_facecolor(ACCENT)
    bp['boxes'][0].set_alpha(0.75)
    ax.set_ylabel(column, fontsize=8)
    ax.set_xticklabels([column[:20]])
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_bar(df, column: str, title: str, out_dir: Path, safe: str):
    if column not in df.columns:
        return None
    vc = df[column].value_counts().head(15)
    if vc.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, max(3, len(vc) * 0.32)))
    ax.barh([str(k)[:30] for k in vc.index], vc.values, color=ACCENT, alpha=0.85)
    ax.set_xlabel('Count', fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_timeseries(df, x_col: str, y_col: str, title: str, out_dir: Path, safe: str):
    if x_col not in df.columns or y_col not in df.columns:
        return None
    sub = df[[x_col, y_col]].copy()
    sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
    sub = sub.dropna(subset=[y_col]).sort_values(x_col)
    if len(sub) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(len(sub)), sub[y_col].values, color=ACCENT, linewidth=1.5, marker='o', markersize=3)
    step = max(1, len(sub) // 8)
    ax.set_xticks(range(0, len(sub), step))
    ax.set_xticklabels([str(v)[:12] for v in sub[x_col].values[::step]], rotation=30, ha='right', fontsize=7)
    ax.set_ylabel(y_col, fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_scatter(df, x_col: str, y_col: str, title: str, out_dir: Path, safe: str):
    if x_col not in df.columns or y_col not in df.columns:
        return None
    sub = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(sub) < 3:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(sub[x_col], sub[y_col], color=ACCENT, alpha=0.6, s=18, edgecolors='none')
    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_correlation_heatmap(df, title: str, out_dir: Path, safe: str):
    num_df = df.select_dtypes(include='number').dropna(axis=1, how='all')
    cols = num_df.columns.tolist()[:12]
    if len(cols) < 2:
        return None
    corr = num_df[cols].corr()
    fig, ax = plt.subplots(figsize=(max(4, len(cols) * 0.6), max(3.5, len(cols) * 0.55)))
    import numpy as np
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c[:12] for c in cols], rotation=40, ha='right', fontsize=7)
    ax.set_yticklabels([c[:12] for c in cols], fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if abs(corr.values[i,j]) > 0.5 else TEXT_COLOR)
    plt.colorbar(im, ax=ax, shrink=0.8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_spatial(df, lat_col: str, lon_col: str, title: str, out_dir: Path, safe: str):
    # Works with both GeoDataFrames and regular DataFrames with lat/lon columns
    try:
        import geopandas as gpd
        if hasattr(df, 'geometry'):
            # GeoDataFrame: extract centroid coords
            pts = df[df.geometry.geom_type == 'Point']
            if pts.empty:
                pts = df.copy()
                pts['_lon'] = df.geometry.centroid.x
                pts['_lat'] = df.geometry.centroid.y
                lon_col, lat_col = '_lon', '_lat'
            else:
                pts = pts.copy()
                pts['_lon'] = pts.geometry.x
                pts['_lat'] = pts.geometry.y
                lon_col, lat_col = '_lon', '_lat'
            df = pts
    except ImportError:
        pass
    if lat_col not in df.columns or lon_col not in df.columns:
        return None
    sub = df[[lon_col, lat_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(sub) < 1:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(sub[lon_col], sub[lat_col], color=ACCENT, alpha=0.65, s=20, edgecolors='none')
    ax.set_xlabel('Longitude / X', fontsize=8)
    ax.set_ylabel('Latitude / Y', fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p

def _load_dataframe(filename: str, session: dict):
    """
    Reload a DataFrame from the original file path stored in the session.
    Returns (df, kind) where kind is 'tabular' or 'geospatial'.
    Returns (None, None) if the file cannot be loaded.
    """
    file_paths = session.get('file_paths', {})
    path_str = file_paths.get(filename)
    if not path_str:
        return None, None
    path = Path(path_str)
    if not path.exists():
        return None, None
    ext = path.suffix.lower()
    try:
        if ext in ('.csv', '.txt'):
            return pd.read_csv(path, low_memory=False), 'tabular'
        elif ext in ('.xlsx', '.xls'):
            return pd.read_excel(path), 'tabular'
        elif ext in ('.geojson', '.json', '.shp'):
            import geopandas as gpd
            return gpd.read_file(path), 'geospatial'
    except Exception:
        return None, None
    return None, None
def generate_followup_charts(
    chart_requests: list,
    session: dict,
    session_id: str,
) -> list:
    """
    Generate charts from structured CHART_REQUEST dicts produced by followup.py.
    Returns list of Path objects for generated PNGs.
    """
    if not HAS_MPL:
        return []
    out_dir = CHART_DIR / f'{session_id}_followup'
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []
    for i, req in enumerate(chart_requests):
        chart_type = req.get('type', '').lower()
        filename   = req.get('file', '')
        title      = req.get('title', f'Chart {i + 1}')
        safe       = re.sub(r'[^a-zA-Z0-9_]', '_', f'{filename}_{chart_type}_{i}')
        # Inline data charts don't need file loading
        if chart_type == '_grouped_bar_inline':
            try:
                p = _fu_grouped_bar_inline(req, out_dir, safe)
                if p:
                    chart_paths.append(p)
            except Exception:
                pass
            continue
        df, kind = _load_dataframe(filename, session)
        if df is None:
            continue
        try:
            p = None
            if chart_type == 'histogram':
                p = _fu_histogram(df, req.get('column', ''), title, out_dir, safe)
            elif chart_type == 'boxplot':
                p = _fu_boxplot(df, req.get('column', ''), title, out_dir, safe)
            elif chart_type == 'bar':
                p = _fu_bar(df, req.get('column', ''), title, out_dir, safe)
            elif chart_type == 'timeseries':
                p = _fu_timeseries(df, req.get('x', ''), req.get('y', ''), title, out_dir, safe)
            elif chart_type == 'scatter':
                p = _fu_scatter(df, req.get('x', ''), req.get('y', ''), title, out_dir, safe)
            elif chart_type == 'correlation_heatmap':
                p = _fu_correlation_heatmap(df, title, out_dir, safe)
            elif chart_type == 'spatial':
                p = _fu_spatial(df, req.get('lat', ''), req.get('lon', ''), title, out_dir, safe)
            if p:
                chart_paths.append(p)
        except Exception:
            continue
    return chart_paths
# ── Individual follow-up chart functions ──────────────────────────────────────
def _fu_histogram(df, column, title, out_dir, safe):
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors='coerce').dropna()
    if len(s) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(s, bins=min(30, max(5, len(s) // 5)), color=ACCENT, alpha=0.85, edgecolor=BG_COLOR)
    ax.set_xlabel(column, fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_boxplot(df, column, title, out_dir, safe):
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors='coerce').dropna()
    if len(s) < 4:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    bp = ax.boxplot(s, patch_artist=True, medianprops={'color': TEXT_COLOR, 'linewidth': 2})
    bp['boxes'][0].set_facecolor(ACCENT)
    bp['boxes'][0].set_alpha(0.75)
    ax.set_ylabel(column, fontsize=8)
    ax.set_xticklabels([column[:20]])
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_bar(df, column, title, out_dir, safe):
    if column not in df.columns:
        return None
    vc = df[column].value_counts().head(15)
    if vc.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, max(3, len(vc) * 0.32)))
    ax.barh([str(k)[:30] for k in vc.index], vc.values, color=ACCENT, alpha=0.85)
    ax.set_xlabel('Count', fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_timeseries(df, x_col, y_col, title, out_dir, safe):
    if x_col not in df.columns or y_col not in df.columns:
        return None
    sub = df[[x_col, y_col]].copy()
    sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
    sub = sub.dropna(subset=[y_col]).sort_values(x_col)
    if len(sub) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(len(sub)), sub[y_col].values, color=ACCENT,
            linewidth=1.5, marker='o', markersize=3)
    step = max(1, len(sub) // 8)
    ax.set_xticks(range(0, len(sub), step))
    ax.set_xticklabels(
        [str(v)[:12] for v in sub[x_col].values[::step]],
        rotation=30, ha='right', fontsize=7,
    )
    ax.set_ylabel(y_col, fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_scatter(df, x_col, y_col, title, out_dir, safe):
    if x_col not in df.columns or y_col not in df.columns:
        return None
    sub = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(sub) < 3:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(sub[x_col], sub[y_col], color=ACCENT, alpha=0.6, s=18, edgecolors='none')
    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_correlation_heatmap(df, title, out_dir, safe):
    import numpy as np
    num_df = df.select_dtypes(include='number').dropna(axis=1, how='all')
    cols = num_df.columns.tolist()[:12]
    if len(cols) < 2:
        return None
    corr = num_df[cols].corr()
    fig, ax = plt.subplots(figsize=(max(4, len(cols) * 0.65), max(3.5, len(cols) * 0.6)))
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c[:12] for c in cols], rotation=40, ha='right', fontsize=7)
    ax.set_yticklabels([c[:12] for c in cols], fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center',
                    fontsize=6,
                    color='white' if abs(corr.values[i, j]) > 0.5 else TEXT_COLOR)
    plt.colorbar(im, ax=ax, shrink=0.8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_spatial(df, lat_col, lon_col, title, out_dir, safe):
    try:
        if hasattr(df, 'geometry'):
            pts = df.copy()
            pts['_lon'] = df.geometry.centroid.x
            pts['_lat'] = df.geometry.centroid.y
            lon_col, lat_col = '_lon', '_lat'
            df = pts
    except Exception:
        pass
    if lat_col not in df.columns or lon_col not in df.columns:
        return None
    sub = df[[lon_col, lat_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(sub[lon_col], sub[lat_col], color=ACCENT, alpha=0.65, s=20, edgecolors='none')
    ax.set_xlabel('Longitude / X', fontsize=8)
    ax.set_ylabel('Latitude / Y', fontsize=8)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p
def _fu_grouped_bar_inline(req: dict, out_dir: Path, safe: str):
    """Render a grouped bar chart from inline parsed data (no file needed)."""
    import numpy as np
    categories = req.get('categories', [])
    series     = req.get('series', {})
    title      = req.get('title', 'Chart')
    x_label    = req.get('x_label', '')
    y_label    = req.get('y_label', '')
    if not categories or not series:
        return None
    n_cats   = len(categories)
    n_series = len(series)
    x        = np.arange(n_cats)
    width    = 0.8 / n_series
    colors   = [ACCENT, '#2d7a55', '#e05050', '#5080e0', '#e0a030']
    fig, ax  = plt.subplots(figsize=(max(5, n_cats * 1.2), 4))
    for i, (sname, vals) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * width
        ax.bar(x + offset, vals[:n_cats], width,
               label=sname, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c[:18] for c in categories], fontsize=8)
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.legend(fontsize=7, facecolor=BG_COLOR, labelcolor=TEXT_COLOR)
    _apply_style(ax, title)
    plt.tight_layout()
    p = out_dir / f'{safe}.png'
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    return p

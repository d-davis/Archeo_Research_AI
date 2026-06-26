"""
Geospatial data preprocessor: SHP, GeoJSON.

Extracts a structured summary dict for LLM context:
  - CRS and projection info
  - Geometry types and feature count
  - Attribute schema with statistics
  - Bounding box and spatial extent estimate
  - Sample features (first 3, attributes only)
  - Archaeological column heuristics

No raw geometry coordinates are passed to the LLM.
"""
from pathlib import Path
from typing import Union

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

from analytics import run_geospatial_analytics

def preprocess_geospatial(file_path: Union[str, Path]) -> dict:
    """Load and summarize a geospatial file. Returns structured summary dict."""
    path = Path(file_path)

    if not HAS_GEOPANDAS:
        return _error_summary(path, 'geopandas not installed. Run: conda install -c conda-forge geopandas')

    gdf = gpd.read_file(path)
    return _summarize_geodataframe(gdf, path)


def _summarize_geodataframe(gdf, path: Path) -> dict:
    summary = {
        'filename':            path.name,
        'file_type':           path.suffix.lower(),
        'data_type':           'geospatial',
        'shape':               [len(gdf), len(gdf.columns)],
        'feature_count':       len(gdf),
        'crs':                 None,
        'geometry_types':      [],
        'bounding_box':        None,
        'attribute_schema':    [],
        'numeric_summary':     {},
        'categorical_summary': {},
        'sample_features':     [],
        'missing_values':      {},
        'notes':               [],
        # Required by context assembler for compatibility
        'columns':             [],
        'sample_rows':         [],
    }

    # CRS information
    if gdf.crs is not None:
        epsg = gdf.crs.to_epsg()
        summary['crs'] = {
            'epsg':          epsg,
            'name':          gdf.crs.name,
            'is_geographic': gdf.crs.is_geographic,
            'is_projected':  gdf.crs.is_projected,
        }
        if gdf.crs.axis_info:
            summary['crs']['units'] = str(gdf.crs.axis_info[0].unit_name)
    else:
        summary['notes'].append('No CRS defined -- coordinate reference system unknown.')

    # Geometry types
    geom_types = gdf.geometry.geom_type.value_counts().to_dict()
    summary['geometry_types'] = {t: int(c) for t, c in geom_types.items()}

    # Bounding box
    try:
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        summary['bounding_box'] = {
            'minx': round(float(bounds[0]), 6),
            'miny': round(float(bounds[1]), 6),
            'maxx': round(float(bounds[2]), 6),
            'maxy': round(float(bounds[3]), 6),
        }
        # Spatial extent note
        x_span = float(bounds[2] - bounds[0])
        y_span = float(bounds[3] - bounds[1])
        if gdf.crs and gdf.crs.is_projected:
            summary['notes'].append(
                f'Extent: {x_span/1000:.2f} km x {y_span/1000:.2f} km (projected units)'
            )
        else:
            summary['notes'].append(
                f'Extent: {x_span:.4f} x {y_span:.4f} degrees'
            )
    except Exception:
        pass

    # Attribute schema (geometry column excluded)
    geo_col = gdf.geometry.name
    attr_cols = [c for c in gdf.columns if c != geo_col]

    for col in attr_cols:
        summary['attribute_schema'].append({
            'name':         col,
            'dtype':        str(gdf[col].dtype),
            'null_count':   int(gdf[col].isnull().sum()),
            'unique_count': int(gdf[col].nunique()),
        })
        # Mirror to 'columns' for context assembler compatibility
        summary['columns'].append({'name': col, 'dtype': str(gdf[col].dtype)})

    # Numeric attribute statistics
    num_cols = gdf[attr_cols].select_dtypes(include='number').columns.tolist()
    if num_cols:
        desc = gdf[attr_cols][num_cols].describe()
        for col in num_cols:
            summary['numeric_summary'][col] = {
                k: round(float(v), 4) for k, v in desc[col].items()
            }

    # Categorical value frequencies (top 10, first 8 cols)
    cat_cols = gdf[attr_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols[:8]:
        vc = gdf[col].value_counts().head(10).to_dict()
        summary['categorical_summary'][col] = {str(k): int(v) for k, v in vc.items()}

    # Missing values
    summary['missing_values'] = {
        col: int(cnt)
        for col, cnt in gdf[attr_cols].isnull().sum().items()
        if cnt > 0
    }

    # Sample features (first 3, attrs only -- no geometry)
    try:
        sample = gdf[attr_cols].head(3).fillna('').astype(str).to_dict(orient='records')
        summary['sample_features'] = sample
        summary['sample_rows'] = sample  # context assembler compatibility
    except Exception:
        pass

    # Total projected area
    if gdf.crs and gdf.crs.is_projected:
        try:
            total_m2 = float(gdf.geometry.area.sum())
            if total_m2 > 0:
                summary['notes'].append(f'Total feature area: ~{total_m2 / 1e6:.4f} km2 (projected)')
        except Exception:
            pass

    # Archaeological column heuristics
    arch_hints = ['site', 'feature', 'artifact', 'period', 'phase', 'culture',
                  'type', 'material', 'condition', 'description', 'name',
                  'context', 'stratum', 'layer', 'excavat', 'survey']
    arch_cols = [c for c in attr_cols if any(h in c.lower() for h in arch_hints)]
    if arch_cols:
        summary['notes'].append(f'Archaeological attribute columns detected: {arch_cols}')

    # Coordinate heuristics (field-survey point datasets)
    coord_hints = ['lat', 'lon', 'lng', 'latitude', 'longitude', 'x', 'y',
                   'easting', 'northing', 'utm', 'coord']
    coord_cols = [c for c in attr_cols if any(h in c.lower() for h in coord_hints)]
    if coord_cols:
        summary['notes'].append(f'Coordinate attribute columns detected: {coord_cols}')

    summary['analytics'] = run_geospatial_analytics(gdf)
    return summary


def _error_summary(path: Path, msg: str) -> dict:
    return {
        'filename': path.name, 'file_type': path.suffix.lower(),
        'data_type': 'geospatial', 'shape': [0, 0],
        'feature_count': 0, 'crs': None, 'geometry_types': [],
        'bounding_box': None, 'attribute_schema': [],
        'numeric_summary': {}, 'categorical_summary': {},
        'sample_features': [], 'missing_values': {},
        'columns': [], 'sample_rows': [],
        'notes': [f'ERROR: {msg}'],
    }

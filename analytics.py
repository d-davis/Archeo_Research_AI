"""
analytics.py
============
Pre-computation analytics module for ArchAI.

All computations are performed in Python (pandas / numpy / scipy) BEFORE the
LLM sees the data. Results are injected into the Phase 1 context as verified
facts under the "analytics" key, so the LLM interprets results rather than
attempting to calculate them.

Public API
----------
run_tabular_analytics(df, user_prompt="", order_col=None)  -> dict
run_geospatial_analytics(gdf)                              -> dict
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

# scipy is optional but strongly recommended
try:
    from scipy import stats as _stats
    from scipy.spatial.distance import cdist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    warnings.warn(
        "scipy not found. Shapiro-Wilk, correlation, and linear trend "
        "analyses require scipy. Install with: pip install scipy",
        stacklevel=2,
    )

# geopandas is optional (only needed for geospatial analytics)
try:
    import geopandas as gpd
    GEO_OK = True
except ImportError:
    GEO_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Z_THRESHOLD      = 3.0      # Z-score outlier threshold
IQR_MULTIPLIER   = 1.5      # Tukey fence multiplier
SHAPIRO_MIN_N    = 3        # Minimum n for Shapiro-Wilk
SHAPIRO_MAX_N    = 5_000    # Maximum n for Shapiro-Wilk
MIN_N_CORR       = 5        # Minimum n for correlation / trend
MAX_CORR_COLS    = 20       # Cap on pairwise correlation columns
ALPHA            = 0.05     # Significance threshold

# Candidate column names used to auto-detect ordering axis
ORDER_CANDIDATES = [
    "depth", "depth_cm", "depth_m", "level", "layer", "phase",
    "year", "date", "period", "sequence", "order", "stratum", "context",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(val: Any, d: int = 4) -> float | None:
    """Round to d decimal places, or None if non-finite."""
    try:
        v = float(val)
        return round(v, d) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _num_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()[:MAX_CORR_COLS]


def _skew_label(s: float) -> str:
    if abs(s) < 0.5:  return "approximately symmetric"
    if abs(s) < 1.0:  return "moderately skewed"
    return f"highly skewed {'right (positive)' if s > 0 else 'left (negative)'}"


def _kurt_label(k: float) -> str:
    if abs(k) < 0.5:  return "approximately normal (mesokurtic)"
    return "leptokurtic (heavy-tailed)" if k > 0 else "platykurtic (light-tailed)"


def _corr_strength(r: float) -> str:
    a = abs(r)
    if a >= 0.7:  return "strong"
    if a >= 0.5:  return "moderate"
    return "weak"


def _detect_order_col(df: pd.DataFrame) -> str | None:
    for cand in ORDER_CANDIDATES:
        matches = [c for c in df.columns if cand in c.lower()]
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# 1. Descriptive statistics
# ---------------------------------------------------------------------------

def _descriptive(df: pd.DataFrame) -> dict:
    out = {}
    for col in _num_cols(df):
        s = df[col].dropna()
        n = len(s)
        if n == 0:
            out[col] = {"error": "no non-null values"}
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        skew = _f(s.skew())
        kurt = _f(s.kurt())
        entry: dict[str, Any] = {
            "n":        n,
            "mean":     _f(s.mean()),
            "median":   _f(s.median()),
            "mode":     _f(s.mode().iloc[0]) if not s.mode().empty else None,
            "std":      _f(s.std()),
            "variance": _f(s.var()),
            "min":      _f(s.min()),
            "max":      _f(s.max()),
            "range":    _f(s.max() - s.min()),
            "q1":       _f(q1),
            "q3":       _f(q3),
            "iqr":      _f(q3 - q1),
            "skewness": skew,
            "skewness_label": _skew_label(skew) if skew is not None else None,
            "excess_kurtosis": kurt,
            "kurtosis_label":  _kurt_label(kurt) if kurt is not None else None,
        }
        # Shapiro-Wilk normality test
        if SCIPY_OK and SHAPIRO_MIN_N <= n <= SHAPIRO_MAX_N:
            try:
                W, p = _stats.shapiro(s)
                normal = bool(p >= ALPHA)
                entry["shapiro_wilk"] = {
                    "W": _f(W, 6),
                    "p_value": _f(p, 6),
                    "normal_at_alpha_0.05": normal,
                    "interpretation": (
                        f"W={W:.4f}, p={p:.4f}: data consistent with normality (n={n})."
                        if normal else
                        f"W={W:.4f}, p={p:.4f}: data significantly non-normal (n={n}). "
                        "Prefer Spearman over Pearson for correlations."
                    ),
                }
            except Exception as e:
                entry["shapiro_wilk"] = {"error": str(e)}
        elif n > SHAPIRO_MAX_N:
            entry["shapiro_wilk"] = {
                "note": f"n={n} exceeds Shapiro-Wilk limit ({SHAPIRO_MAX_N}). "
                        "Consider Kolmogorov-Smirnov for large samples."
            }
        out[col] = entry
    return out


# ---------------------------------------------------------------------------
# 2. Categorical frequency counts
# ---------------------------------------------------------------------------

def _categorical(df: pd.DataFrame, max_cats: int = 30) -> dict:
    out = {}
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        s = df[col].dropna()
        n = len(s)
        if n == 0 or s.nunique() > max_cats:
            continue
        vc = s.value_counts()
        out[col] = {
            "n": n,
            "n_unique": int(s.nunique()),
            "mode": str(vc.index[0]),
            "frequencies": {str(k): int(v) for k, v in vc.items()},
            "relative_frequencies": {str(k): round(v / n, 4) for k, v in vc.items()},
        }
    return out


# ---------------------------------------------------------------------------
# 3. Outlier detection
# ---------------------------------------------------------------------------

def _outliers(df: pd.DataFrame) -> dict:
    out = {}
    for col in _num_cols(df):
        s = df[col].dropna()
        if len(s) < 4:
            continue
        mean, std = float(s.mean()), float(s.std())
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        z_out = s[abs((s - mean) / std) > Z_THRESHOLD] if std > 0 else pd.Series(dtype=float)
        t_out = s[(s < lo) | (s > hi)]
        out[col] = {
            "z_threshold": Z_THRESHOLD,
            "z_outlier_count": int(len(z_out)),
            "z_outlier_values": [_f(v) for v in z_out.values[:10]],
            "tukey_lower_fence": _f(lo),
            "tukey_upper_fence": _f(hi),
            "tukey_outlier_count": int(len(t_out)),
            "tukey_outlier_values": [_f(v) for v in t_out.values[:10]],
        }
    return out


# ---------------------------------------------------------------------------
# 4. Rate of change
# ---------------------------------------------------------------------------

def _rate_of_change(df: pd.DataFrame, order_col: str | None) -> dict:
    out: dict[str, Any] = {}
    try:
        w = df.sort_values(order_col).copy() if order_col and order_col in df.columns else df.copy()
    except Exception:
        w = df.copy()
    out["ordered_by"] = order_col if order_col and order_col in df.columns else "row_order"
    for col in _num_cols(w):
        s = w[col].dropna()
        if len(s) < 2:
            continue
        abs_ch = s.diff()
        pct_ch = s.pct_change() * 100
        first, last = float(s.iloc[0]), float(s.iloc[-1])
        out[col] = {
            "mean_absolute_change":    _f(abs_ch.mean()),
            "mean_pct_change":         _f(pct_ch.replace([np.inf, -np.inf], np.nan).mean()),
            "max_single_increase":     _f(abs_ch.max()),
            "max_single_decrease":     _f(abs_ch.min()),
            "first_to_last_absolute":  _f(last - first),
            "first_to_last_pct":       _f((last - first) / first * 100) if first != 0 else None,
        }
    return out


# ---------------------------------------------------------------------------
# 5. Ratios
# ---------------------------------------------------------------------------

def _ratios(df: pd.DataFrame) -> dict:
    cols = [c for c in _num_cols(df) if (df[c] > 0).all()]
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = _f(df[a].mean() / df[b].mean())
            if r is not None:
                pairs.append({"numerator": a, "denominator": b, "mean_ratio": r,
                               "note": f"Mean {a} is {r:.2f}x mean {b}"})
            if len(pairs) >= 20:
                break
        if len(pairs) >= 20:
            break
    return {"column_ratios": pairs}


# ---------------------------------------------------------------------------
# 6. Correlation
# ---------------------------------------------------------------------------

def _correlations(df: pd.DataFrame) -> dict:
    if not SCIPY_OK:
        return {"error": "scipy required"}
    cols = _num_cols(df)
    if len(cols) < 2:
        return {"note": "fewer than 2 numeric columns"}
    p_pairs, s_pairs = [], []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            xy = df[[a, b]].dropna()
            if len(xy) < MIN_N_CORR:
                continue
            xa, xb = xy[a].values, xy[b].values
            try:
                rp, pp = _stats.pearsonr(xa, xb)
                if abs(rp) >= 0.3 and pp < ALPHA:
                    p_pairs.append({"col_a": a, "col_b": b, "n": len(xy),
                                    "r": _f(rp), "p_value": _f(pp, 6),
                                    "strength": _corr_strength(rp)})
            except Exception:
                pass
            try:
                rs, ps = _stats.spearmanr(xa, xb)
                if abs(rs) >= 0.3 and ps < ALPHA:
                    s_pairs.append({"col_a": a, "col_b": b, "n": len(xy),
                                    "rho": _f(rs), "p_value": _f(ps, 6),
                                    "strength": _corr_strength(rs)})
            except Exception:
                pass
    return {
        "pearson_significant_pairs": p_pairs,
        "spearman_significant_pairs": s_pairs,
        "note": f"Reporting |r|>=0.3 and p<{ALPHA}. Use Spearman for non-normal data.",
    }


# ---------------------------------------------------------------------------
# 7. Cross-tabulation
# ---------------------------------------------------------------------------

def _crosstab(df: pd.DataFrame, max_cats: int = 10) -> dict:
    cat_cols = [c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
                if df[c].nunique() <= max_cats]
    results = []
    for i, a in enumerate(cat_cols):
        for b in cat_cols[i + 1:]:
            ct = pd.crosstab(df[a], df[b])
            results.append({"col_a": a, "col_b": b, "crosstab": ct.to_dict()})
            if len(results) >= 6:
                break
        if len(results) >= 6:
            break
    return {"cross_tabulations": results}


# ---------------------------------------------------------------------------
# 8. Linear trend
# ---------------------------------------------------------------------------

def _linear_trend(df: pd.DataFrame, order_col: str | None) -> dict:
    if not SCIPY_OK:
        return {"error": "scipy required"}
    w = df.copy()
    if order_col and order_col in w.columns:
        w = w.sort_values(order_col)
        try:
            x = w[order_col].values.astype(float)
            x_label = order_col
        except (ValueError, TypeError):
            x = np.arange(len(w), dtype=float)
            x_label = "row_index"
            order_col = None
    else:
        x = np.arange(len(w), dtype=float)
        x_label = "row_index"
    out = {}
    for col in _num_cols(w):
        if col == order_col:
            continue
        xy = pd.DataFrame({"x": x, "y": w[col].values}).dropna()
        if len(xy) < MIN_N_CORR:
            continue
        try:
            slope, intercept, r, p, se = _stats.linregress(xy["x"].values, xy["y"].values)
            out[col] = {
                "x_variable":       x_label,
                "slope":            _f(slope),
                "intercept":        _f(intercept),
                "r_squared":        _f(r ** 2),
                "p_value":          _f(p, 6),
                "std_error":        _f(se),
                "trend_direction":  "increasing" if slope > 0 else "decreasing",
                "significant":      bool(p < ALPHA),
            }
        except Exception as e:
            out[col] = {"error": str(e)}
    return out


# ---------------------------------------------------------------------------
# 9. Geospatial helpers
# ---------------------------------------------------------------------------

def _geo_bbox(gdf) -> dict:
    try:
        b = gdf.total_bounds
        return {"min_x": _f(b[0]), "min_y": _f(b[1]),
                "max_x": _f(b[2]), "max_y": _f(b[3]),
                "width": _f(b[2] - b[0]), "height": _f(b[3] - b[1])}
    except Exception as e:
        return {"error": str(e)}


def _geo_density(gdf) -> dict:
    try:
        b = gdf.total_bounds
        if gdf.crs and gdf.crs.is_geographic:
            lat_mid = (b[1] + b[3]) / 2
            w_km = (b[2] - b[0]) * 111.32 * math.cos(math.radians(lat_mid))
            h_km = (b[3] - b[1]) * 110.574
        else:
            w_km = (b[2] - b[0]) / 1000
            h_km = (b[3] - b[1]) / 1000
        area = w_km * h_km
        return {
            "bounding_box_area_km2":    _f(area),
            "feature_count":            len(gdf),
            "feature_density_per_km2":  _f(len(gdf) / area) if area > 0 else None,
        }
    except Exception as e:
        return {"error": str(e)}


def _geo_nn(gdf) -> dict:
    try:
        if not SCIPY_OK:
            return {"error": "scipy required for nearest-neighbour distance"}
        geom_types = gdf.geom_type.unique()
        if not any(g in ("Point", "MultiPoint") for g in geom_types):
            return {"note": "Nearest-neighbour distance computed for Point layers only."}
        pts = gdf[gdf.geom_type == "Point"]
        if len(pts) < 2:
            return {"note": "Fewer than 2 point features."}
        coords = np.array([[p.x, p.y] for p in pts.geometry])
        dm = cdist(coords, coords)
        np.fill_diagonal(dm, np.inf)
        min_d = dm.min(axis=1)
        return {
            "n_points":               len(pts),
            "mean_nn_distance":       _f(min_d.mean()),
            "min_nn_distance":        _f(min_d.min()),
            "max_nn_distance":        _f(min_d.max()),
            "units": "degrees if geographic CRS, metres if projected CRS",
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tabular_analytics(
    df: pd.DataFrame,
    user_prompt: str = "",
    order_col: str | None = None,
) -> dict:
    """
    Run the full tabular analytics suite on a DataFrame.

    Parameters
    ----------
    df          : pandas DataFrame (loaded by tabular preprocessor)
    user_prompt : user research question (reserved for future intent parsing)
    order_col   : optional ordering column (depth, year, layer, etc.)
                  Auto-detected from common archaeological field names if None.

    Returns
    -------
    dict with keys: row_count, column_count, order_column_detected,
                    descriptive, categorical, outliers, rate_of_change,
                    ratios, correlations, cross_tabulation, linear_trend
    """
    if df.empty:
        return {"error": "DataFrame is empty; no analytics computed."}

    if order_col is None:
        order_col = _detect_order_col(df)

    result: dict[str, Any] = {
        "row_count":              len(df),
        "column_count":           len(df.columns),
        "order_column_detected":  order_col,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result["descriptive"]       = _descriptive(df)
        result["categorical"]       = _categorical(df)
        result["outliers"]          = _outliers(df)
        result["rate_of_change"]    = _rate_of_change(df, order_col)
        result["ratios"]            = _ratios(df)
        result["correlations"]      = _correlations(df)
        result["cross_tabulation"]  = _crosstab(df)
        result["linear_trend"]      = _linear_trend(df, order_col)

    return result


def run_geospatial_analytics(gdf) -> dict:
    """
    Run geospatial analytics on a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas GeoDataFrame (loaded by geospatial preprocessor)

    Returns
    -------
    dict with keys: feature_count, geometry_types, crs,
                    bounding_box, feature_density, nearest_neighbour,
                    attribute_analytics
    """
    if not GEO_OK:
        return {"error": "geopandas not available; geospatial analytics skipped."}

    result: dict[str, Any] = {
        "feature_count":   len(gdf),
        "geometry_types":  gdf.geom_type.value_counts().to_dict(),
        "crs":             str(gdf.crs) if gdf.crs else "undefined",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result["bounding_box"]       = _geo_bbox(gdf)
        result["feature_density"]    = _geo_density(gdf)
        result["nearest_neighbour"]  = _geo_nn(gdf)

        # Tabular analytics on the attribute table
        attr_df = gdf.drop(columns=gdf.geometry.name, errors="ignore")
        if not attr_df.empty:
            result["attribute_analytics"] = run_tabular_analytics(attr_df)

    return result

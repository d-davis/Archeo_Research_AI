"""
Tabular data preprocessor: CSV, XLSX, TXT.

Extracts a structured summary dict. No raw cell values reach the LLM --
only schema metadata, statistics, detected patterns, and a 5-row sample.

Heuristics auto-detect coordinate and temporal columns common in
archaeological survey data and flag them in notes.
"""
from pathlib import Path
from typing import Union
import pandas as pd


def preprocess_tabular(file_path: Union[str, Path]) -> dict:
    """Load and summarize a tabular file. Returns structured summary dict."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == '.csv':
        df = pd.read_csv(path, low_memory=False)
    elif ext in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    elif ext == '.txt':
        df = None
        for sep in ['\t', ',', '|']:
            try:
                candidate = pd.read_csv(path, sep=sep)
                if candidate.shape[1] > 1:
                    df = candidate
                    break
            except Exception:
                continue
        if df is None:
            return _summarize_text_doc(path)
    else:
        raise ValueError(f"Unsupported file type for Phase 1: {ext}")

    return _summarize_dataframe(df, path)


def _summarize_dataframe(df: pd.DataFrame, path: Path) -> dict:
    summary = {
        'filename':            path.name,
        'file_type':           path.suffix.lower(),
        'data_type':           'tabular',
        'shape':               list(df.shape),
        'columns':             [],
        'numeric_summary':     {},
        'categorical_summary': {},
        'sample_rows':         [],
        'missing_values':      {},
        'notes':               [],
    }

    # Column metadata
    for col in df.columns:
        summary['columns'].append({
            'name':         col,
            'dtype':        str(df[col].dtype),
            'null_count':   int(df[col].isnull().sum()),
            'unique_count': int(df[col].nunique()),
        })

    # Numeric statistics
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        desc = df[num_cols].describe()
        for col in num_cols:
            summary['numeric_summary'][col] = {
                k: round(float(v), 4) for k, v in desc[col].items()
            }

    # Categorical value frequencies (top 10, first 10 columns)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols[:10]:
        vc = df[col].value_counts().head(10).to_dict()
        summary['categorical_summary'][col] = {str(k): int(v) for k, v in vc.items()}

    # 5-row sample
    try:
        summary['sample_rows'] = (
            df.head(5).fillna('').astype(str).to_dict(orient='records')
        )
    except Exception:
        summary['notes'].append("Could not serialize sample rows.")

    # Missing values
    summary['missing_values'] = {
        col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0
    }

    # Duplicates
    dup = int(df.duplicated().sum())
    if dup > 0:
        summary['notes'].append(f"{dup} duplicate rows detected.")

    # Coordinate columns heuristic
    coord_hints = ['lat', 'lon', 'lng', 'latitude', 'longitude',
                   'x', 'y', 'easting', 'northing', 'coord', 'utm']
    coord_cols = [c for c in df.columns if any(h in c.lower() for h in coord_hints)]
    if coord_cols:
        summary['notes'].append(f"Coordinate columns detected: {coord_cols}")
        for col in coord_cols:
            if col in summary['numeric_summary']:
                r = summary['numeric_summary'][col]
                summary['notes'].append(
                    f"  {col}: range [{r.get('min')}, {r.get('max')}], mean {r.get('mean')}"
                )

    # Temporal columns heuristic
    date_hints = ['date', 'year', 'period', 'age', 'century', 'phase', 'time']
    date_cols = [c for c in df.columns if any(h in c.lower() for h in date_hints)]
    if date_cols:
        summary['notes'].append(f"Temporal columns detected: {date_cols}")

    return summary


def _summarize_text_doc(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    lines = text.splitlines()
    return {
        'filename':             path.name,
        'file_type':            '.txt',
        'data_type':            'text_document',
        'shape':                [len(lines), 1],
        'word_count':           len(text.split()),
        'line_count':           len(lines),
        'preview':              text[:2000],
        'columns':              [],
        'numeric_summary':      {},
        'categorical_summary':  {},
        'sample_rows':          [],
        'missing_values':       {},
        'notes':                ['File treated as plain text document (non-tabular).'],
    }
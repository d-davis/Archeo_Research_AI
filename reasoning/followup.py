"""
Follow-up question answering with inline chart generation.
run_followup() returns (answer_text: str, chart_paths: list[Path])
Five fallback passes for chart detection:
  1. Explicit CHART_REQUEST: lines
  2. JSON code blocks
  3. Text X-axis/Y-axis/Data blocks
  4. Empty code blocks preceding chart prose
  5. Unconditional prose inference
"""
import json
import re
from pathlib import Path

import ollama

from session import get_history_text
from visualize import generate_followup_charts

FOLLOWUP_SYSTEM_PROMPT = """You are an expert archaeological analyst in an ongoing research conversation.
You have access to the original datasets (as structured summaries), the per-file Phase 1
analyses, the final interpretation report, and the conversation history.

RULES:
1. Ground every answer in the provided data. Cite specific files when relevant.
2. Be concise and focused. Follow-up answers are not full reports.
3. If the question asks about something the available data cannot answer, say so clearly.
4. If the user asks to add new data or restart the analysis, do not comply.
   Instead, instruct them to start a new session with:
   python main.py --resume <session_file.json> --files <new_files>
5. Distinguish observation (what the data shows) from inference (what you conclude).
6. Do not reproduce large portions of the original report unnecessarily.

CHART REQUESTS:
If a chart or plot would help answer the question, embed a CHART_REQUEST: line
BEFORE your prose answer, using this exact format (no markdown wrapper, no backticks):

CHART_REQUEST:{"type":"timeseries","file":"survey.csv","x":"Year","y":"Discharge","title":"Discharge Over Time"}

Available chart types and required fields:
  histogram            file, column, title
  boxplot              file, column, title
  bar                  file, column, title
  timeseries           file, x, y, title
  scatter              file, x, y, title
  correlation_heatmap  file, title
  spatial              file, lat, lon, title
  kde_heatmap          file, lat, lon, title           
  geometry_plot        file, title, column (optional)  
  spatial_colored      file, lat, lon, color_col, title

CROSS-FILE COMPARISONS:
When the user asks which file had the greatest/least of something, look at the
PRE-COMPUTED COLUMN TOTALS PER FILE section and compare values directly. State
the answer using the filename. Do not assume what each file represents unless the user's
question or the filename makes this explicit. 

PROHIBITED: Never describe a chart as text or a code block.
Do NOT wrap CHART_REQUEST: in backticks or any markdown.
Use the exact filename from FILES ANALYZED.
Only request a chart when it genuinely adds analytical value."""


# ── Helper: normalise loose JSON chart descriptions ───────────────────────────

def _normalize_loose_chart(obj: dict, session: dict) -> dict | None:
    type_map = {
        'line_chart': 'timeseries', 'line': 'timeseries',
        'scatter_plot': 'scatter', 'scatter': 'scatter',
        'bar_chart': 'bar', 'bar': 'bar',
        'histogram': 'histogram',
        'box_plot': 'boxplot', 'boxplot': 'boxplot',
        'heatmap': 'correlation_heatmap', 'correlation': 'correlation_heatmap',
        'grouped_bar': '_grouped_bar_inline',
        'kde': 'kde_heatmap', 'kernel_density': 'kde_heatmap',
        'geometry': 'geometry_plot', 'map': 'geometry_plot',
    }
    raw_type = obj.get('chart_type', obj.get('type', ''))
    chart_type = type_map.get(raw_type.lower().replace(' ', '_'))
    if not chart_type:
        return None
    title = obj.get('title', 'Chart')
    files = session.get('files_analyzed', [])
    tabular_exts = {'.csv', '.xlsx', '.xls', '.txt'}
    file = next(
        (f for f in files if Path(f).suffix.lower() in tabular_exts),
        files[0] if files else '',
    )
    data = obj.get('data', {})
    x_list = data.get('x_axis', data.get('x', []))
    y_list = data.get('y_axis', data.get('y', []))
    x = x_list[0] if isinstance(x_list, list) and x_list else (x_list or '')
    y = y_list[0] if isinstance(y_list, list) and y_list else (y_list or '')
    req: dict = {'type': chart_type, 'file': file, 'title': title}
    if chart_type in ('timeseries', 'scatter'):
        req['x'] = x
        req['y'] = y
    elif chart_type in ('histogram', 'boxplot', 'bar'):
        req['column'] = y or x
    return req if (req.get('column') or req.get('x') or chart_type in
                   ('correlation_heatmap', '_grouped_bar_inline',
                    'kde_heatmap', 'geometry_plot')) else None


# ── Helper: parse text X-axis/Y-axis/Data blocks ──────────────────────────────

def _parse_text_chart(block: str, session: dict) -> dict | None:
    lines_in = block.strip().splitlines()
    x_label, y_label, data_lines = '', '', []
    in_data = False
    for line in lines_in:
        stripped = line.strip()
        if stripped.lower().startswith('x-axis:'):
            x_label = stripped.split(':', 1)[1].strip()
        elif stripped.lower().startswith('y-axis:'):
            y_label = stripped.split(':', 1)[1].strip()
        elif stripped.lower().startswith('data:'):
            in_data = True
        elif in_data and stripped.startswith('-'):
            data_lines.append(stripped[1:].strip())
    if not data_lines:
        return None
    categories: list[str] = []
    series_data: dict[str, list[float]] = {}
    value_pattern = re.compile(r'([\d.]+)%?\s*\(([^)]+)\)')
    for dl in data_lines:
        if ':' not in dl:
            continue
        cat, rest = dl.split(':', 1)
        categories.append(cat.strip())
        for m in value_pattern.finditer(rest):
            val, sname = float(m.group(1)), m.group(2).strip()
            series_data.setdefault(sname, []).append(val)
    if not categories or not series_data:
        return None
    files = session.get('files_analyzed', [])
    tabular_exts = {'.csv', '.xlsx', '.xls', '.txt'}
    file = next(
        (f for f in files if Path(f).suffix.lower() in tabular_exts),
        files[0] if files else '',
    )
    return {
        'type': '_grouped_bar_inline', 'file': file,
        'title': y_label or 'Chart', 'categories': categories,
        'series': series_data, 'x_label': x_label, 'y_label': y_label,
    }


# ── Helper: column name fuzzy matching ────────────────────────────────────────

def _best_col_match(text: str, known_cols: set) -> str | None:
    text_clean = text.strip().lower().replace(' ', '_')
    for col in known_cols:
        if col.lower() == text_clean or col.lower() == text.strip().lower():
            return col
    for col in known_cols:
        if text_clean in col.lower() or col.lower() in text_clean:
            return col
    return None


# ── Helper: parse natural-language chart requests ─────────────────────────────

def _parse_natural_language_chart(text: str, session: dict) -> dict | None:
    known_cols = set()
    for fs in session.get('preprocessed_summaries', []):
        if not isinstance(fs, dict):
            continue
        for col in fs.get('columns', []):
            if isinstance(col, dict):
                known_cols.add(col.get('name', ''))
        for col in fs.get('attribute_schema', []):
            if isinstance(col, dict):
                known_cols.add(col.get('name', ''))
    known_cols.discard('')
    files = session.get('files_analyzed', [])
    tabular_exts = {'.csv', '.xlsx', '.xls', '.txt'}
    file = next(
        (f for f in files if Path(f).suffix.lower() in tabular_exts),
        files[0] if files else '',
    )
    text = text.strip()
    # "X by Y"
    by_match = re.match(r'(.+?)\s+by\s+(.+?)(?:\s+from\s+(\S+))?$', text, re.IGNORECASE)
    if by_match:
        y_raw, x_raw = by_match.group(1).strip(), by_match.group(2).strip()
        if by_match.group(3):
            file = by_match.group(3)
        y_col = _best_col_match(y_raw, known_cols)
        x_col = _best_col_match(x_raw, known_cols)
        if y_col and x_col:
            return {'type': 'bar', 'file': file, 'column': y_col,
                    'title': f'{y_col} by {x_col}', 'x': x_col, 'y': y_col}
        if y_col:
            return {'type': 'bar', 'file': file, 'column': y_col,
                    'title': f'{y_col} by {x_raw}'}
    # "X vs Y"
    vs_match = re.search(r'(\w[\w\s]*?)\s+vs\.?\s+(\w[\w\s]*)', text, re.IGNORECASE)
    if vs_match:
        x_col = _best_col_match(vs_match.group(1).strip(), known_cols)
        y_col = _best_col_match(vs_match.group(2).strip(), known_cols)
        if x_col and y_col:
            return {'type': 'scatter', 'file': file, 'x': x_col, 'y': y_col,
                    'title': f'{x_col} vs {y_col}'}
    # "histogram of X"
    hist_match = re.search(r'histogram\s+of\s+(.+)', text, re.IGNORECASE)
    if hist_match:
        col = _best_col_match(hist_match.group(1).strip(), known_cols)
        if col:
            return {'type': 'histogram', 'file': file, 'column': col,
                    'title': f'Distribution of {col}'}
    col = _best_col_match(text, known_cols)
    if col:
        return {'type': 'bar', 'file': file, 'column': col, 'title': col}
    return None


# ── Helper: infer chart from prose when no explicit request found ─────────────

def _infer_chart_from_prose(text: str, session: dict) -> dict | None:
    known_cols = set()
    for fs in session.get('preprocessed_summaries', []):
        if not isinstance(fs, dict):
            continue
        for col in fs.get('columns', []):
            if isinstance(col, dict):
                known_cols.add(col.get('name', ''))
        for col in fs.get('attribute_schema', []):
            if isinstance(col, dict):
                known_cols.add(col.get('name', ''))
    known_cols.discard('')

    files = session.get('files_analyzed', [])
    tabular_exts = {'.csv', '.xlsx', '.xls', '.txt'}
    file = next(
        (f for f in files if Path(f).suffix.lower() in tabular_exts),
        files[0] if files else '',
    )

    # Find column names mentioned in the text
    mentioned = [
        col for col in known_cols
        if col.lower() in text.lower()
        or col.lower().replace('_', ' ') in text.lower()
        or col.lower().replace(' ', '_') in text.lower()
    ]

    text_lower = text.lower()

    # KDE / spatial density
    if any(kw in text_lower for kw in ('kernel density', 'kde', 'density map',
                                        'heatmap', 'heat map')):
        lat_col = next((c for c in known_cols
                        if any(h in c.lower() for h in ['lat', 'y', 'northing'])), None)
        lon_col = next((c for c in known_cols
                        if any(h in c.lower() for h in ['lon', 'lng', 'x', 'easting'])), None)
        if lat_col and lon_col:
            return {'type': 'kde_heatmap', 'file': file,
                    'lat': lat_col, 'lon': lon_col,
                    'title': 'Kernel Density Estimation'}
        geo_files = [f for f in files if Path(f).suffix.lower() in ('.shp', '.geojson', '.json')]
        if geo_files:
            return {'type': 'geometry_plot', 'file': geo_files[0],
                    'title': 'Spatial Distribution'}

    # Spatial / geographic
    if any(kw in text_lower for kw in ('spatial', 'geographic', 'map of',
                                        'geometry', 'coordinates')):
        geo_files = [f for f in files if Path(f).suffix.lower() in ('.shp', '.geojson', '.json')]
        if geo_files:
            return {'type': 'geometry_plot', 'file': geo_files[0],
                    'title': 'Feature Distribution Map'}

    if not mentioned:
        return None

    # Histogram / distribution
    if 'histogram' in text_lower or 'distribution' in text_lower:
        return {'type': 'histogram', 'file': file,
                'column': mentioned[0], 'title': f'Distribution of {mentioned[0]}'}

    # Scatter
    if 'scatter' in text_lower:
        if len(mentioned) >= 2:
            return {'type': 'scatter', 'file': file,
                    'x': mentioned[0], 'y': mentioned[1],
                    'title': f'{mentioned[0]} vs {mentioned[1]}'}
        return {'type': 'histogram', 'file': file,
                'column': mentioned[0], 'title': f'Distribution of {mentioned[0]}'}

    # Time series
    if 'trend' in text_lower or 'over time' in text_lower or 'time series' in text_lower:
        return {'type': 'timeseries', 'file': file,
                'x': mentioned[1] if len(mentioned) > 1 else mentioned[0],
                'y': mentioned[0], 'title': f'{mentioned[0]} over time'}

    # Bar / frequency
    if 'bar' in text_lower or 'frequency' in text_lower or 'count' in text_lower:
        return {'type': 'bar', 'file': file,
                'column': mentioned[0], 'title': f'{mentioned[0]} frequency'}

    # Default
    return {'type': 'bar', 'file': file,
            'column': mentioned[0], 'title': mentioned[0]}


# ── Main function ─────────────────────────────────────────────────────────────

def run_followup(question: str, session: dict) -> tuple[str, list]:
    """
    Answer a follow-up question using full session context.
    Returns (answer_text, chart_paths).
    """
    file_summaries_str = json.dumps(
        session.get('preprocessed_summaries', []), indent=2, default=str)
    phase1_str = json.dumps(
        session.get('phase1_results', []), indent=2, default=str)
    history_text = get_history_text(session)

    col_info = '\n'.join(
        f'  {fs["filename"]}: ' +
        ', '.join(f'{c["name"]} ({c["dtype"]})' for c in fs.get('columns', [])
                  if isinstance(c, dict) and 'name' in c)
        for fs in session.get('preprocessed_summaries', [])
        if isinstance(fs, dict) and fs.get('columns')
    )

    # Pre-compute column totals per file for cross-file comparison
    totals_summary = []
    for fs in session.get('preprocessed_summaries', []):
        if not isinstance(fs, dict):
            continue
        ct = fs.get('column_totals', {})
        if ct:
            # Only include numeric columns, cap at 15 per file
            items = list(ct.items())[:15]
            totals_summary.append(
                f'  {fs["filename"]}: ' +
                ', '.join(f'{col}={val}' for col, val in items)
            )
    totals_block = 'PRE-COMPUTED COLUMN TOTALS PER FILE (ground truth, use verbatim):\n' + '\n'.join(totals_summary) if totals_summary else ''

    user_message = (
        f'ORIGINAL RESEARCH QUESTION: {session["original_prompt"]}\n\n'
        f'FILES ANALYZED: {", ".join(session.get("files_analyzed", []))}\n\n'
        + (f'COLUMN NAMES AND TYPES:\n{col_info}\n\n' if col_info else '')
        + (f'{totals_block}\n\n' if totals_block else '')
        + f'FILE SUMMARIES:\n{file_summaries_str}\n\n'
        f'PHASE 1 ANALYSES:\n{phase1_str}\n\n'
        f'FINAL INTERPRETATION REPORT:\n{session["final_narrative"]}\n\n'
        + (f'CONVERSATION HISTORY:\n{history_text}\n\n' if history_text else '')
        + f'FOLLOW-UP QUESTION: {question}\n\n'
        'If a chart would help, embed a CHART_REQUEST: line first (no backticks). '
        'Then write your prose answer.'
    )

    print(f"DEBUG USER MESSAGE (first 1000 chars):\n{user_message[:1000]}")#DEBUG
    response = ollama.chat(
        model=session['model'],
        messages=[
            {'role': 'system', 'content': FOLLOWUP_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message},
        ],
        options={'temperature': 0.1},
    )

    raw = response['message']['content']
    print(f"DEBUG RAW:\n{raw[:800]}") #DEBUG

    # ── Pass 1: explicit CHART_REQUEST: lines ─────────────────────────────
    chart_requests: list[dict] = []
    clean_lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if 'CHART_REQUEST:' in stripped:
            json_part = stripped.split('CHART_REQUEST:', 1)[1].strip().strip('`').strip()
            try:
                chart_requests.append(json.loads(json_part))
            except json.JSONDecodeError:
                req = _parse_natural_language_chart(json_part, session)
                if req:
                    chart_requests.append(req)
        else:
            clean_lines.append(line)

    cleaned_text = '\n'.join(clean_lines)

    # ── Pass 2: JSON code-block fallback ──────────────────────────────────
    json_block_re = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
    for m in json_block_re.finditer(cleaned_text):
        try:
            obj = json.loads(m.group(1))
            if 'chart_type' in obj or ('type' in obj and 'data' in obj):
                req = _normalize_loose_chart(obj, session)
                if req:
                    chart_requests.append(req)
                    cleaned_text = cleaned_text.replace(m.group(0), '').strip()
        except json.JSONDecodeError:
            pass

    # ── Pass 3: X-axis/Y-axis/Data text blocks ────────────────────────────
    text_block_re = re.compile(r'```[^\n]*\n(.*?)```', re.DOTALL)
    for m in text_block_re.finditer(cleaned_text):
        block = m.group(1)
        if 'X-axis:' in block and 'Y-axis:' in block and 'Data:' in block:
            req = _parse_text_chart(block, session)
            if req:
                chart_requests.append(req)
                cleaned_text = cleaned_text.replace(m.group(0), '').strip()

    # ── Pass 4: empty code blocks ─────────────────────────────────────────
    empty_block_re = re.compile(r'```\s*```', re.DOTALL)
    if empty_block_re.search(cleaned_text):
        cleaned_text = empty_block_re.sub('', cleaned_text).strip()
        if not chart_requests:
            inferred = _infer_chart_from_prose(cleaned_text, session)
            if inferred:
                chart_requests.append(inferred)

    # ── Pass 5: unconditional prose inference ─────────────────────────────
    chart_keywords = {
        'scatterplot', 'scatter plot', 'histogram', 'bar chart', 'bar graph',
        'line chart', 'line plot', 'chart shows', 'plot shows', 'plot of',
        'chart of', 'the following chart', 'the following plot', 'visualiz',
        'kernel density', 'kde', 'density map', 'heatmap', 'heat map',
        'spatial distribution', 'point distribution', 'map of',
        'geometry', 'geographic', 'coordinates',
    }
    if not chart_requests and any(kw in cleaned_text.lower() for kw in chart_keywords):
        inferred = _infer_chart_from_prose(cleaned_text, session)
        if inferred:
            chart_requests.append(inferred)

    # ── Strip remaining ASCII box art ─────────────────────────────────────
    ascii_box_re = re.compile(r'```[^\n]*\n[\s\S]*?[+\-|=]{3,}[\s\S]*?```', re.DOTALL)
    cleaned_text = ascii_box_re.sub('', cleaned_text).strip()

    answer_text = cleaned_text.strip()

    # ── Generate charts ───────────────────────────────────────────────────
    chart_paths: list = []
    if chart_requests:
        session_id = session.get('session_id', 'followup')
        chart_paths = generate_followup_charts(
            chart_requests=chart_requests,
            session=session,
            session_id=session_id,
        )
    #DEBUG
    print(f"DEBUG chart_requests: {chart_requests}")
    print(f"DEBUG chart_paths: {chart_paths}")
    print(f"DEBUG file_paths: {session.get('file_paths', {})}")
    return answer_text, chart_paths

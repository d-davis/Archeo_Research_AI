"""
Follow-up question answering with inline chart generation.

The LLM may request charts using CHART_REQUEST: lines embedded in its response.
These are parsed out, used to generate PNGs, and stripped from the visible text.

run_followup() returns (answer_text: str, chart_paths: list[Path])

Three chart sources are supported:
  1. Explicit CHART_REQUEST: lines (preferred)
  2. JSON code blocks describing a chart (fallback)
  3. Text-based X-axis/Y-axis/Data code blocks (second fallback)
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
BEFORE your prose answer, using this exact format (no markdown wrapper):

CHART_REQUEST:{"type":"timeseries","file":"survey.csv","x":"Year","y":"Discharge","title":"Discharge Over Time"}

Available chart types and required fields:
  histogram           file, column, title
  boxplot             file, column, title
  bar                 file, column, title
  timeseries          file, x, y, title
  scatter             file, x, y, title
  correlation_heatmap file, title
  spatial             file, lat, lon, title

PROHIBITED: Never describe a chart as text, ASCII, or a code block with X-axis /
Y-axis / Data labels. Those produce no graphic. If you cannot express data as a
CHART_REQUEST type, state the numbers in prose instead.

Use the exact filename as it appears in FILES ANALYZED.
Only request a chart when it genuinely adds analytical value."""


def _normalize_loose_chart(obj: dict, session: dict) -> dict | None:
    """Convert LLM free-form chart JSON into a normalised CHART_REQUEST dict."""
    type_map = {
        'line_chart': 'timeseries', 'line': 'timeseries',
        'scatter_plot': 'scatter', 'scatter': 'scatter',
        'bar_chart': 'bar', 'bar': 'bar',
        'histogram': 'histogram',
        'box_plot': 'boxplot', 'boxplot': 'boxplot',
        'heatmap': 'correlation_heatmap', 'correlation': 'correlation_heatmap',
        'grouped_bar': '_grouped_bar_inline',
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
    return req if (req.get('column') or req.get('x') or chart_type in ('correlation_heatmap', '_grouped_bar_inline')) else None


def _parse_text_chart(block: str, session: dict) -> dict | None:
    """
    Convert a text-based chart description (X-axis / Y-axis / Data block)
    into a _grouped_bar_inline CHART_REQUEST dict.
    """
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

    # Parse "Early Period: 30% (Site A), 40% (Site C), 50% (Site B)"
    categories: list[str] = []
    series_data: dict[str, list[float]] = {}
    value_pattern = re.compile(r'([d.]+)%?s*(([^)]+))')

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
        'type': '_grouped_bar_inline',
        'file': file,
        'title': y_label or 'Chart',
        'categories': categories,
        'series': series_data,
        'x_label': x_label,
        'y_label': y_label,
    }


def run_followup(question: str, session: dict) -> tuple[str, list]:
    """
    Answer a follow-up question using full session context.

    Args:
        question: The researcher's follow-up question
        session:  The current session object

    Returns:
        Tuple of (answer_text, chart_paths) where chart_paths is a list
        of Path objects for any generated chart PNGs.
    """
    file_summaries_str = json.dumps(
        session.get('preprocessed_summaries', []),
        indent=2, default=str,
    )
    phase1_str = json.dumps(
        session.get('phase1_results', []),
        indent=2, default=str,
    )
    history_text = get_history_text(session)

    user_message = (
        f'ORIGINAL RESEARCH QUESTION: {session["original_prompt"]}\n\n'
        f'FILES ANALYZED: {", ".join(session.get("files_analyzed", []))}\n\n'
        f'FILE SUMMARIES:\n{file_summaries_str}\n\n'
        f'PHASE 1 ANALYSES:\n{phase1_str}\n\n'
        f'FINAL INTERPRETATION REPORT:\n{session["final_narrative"]}\n\n'
        + (f'CONVERSATION HISTORY:\n{history_text}\n\n' if history_text else '')
        + f'FOLLOW-UP QUESTION: {question}\n\n'
        'If a chart would help, embed a CHART_REQUEST: line first. '
        'Then write your prose answer.'
    )

    response = ollama.chat(
        model=session['model'],
        messages=[
            {'role': 'system', 'content': FOLLOWUP_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message},
        ],
        options={'temperature': 0.3},
    )

    raw = response['message']['content']

    # ── Pass 1: parse explicit CHART_REQUEST: lines ────────────────────────
    chart_requests: list[dict] = []
    clean_lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith('CHART_REQUEST:'):
            json_part = stripped[len('CHART_REQUEST:'):].strip()
            try:
                chart_requests.append(json.loads(json_part))
            except json.JSONDecodeError:
                pass
        else:
            clean_lines.append(line)

    cleaned_text = '\n'.join(clean_lines)

    # ── Pass 2: JSON code-block fallback ───────────────────────────────────
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

    # ── Pass 3: text chart block fallback (X-axis / Y-axis / Data) ────────
    text_block_re = re.compile(r'```[^\n]*\n(.*?)```', re.DOTALL)
    for m in text_block_re.finditer(cleaned_text):
        block = m.group(1)
        if 'X-axis:' in block and 'Y-axis:' in block and 'Data:' in block:
            req = _parse_text_chart(block, session)
            if req:
                chart_requests.append(req)
                cleaned_text = cleaned_text.replace(m.group(0), '').strip()

    answer_text = cleaned_text.strip()

    # ── Generate charts ────────────────────────────────────────────────────
    chart_paths: list = []
    if chart_requests:
        session_id = session.get('session_id', 'followup')
        chart_paths = generate_followup_charts(
            chart_requests=chart_requests,
            session=session,
            session_id=session_id,
        )

    return answer_text, chart_paths

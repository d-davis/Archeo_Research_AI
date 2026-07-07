"""
Phase 2: Cross-file synthesis and narrative generation.
"""
import json
from typing import List
import ollama
from config import get_num_predict

PHASE2_SYSTEM_PROMPT = """You are an expert archaeological analyst producing a formal interpretive report.
You have received individual analyses of one or more datasets (Phase 1 results).

RULES:
1. Ground every claim in the provided data. Cite which file(s) support each claim.
2. Identify cross-dataset patterns, correlations, and contradictions explicitly.
3. Clearly distinguish evidence (what the data shows) from inference (what you conclude).
4. Flag where data is insufficient or where alternative interpretations are equally valid.
5. Use appropriate epistemic caution. Do not introduce information not in the provided analyses.
6. Only reference variables present in the provided analyses.
7. Use PRE-COMPUTED COLUMN TOTALS PER FILE verbatim for cross-file comparisons.

Format your response as Markdown with these sections:
## Summary
## Data Overview
## Key Findings
## Cross-Dataset Patterns
## Archaeological Interpretation
## Limitations and Uncertainties
## Recommended Next Steps

Professional language, 600-1200 words."""


def run_phase2(
    phase1_results: List[dict],
    user_prompt: str,
    model: str,
    file_summaries: List[dict],
    tier: str = 'mid',
) -> str:
    slim_phase1 = []
    for r in phase1_results:
        if not isinstance(r, dict):
            continue
        slim_phase1.append({
            'filename': r.get('filename'),
            'data_overview': r.get('data_overview'),
            'key_observations': r.get('key_observations', [])[:5],
            'patterns_detected': r.get('patterns_detected', [])[:5],
            'archaeological_relevance': r.get('archaeological_relevance'),
            'confidence': r.get('confidence'),
            'limitations': r.get('limitations', [])[:3],
        })
    results_str = json.dumps(slim_phase1, indent=2, default=str)

    slim_summaries = [
        {k: v for k, v in fs.items()
         if k not in ('analytics', 'sample_rows', '_thumbnail_b64',
                      'text_content', 'vision_description')}
        for fs in file_summaries if isinstance(fs, dict)
    ]
    file_list = ', '.join(s['filename'] for s in file_summaries)

    totals_lines = []
    for fs in file_summaries:
        if not isinstance(fs, dict):
            continue
        ct = fs.get('column_totals', {})
        if ct:
            items = list(ct.items())[:15]
            totals_lines.append(
                f'  {fs["filename"]}:\n' +
                '\n'.join(f'    SUM of {col} = {val}' for col, val in items)
            )
    totals_block = (
        'PRE-COMPUTED COLUMN TOTALS PER FILE (ground truth, use verbatim):\n'
        + '\n'.join(totals_lines) + '\n\n'
    ) if totals_lines else ''

    file_blocks = []
    for fs in slim_summaries:
        fname = fs.get('filename', 'unknown')
        num = fs.get('numeric_summary', {})
        cat = fs.get('categorical_summary', {})
        block = f'=== FILE: {fname} ===\n'
        if num:
            block += 'Numeric summary (SUM = column total, all values from THIS file only):\n'
            for col, stats in num.items():
                stats_str = ', '.join(f'{k}={v}' for k, v in stats.items() if v is not None)
                block += f'  [{fname}] {col}: {stats_str}\n'
        if cat:
            block += 'Categorical summary:\n'
            for col, vc in list(cat.items())[:5]:
                block += f'  {col}: {dict(list(vc.items())[:5])}\n'
        file_blocks.append(block)
    file_summaries_str = '\n'.join(file_blocks)

    user_message = (
        f"RESEARCHER QUESTION: {user_prompt}\n\n"
        f"FILES ANALYZED: {file_list}\n\n"
        + totals_block
        + f"PER-FILE STATISTICS (each section is for ONE file only):\n{file_summaries_str}\n\n"
        + f"PHASE 1 INDIVIDUAL ANALYSES:\n{results_str}\n\n"
        "Synthesize the above into a unified archaeological interpretation "
        "that directly addresses the researcher's question."
    )

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': PHASE2_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message},
        ],
        options={
            'temperature': 0.15,
            'repeat_penalty': 1.15,
            'num_predict': get_num_predict(tier, 'phase2'),
        },
    )
    return response['message']['content']

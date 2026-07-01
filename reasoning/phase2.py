"""
Phase 2: Cross-file synthesis and narrative generation.

Receives all Phase 1 per-file results and synthesizes a unified
archaeological interpretation addressing the researcher's query.

Model is instructed to:
  - Cite which files support each claim
  - Surface cross-dataset patterns and contradictions
  - Distinguish evidence from inference
  - Flag uncertainty honestly
"""
import json
from typing import List
import ollama

PHASE2_SYSTEM_PROMPT = """You are an expert archaeological analyst producing a formal interpretive report.
You have received individual analyses of one or more datasets (Phase 1 results).

RULES:
1. Ground every claim in the provided data. Cite which file(s) support each claim.
2. Identify cross-dataset patterns, correlations, and contradictions explicitly.
3. Clearly distinguish evidence (what the data shows) from inference (what you conclude).
4. Flag where data is insufficient or where alternative interpretations are equally valid.
5. Use appropriate epistemic caution. Do not introduce information not in
   the provided analyses.
6. Only reference variables present in the provided analyses. Do not introduce
   external information.
7. Use PRE-COMPUTED COLUMN TOTALS PER FILE verbatim for cross-file comparisons.
   Do not estimate from means.
   
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
) -> str:
    """Synthesize all Phase 1 results into a Markdown narrative."""
    # Trim Phase 1 results to key fields only
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
    # Trim file summaries: drop analytics blob and raw content
    slim_summaries = [
        {k: v for k, v in fs.items()
         if k not in ('analytics', 'sample_rows', '_thumbnail_b64',
                      'text_content', 'vision_description')}
        for fs in file_summaries if isinstance(fs, dict)
    ]
    file_list   = ', '.join(s['filename'] for s in file_summaries)

    # Pre-compute cross-file column totals
    totals_lines = []
    for fs in file_summaries:
        if not isinstance(fs, dict):
            continue
        ct = fs.get('column_totals', {})
        if ct:
            # Only include numeric columns, cap at 15 per file
            items = list(ct.items())[:15]
            totals_lines.append(
                f'  {fs["filename"]}:\n' +
                '\n'.join(f'    SUM of {col} = {val}' for col, val in items)
            )
    totals_block = (
        'PRE-COMPUTED COLUMN TOTALS PER FILE (ground truth, use verbatim), NOT row counts):\n'
        + '\n'.join(totals_lines) + '\n\n'
    ) if totals_lines else ''

    print(f"DEBUG totals_block:\n{totals_block}")#DEBUG
    
    user_message = (
        f"RESEARCHER QUESTION: {user_prompt}\n\n"
        f"FILES ANALYZED: {slim_summaries}\n\n"
        + totals_block
        + f"PHASE 1 INDIVIDUAL ANALYSES:\n{results_str}\n\n"
        "Synthesize the above into a unified archaeological interpretation "
        "that directly addresses the researcher's question."
    )

    print(f"DEBUG user_message length: {len(user_message)} chars")#DEBUG
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': PHASE2_SYSTEM_PROMPT},
            {'role': 'user',   'content': user_message},
        ],
        options={'temperature': 0.15},
    )
    return response['message']['content']

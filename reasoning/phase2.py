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
5. Use appropriate epistemic caution. Do not introduce information not in the provided analyses.

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
    results_str = json.dumps(phase1_results, default=str, indent=2)
    file_list   = ', '.join(s['filename'] for s in file_summaries)

    user_message = (
        f"RESEARCHER QUESTION: {user_prompt}\n\n"
        f"FILES ANALYZED: {file_list}\n\n"
        f"PHASE 1 INDIVIDUAL ANALYSES:\n{results_str}\n\n"
        "Synthesize the above into a unified archaeological interpretation "
        "that directly addresses the researcher's question."
    )

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': PHASE2_SYSTEM_PROMPT},
            {'role': 'user',   'content': user_message},
        ],
        options={'temperature': 0.2},
    )
    return response['message']['content']
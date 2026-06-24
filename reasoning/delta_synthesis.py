"""
Delta synthesis for resumed sessions with new data.

When a user resumes a session with --resume and adds new files, this module
produces a "Supplementary Findings" section rather than rewriting the
original report. This preserves the audit trail: each session's contribution
is clearly demarcated with timestamps and file attribution.

The supplementary section is:
  - Produced by delta synthesis here
  - Then run through the full two-pass critique + revision loop in main.py
  - Then appended to the original report by output.append_supplementary()

Three explicit sub-questions the synthesis must answer:
  1. What does the new data CORROBORATE from the original conclusions?
  2. What does it CONTRADICT or COMPLICATE?
  3. What does it EXTEND into new interpretive territory?
"""
import json
import ollama
from typing import List

DELTA_SYSTEM_PROMPT = """You are an expert archaeological analyst updating an interpretation
with newly added data.

You have the original report and Phase 1 analyses of the new files only.

YOUR TASK: Write a "Supplementary Findings" section (NOT a full new report).

Structure your response as Markdown with these sections:

## Supplementary Findings
### New Data Overview
### Corroboration
  What new data confirms or strengthens prior conclusions. Cite files.
### Contradictions and Complications
  Where new data conflicts with or complicates prior conclusions. Be direct.
### New Interpretive Directions
  What the new data reveals that was not addressable before.
### Updated Confidence Assessment
  Revised confidence level for key claims, with rationale.
### Integration with Prior Analysis
  1-2 paragraph synthesis of how the combined dataset changes the overall picture.

Target length: 400-700 words.
Do not reproduce or restate the original report at length.
Do not present this as a standalone document. It is an appendix to the original."""


def run_delta_synthesis(
    session: dict,
    new_phase1_results: List[dict],
    new_file_summaries: List[dict],
    new_files: List[str],
    model: str,
) -> str:
    """
    Produce a Supplementary Findings section integrating new data.

    Args:
        session:             Loaded session object (prior context + narrative)
        new_phase1_results:  Phase 1 results for the new files only
        new_file_summaries:  Preprocessed summaries of the new files only
        new_files:           New filenames
        model:               Ollama model name

    Returns:
        Markdown string starting with '## Supplementary Findings'
    """
    new_p1_str    = json.dumps(new_phase1_results, indent=2, default=str)
    prior_files   = ', '.join(session.get('files_analyzed', []))
    new_files_str = ', '.join(new_files)

    user_message = (
        f'ORIGINAL RESEARCH QUESTION: {session["original_prompt"]}\n\n'
        f'PRIOR FILES: {prior_files}\n'
        f'NEW FILES ADDED: {new_files_str}\n\n'
        f'ORIGINAL INTERPRETATION REPORT:\n{session["final_narrative"]}\n\n'
        f'PHASE 1 ANALYSES OF NEW FILES:\n{new_p1_str}\n\n'
        'Write the Supplementary Findings section.'
    )

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': DELTA_SYSTEM_PROMPT},
            {'role': 'user',   'content': user_message},
        ],
        options={'temperature': 0.2},
    )
    return response['message']['content']
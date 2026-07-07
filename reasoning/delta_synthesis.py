"""
Delta synthesis for resumed sessions with new data.
"""
import json
import ollama
from typing import List
from config import get_num_predict

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


def _has_repetition(text: str, threshold: int = 3) -> bool:
    sentences = [s.strip() for s in text.split('. ') if len(s.strip()) > 40]
    seen: dict[str, int] = {}
    for s in sentences:
        seen[s] = seen.get(s, 0) + 1
        if seen[s] >= threshold:
            return True
    return False


def _truncate_at_repetition(text: str) -> str:
    lines = text.split('\n')
    seen_lines: dict[str, int] = {}
    clean: list[str] = []
    for line in lines:
        key = line.strip()
        if key:
            seen_lines[key] = seen_lines.get(key, 0) + 1
            if seen_lines[key] > 2:
                break
        clean.append(line)
    truncated = '\n'.join(clean).strip()
    truncated += (
        '\n\n> **Note:** Output was truncated due to detected repetition. '
        'Consider switching to a higher-tier model for resumed sessions with large context.\n'
    )
    return truncated


def run_delta_synthesis(
    session: dict,
    new_phase1_results: List[dict],
    new_file_summaries: List[dict],
    new_files: List[str],
    model: str,
    tier: str = 'mid',
) -> str:
    new_p1_str = json.dumps(new_phase1_results, indent=2, default=str)
    prior_files = ', '.join(session.get('files_analyzed', []))
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
            {'role': 'user', 'content': user_message},
        ],
        options={
            'temperature': 0.2,
            'repeat_penalty': 1.15,
            'num_predict': get_num_predict(tier, 'delta'),
        },
    )

    content = response['message']['content']

    if _has_repetition(content):
        content = _truncate_at_repetition(content)

    return content

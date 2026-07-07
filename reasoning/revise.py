"""
Phase 3b: Revision model.
"""
import json
import ollama
from config import get_num_predict

REVISION_SYSTEM_PROMPT = """You are an expert archaeological writer revising an interpretation report
based on a peer reviewer's structured critique.

RULES:
1. Address every HIGH-severity flag. These are mandatory changes.
2. Address every MEDIUM-severity flag unless you document a reason not to.
3. Consider LOW-severity flags at your discretion.
4. Do NOT change content that was not flagged. Preserve well-supported claims.
5. Do NOT add new claims that are not supported by the data.
6. When a flagged claim contains a useful hypothesis, DO NOT remove it.
   Reframe with explicit hedging: "This may suggest...", "One possible interpretation is...",
   "This pattern warrants further investigation to determine whether..."
7. Maintain the same Markdown section structure as the original.

After the revised report, append:
## Revision Log
- **[Flag type] [Severity]**: Original claim -> What was changed and why.
- **[Flag type] [Severity] -- No change**: Reason original wording was retained.

Be precise. The revision log is part of the scientific record."""


def run_revision(
    original_narrative: str,
    critique_result: dict,
    user_prompt: str,
    model: str,
    file_summaries: list,
    tier: str = 'mid',
) -> str:
    flags = critique_result.get('flagged_claims', [])
    if not flags or critique_result.get('overall_assessment') in ('error', 'parse_error'):
        return original_narrative + '\n\n## Revision Log\n\nNo changes required. Critique found no significant issues.\n'

    file_list = ', '.join(s['filename'] for s in file_summaries)
    critique_str = json.dumps(critique_result, indent=2, default=str)

    flag_summary = '\n'.join(
        f"- [{f.get('severity','?').upper()}] {f.get('problem_type','?')}: "
        f"\"{f.get('claim','?')[:120]}\" -> {f.get('revision_suggestion','?')}"
        for f in flags
    )

    user_message = (
        f'RESEARCHER QUESTION: {user_prompt}\n\n'
        f'FILES ANALYZED: {file_list}\n\n'
        f'CRITIC ASSESSMENT: {critique_result.get("overall_assessment", "").upper()}\n'
        f'CRITIQUE RATIONALE: {critique_result.get("overall_rationale", "")}\n\n'
        f'FLAGS TO ADDRESS:\n{flag_summary}\n\n'
        f'FULL CRITIQUE JSON:\n{critique_str}\n\n'
        f'ORIGINAL REPORT:\n{original_narrative}\n\n'
        'Revise the report to address the flags. Append a Revision Log as instructed.'
    )

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': REVISION_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message},
        ],
        options={
            'temperature': 0.15,
            'repeat_penalty': 1.15,
            'num_predict': get_num_predict(tier, 'revise'),
        },
    )
    return response['message']['content']

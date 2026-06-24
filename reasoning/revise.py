"""
Phase 3b: Revision model.

Receives the original Phase 2 narrative and the critic's structured JSON
output. Produces a revised Markdown narrative that addresses every flagged
claim with a documented change.

The revision model is instructed to:
  1. Address every high-severity flag (required)
  2. Address every medium-severity flag unless there is a documented reason not to
  3. Consider but optionally address low-severity flags
  4. Preserve all well-supported content unchanged
  5. Append a Revision Log section listing exactly what changed and why

The revision log gives the human analyst a clear audit trail: they can
compare original vs. revised and decide whether the critic's concerns were
addressed appropriately.
"""
import json
import ollama

REVISION_SYSTEM_PROMPT = """You are an expert archaeological writer revising an interpretation report
based on a peer reviewer's structured critique.

RULES:
1. Address every HIGH-severity flag. These are mandatory changes.
2. Address every MEDIUM-severity flag unless you document a reason not to.
3. Consider LOW-severity flags at your discretion.
4. Do NOT change content that was not flagged. Preserve well-supported claims.
5. Do NOT add new claims that are not supported by the data.
6. Maintain the same Markdown section structure as the original.

After the revised report, append a section titled:

## Revision Log

List each change made in this format:
- **[Flag type] [Severity]**: Original claim -> What was changed and why.

If a flagged claim was reviewed but NOT changed, document the reason:
- **[Flag type] [Severity] -- No change**: Reason the original wording was retained.

Be precise and honest. The revision log is part of the scientific record."""


def run_revision(
    original_narrative: str,
    critique_result: dict,
    user_prompt: str,
    model: str,
    file_summaries: list,
) -> str:
    """
    Revise the Phase 2 narrative to address the critic's flags.

    Args:
        original_narrative: Phase 2 Markdown narrative
        critique_result:    Structured critique dict from critic.py
        user_prompt:        Original researcher query
        model:              Ollama model name
        file_summaries:     Preprocessed file summaries (for reference)

    Returns:
        Revised Markdown string with appended Revision Log.
    """
    # If critique had no flags or errored, return original unchanged
    flags = critique_result.get('flagged_claims', [])
    if not flags or critique_result.get('overall_assessment') in ('error', 'parse_error'):
        return original_narrative + '\n\n## Revision Log\n\nNo changes required. Critique found no significant issues.\n'

    file_list    = ', '.join(s['filename'] for s in file_summaries)
    critique_str = json.dumps(critique_result, indent=2, default=str)

    # Summarize flags for the prompt to keep context tight
    flag_summary = '\n'.join(
        f"- [{f.get('severity','?').upper()}] {f.get('problem_type','?')}: "
        f"\"{f.get('claim','?')[:120]}\" "
        f"-> {f.get('revision_suggestion','?')}"
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
            {'role': 'user',   'content': user_message},
        ],
        options={'temperature': 0.15},
    )
    return response['message']['content']
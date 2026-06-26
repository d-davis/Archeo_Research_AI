"""
Phase 3a: Critique model.

The same Ollama model used for Phase 1/2 reasoning runs here with a
completely separate context and a dedicated critic system prompt.

The critic evaluates the Phase 2 narrative against a structured rubric:
  1. Claim support       -- is each interpretive claim grounded in the data?
  2. Confidence fit      -- is stated certainty appropriate to the evidence?
  3. Alternative hypotheses -- were plausible alternative interpretations considered?
  4. Overreach detection -- does the narrative go beyond what the data supports?
  5. Evidence traceability -- are file sources cited for key claims?

Output is structured JSON. Each flagged claim includes:
  - the claim text
  - the problem type (unsupported / overconfident / missing_alternative / untraceable)
  - a recommended revision action
  - a severity level (low / medium / high)

The critic does NOT rewrite anything. Rewriting is the sole job of revise.py.
"""
import json
import ollama
from typing import List

CRITIC_SYSTEM_PROMPT = """You are a rigorous peer reviewer evaluating an archaeological interpretation report.
Your sole job is to critique it. Do NOT rewrite or improve it.

Evaluate the report against this rubric:

1. CLAIM SUPPORT: Is each interpretive claim grounded in the data provided?
   Flag any claim that goes beyond what the cited datasets contain.

2. CONFIDENCE FIT: Is the stated level of certainty appropriate?
   Flag claims that are overconfident given data quality, sample size, or gaps.

3. ALTERNATIVE HYPOTHESES: Were plausible competing interpretations mentioned?
   Flag sections where an alternative explanation is equally valid but ignored.

4. OVERREACH: Does the narrative draw cultural, historical, or causal conclusions
   that the data cannot support alone?

5. EVIDENCE TRACEABILITY: Are key interpretive claims linked to specific files?
   Flag claims where the supporting data source is unclear.

IMPORTANT: Do not flag well-supported, appropriately hedged claims.
Do NOT flag statements that are explicitly framed as hypotheses, speculative
interpretations, or potential research directions. Phrases such as "may suggest",
"could indicate", "one possible interpretation", "warrants further investigation",
"raises the question of", "tentatively suggests", or similar hedging language
indicate the author is appropriately signalling uncertainty. These are analytically
valuable and must be preserved intact.
Only flag statements that make unhedged factual assertions not grounded in the data.
Reserve flags for genuine problems. Aim for 2-6 flags per report.

Respond in valid JSON using this exact schema:
{
  "overall_assessment": "<strong|adequate|weak>",
  "overall_rationale": "<1-2 sentences>",
  "flagged_claims": [
    {
      "claim": "<exact quoted phrase or sentence from the report>",
      "problem_type": "<unsupported|overconfident|missing_alternative|overreach|untraceable>",
      "explanation": "<why this is a problem>",
      "severity": "<low|medium|high>",
      "revision_suggestion": "<specific action the author should take>"
    }
  ],
  "commended_elements": ["<well-done aspect>", ...],
  "missing_sections": ["<important topic not addressed>", ...]
}"""


def run_critique(
    narrative: str,
    phase1_results: List[dict],
    user_prompt: str,
    model: str,
    file_summaries: List[dict],
) -> dict:
    """
    Critique a Phase 2 narrative using the same model with an isolated context.

    Args:
        narrative:      The Phase 2 Markdown narrative to critique
        phase1_results: All Phase 1 per-file JSON results (for context)
        user_prompt:    The original researcher query
        model:          Ollama model name
        file_summaries: Preprocessed file summaries (for context)

    Returns:
        Structured critique dict, or minimal error dict on failure.
    """
    file_list = ', '.join(s['filename'] for s in file_summaries)
    p1_summary = json.dumps(
        [{'filename': r.get('filename'), 'confidence': r.get('confidence')} for r in phase1_results],
        indent=2
    )

    user_message = (
        f'RESEARCHER QUESTION: {user_prompt}\n\n'
        f'FILES ANALYZED: {file_list}\n\n'
        f'PHASE 1 CONFIDENCE SUMMARY:\n{p1_summary}\n\n'
        f'REPORT TO CRITIQUE:\n{narrative}\n\n'
        'Review this report and return your structured critique as JSON.'
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': CRITIC_SYSTEM_PROMPT},
                {'role': 'user',   'content': user_message},
            ],
            format='json',
            options={'temperature': 0.1},
        )
        result = json.loads(response['message']['content'])
        result['_phase']  = '3a_critique'
        result['_model']  = model
        return result

    except json.JSONDecodeError as e:
        return {
            '_phase':       '3a_critique',
            '_model':       model,
            '_parse_error': str(e),
            'raw_response': response['message']['content'],
            'flagged_claims':     [],
            'overall_assessment': 'parse_error',
            'overall_rationale':  'Critique model returned malformed JSON.',
            'commended_elements': [],
            'missing_sections':   [],
        }
    except Exception as e:
        return {
            '_phase':             '3a_critique',
            '_model':             model,
            '_error':             str(e),
            'flagged_claims':     [],
            'overall_assessment': 'error',
            'overall_rationale':  f'Critique failed: {e}',
            'commended_elements': [],
            'missing_sections':   [],
        }

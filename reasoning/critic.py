"""
Phase 3a: Critique model.
"""
import json
import ollama
from typing import List
from config import get_num_predict

CRITIC_SYSTEM_PROMPT = """You are a rigorous peer reviewer evaluating an archaeological interpretation report.
Your sole job is to critique it. Do NOT rewrite or improve it.

Evaluate the report against this rubric:

1. CLAIM SUPPORT: Is each interpretive claim grounded in the data provided?
2. CONFIDENCE FIT: Is the stated level of certainty appropriate?
3. ALTERNATIVE HYPOTHESES: Were plausible competing interpretations mentioned?
4. OVERREACH: Does the narrative draw conclusions the data cannot support alone?
5. EVIDENCE TRACEABILITY: Are key claims linked to specific files?

IMPORTANT: Do not flag well-supported, appropriately hedged claims.
Do NOT flag statements framed as hypotheses. Phrases such as "may suggest",
"could indicate", "one possible interpretation", "warrants further investigation",
"tentatively suggests" indicate appropriate uncertainty. Preserve these intact.

Only flag unhedged factual assertions not grounded in the data.
Aim for 2-6 flags per report.

Respond in valid JSON:
{
 "overall_assessment": "<strong|adequate|weak>",
 "overall_rationale": "<1-2 sentences>",
 "flagged_claims": [
   {
     "claim": "<exact quoted phrase>",
     "problem_type": "<unsupported|overconfident|missing_alternative|overreach|untraceable>",
     "explanation": "<why this is a problem>",
     "severity": "<low|medium|high>",
     "revision_suggestion": "<specific action>"
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
    tier: str = 'mid',
) -> dict:
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
                {'role': 'user', 'content': user_message},
            ],
            format='json',
            options={
                'temperature': 0.1,
                'repeat_penalty': 1.15,
                'num_predict': get_num_predict(tier, 'critique'),
            },
        )
        result = json.loads(response['message']['content'])
        result['_phase'] = '3a_critique'
        result['_model'] = model
        return result
    except json.JSONDecodeError as e:
        return {
            '_phase': '3a_critique', '_model': model,
            '_parse_error': str(e),
            'raw_response': response['message']['content'],
            'flagged_claims': [], 'overall_assessment': 'parse_error',
            'overall_rationale': 'Critique model returned malformed JSON.',
            'commended_elements': [], 'missing_sections': [],
        }
    except Exception as e:
        return {
            '_phase': '3a_critique', '_model': model, '_error': str(e),
            'flagged_claims': [], 'overall_assessment': 'error',
            'overall_rationale': f'Critique failed: {e}',
            'commended_elements': [], 'missing_sections': [],
        }

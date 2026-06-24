"""
Follow-up question answering.

Injects the full session context (preprocessed file summaries, Phase 1
results, final narrative, conversation history) alongside the new question
and returns a focused, grounded answer.

Temperature is slightly higher than analysis passes (0.3) to allow more
natural conversational responses while remaining data-grounded.

Note on context size: follow-up prompts are large because they carry the
full session context. On CPU/low-GPU tiers with small context windows,
only the final narrative and conversation history are injected if the full
context would exceed the model's limit. The context_assembly module's
token estimator handles this.
"""
import json
import ollama
from session import get_history_text

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
6. Do not reproduce large portions of the original report unnecessarily."""


def run_followup(question: str, session: dict) -> str:
    """
    Answer a follow-up question using full session context.

    Args:
        question: The researcher's follow-up question
        session:  The current session object

    Returns:
        Markdown-formatted answer string.
    """
    file_summaries_str = json.dumps(
        session.get('preprocessed_summaries', []),
        indent=2, default=str
    )
    phase1_str = json.dumps(
        session.get('phase1_results', []),
        indent=2, default=str
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
        'Answer the question concisely and accurately.'
    )

    response = ollama.chat(
        model=session['model'],
        messages=[
            {'role': 'system', 'content': FOLLOWUP_SYSTEM_PROMPT},
            {'role': 'user',   'content': user_message},
        ],
        options={'temperature': 0.3},
    )
    return response['message']['content']
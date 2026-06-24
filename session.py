"""
Session management for the Archaeological AI Interpreter.

Handles:
  - Creating new session objects
  - Saving sessions to ./sessions/<name>.json
  - Loading sessions from file
  - Appending conversation turns
  - Rolling summary (kicks in at ROLLING_SUMMARY_THRESHOLD turns)

Session file schema:
{
  session_id, session_name, created_at, updated_at,
  original_prompt, files_analyzed, model,
  preprocessed_summaries, phase1_results,
  final_narrative, critique_result,
  conversation_history: [{role, content, timestamp}],
  conversation_summary,   # populated after rolling summary
  turn_count,
  supplementary_sections: [{added_files, timestamp, content}]
}
"""
import json
import ollama
from datetime import datetime
from pathlib import Path
from typing import List, Optional

SESSIONS_DIR = Path('./sessions')
ROLLING_SUMMARY_THRESHOLD = 8   # apply rolling summary when turn_count hits this
SUMMARY_BATCH             = 4   # number of oldest exchanges to summarize at once


def new_session(
    original_prompt: str,
    files_analyzed: List[str],
    model: str,
    preprocessed_summaries: List[dict],
    phase1_results: List[dict],
    final_narrative: str,
    critique_result: Optional[dict] = None,
    session_name: Optional[str] = None,
) -> dict:
    """Create and return a new session object."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return {
        'session_id':             ts,
        'session_name':           session_name or f'session_{ts}',
        'created_at':             datetime.now().isoformat(),
        'updated_at':             datetime.now().isoformat(),
        'original_prompt':        original_prompt,
        'files_analyzed':         files_analyzed,
        'model':                  model,
        'preprocessed_summaries': preprocessed_summaries,
        'phase1_results':         phase1_results,
        'final_narrative':        final_narrative,
        'critique_result':        critique_result,
        'conversation_history':   [],
        'conversation_summary':   None,
        'turn_count':             0,
        'supplementary_sections': [],
    }


def save_session(session: dict, output_dir: Optional[Path] = None) -> Path:
    """Save session JSON to disk. Returns the saved path."""
    out_dir = output_dir or SESSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    session['updated_at'] = datetime.now().isoformat()
    path = out_dir / f"{session['session_name']}.json"
    path.write_text(json.dumps(session, indent=2, default=str), encoding='utf-8')
    return path


def load_session(session_path: str) -> dict:
    """Load a session from a JSON file. Raises FileNotFoundError if missing."""
    path = Path(session_path)
    if not path.exists():
        raise FileNotFoundError(f'Session file not found: {session_path}')
    return json.loads(path.read_text(encoding='utf-8'))


def add_turn(session: dict, role: str, content: str) -> None:
    """
    Append a conversation turn.
    role: 'user' or 'assistant'
    turn_count increments only on 'assistant' turns (one per exchange).
    """
    session['conversation_history'].append({
        'role':      role,
        'content':   content,
        'timestamp': datetime.now().isoformat(),
    })
    if role == 'assistant':
        session['turn_count'] += 1


def get_history_text(session: dict) -> str:
    """
    Return conversation history as a formatted string for context injection.
    Prepends rolling summary if present.
    """
    parts = []
    if session.get('conversation_summary'):
        parts.append(
            f'[Earlier conversation summary]\n{session["conversation_summary"]}'
        )
    for turn in session['conversation_history']:
        role = turn['role'].upper()
        parts.append(f'{role}: {turn["content"]}')
    return '\n\n'.join(parts)


def maybe_apply_rolling_summary(session: dict, model: str) -> bool:
    """
    When turn_count >= ROLLING_SUMMARY_THRESHOLD, summarize the oldest
    SUMMARY_BATCH exchanges (user + assistant pairs) into a compact paragraph
    and remove them from the live history.

    The summary is appended to session['conversation_summary'].
    Returns True if a summary was applied, False otherwise.
    """
    if session['turn_count'] < ROLLING_SUMMARY_THRESHOLD:
        return False

    history = session['conversation_history']
    batch_size = SUMMARY_BATCH * 2  # user + assistant turns per exchange
    if len(history) < batch_size:
        return False

    to_summarize = history[:batch_size]
    remaining    = history[batch_size:]

    convo_text = '\n'.join(
        f"{t['role'].upper()}: {t['content']}" for t in to_summarize
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are summarizing a conversation between a researcher and '
                        'an archaeological AI analyst. Produce a 2-4 sentence summary '
                        'capturing the key questions asked and conclusions reached. '
                        'Be factual and concise.'
                    )
                },
                {'role': 'user', 'content': f'Summarize:\n\n{convo_text}'},
            ],
            options={'temperature': 0.1},
        )
        new_summary = resp['message']['content']
        existing = session.get('conversation_summary') or ''
        session['conversation_summary'] = (
            f'{existing}\n\n{new_summary}'.strip()
        )
        session['conversation_history'] = remaining
        return True
    except Exception:
        return False  # fail silently, preserve full history
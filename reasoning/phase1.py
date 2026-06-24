"""
Phase 1: Per-file archaeological analysis.
Updated system prompt to handle PDF text content.
Replaces reasoning/phase1.py.
"""
import json
import ollama

PHASE1_SYSTEM_PROMPT = """You are an expert archaeological data analyst.
You will receive a structured summary of a single dataset and a researcher's question.

YOUR TASK: Analyze THIS FILE ONLY. Do not speculate about data not shown.
Be precise about what is observed in the data vs. what you infer from it.

For PDF documents, the full extracted text is provided in 'text_content'.
The text includes page labels ([Page N]) so you can cite page numbers.
Identify and extract tabular data, lists, measurements, and structured content
directly from the text. Do not ignore tables just because they appear as plain
text -- archaeological reports always label tables ("Table 1:", "Table 2:" etc.).

Respond in valid JSON using this exact schema:
{
  "filename": "<filename>",
  "data_overview": "<1-2 sentences describing what this dataset represents>",
  "key_observations": ["<data-supported observation>", ...],
  "patterns_detected": ["<distributions, clusters, anomalies, or trends>", ...],
  "tables_identified": ["<brief description of each table found, e.g. Table 1: artifact counts by type>"],
  "archaeological_relevance": "<how this data relates to the researcher's question>",
  "limitations": ["<data quality issues, gaps, or caveats>", ...],
  "confidence": "<low|medium|high>",
  "suggested_cross_references": ["<data types that would strengthen interpretation>", ...]
}

Be concise. If the data is sparse or ambiguous, say so explicitly."""


def run_phase1(file_summary: dict, user_prompt: str, model: str) -> dict:
    """
    Analyze a single preprocessed file against the user's prompt.
    Returns a JSON observation dict, or an error dict on failure.
    """
    # For PDF summaries, pass text_content directly in the serialization
    summary_str = json.dumps(file_summary, default=str, indent=2)
    user_message = (
        f'RESEARCHER QUESTION: {user_prompt}\n\n'
        f'FILE SUMMARY:\n{summary_str}\n\n'
        'Analyze this file in the context of the researcher\'s question.'
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': PHASE1_SYSTEM_PROMPT},
                {'role': 'user',   'content': user_message},
            ],
            format='json',
            options={'temperature': 0.1},
        )
        result = json.loads(response['message']['content'])
        result['_phase'] = 1
        result['_model'] = model
        return result

    except json.JSONDecodeError as e:
        return {
            'filename':     file_summary.get('filename', 'unknown'),
            '_phase':       1,
            '_model':       model,
            '_parse_error': str(e),
            'raw_response': response['message']['content'],
        }
    except Exception as e:
        return {
            'filename': file_summary.get('filename', 'unknown'),
            '_phase':   1,
            '_model':   model,
            '_error':   str(e),
        }

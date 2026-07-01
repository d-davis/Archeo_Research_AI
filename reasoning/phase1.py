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
Be precise about what is observed vs. what you infer.
STRICT DATA BOUNDARY: Only reference variables explicitly present in the FILE SUMMARY.

For PDF documents: extract tabular data, lists, and measurements from 'text_content'.
Cite page numbers using [Page N] labels. Tables are labelled "Table 1:", "Table 2:", etc.

COUNTING AND AGGREGATION:
Each numeric column in 'numeric_summary' contains a 'sum' field with the exact
column total. Use this verbatim for totals — do NOT use 'mean', 'count', estimate, or
calculate. Example: "Total_Ceramics sum = 880.0 (from numeric_summary)."

Respond in valid JSON:
{
  "filename": "<filename>",
  "data_overview": "<1-2 sentences>",
  "key_observations": ["<data-supported observation>", ...],
  "patterns_detected": ["<distributions, clusters, anomalies, trends>", ...],
  "tables_identified": ["<brief description>"],
  "archaeological_relevance": "<how this relates to the researcher's question>",
  "limitations": ["<data quality issues or gaps>", ...],
  "confidence": "<low|medium|high>",
  "suggested_cross_references": ["<data types that would strengthen interpretation>", ...]
}

Be concise. If data is sparse or ambiguous, say so explicitly."""

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

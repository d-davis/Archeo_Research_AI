"""
Context assembly and token budget management.
Combines preprocessed file summaries + user prompt within a token budget.
"""
import json
from typing import List

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4  # rough fallback


def _serialize(summary: dict) -> str:
    return json.dumps(summary, default=str, ensure_ascii=False, indent=2)


def assemble_context(
    preprocessed_files: List[dict],
    user_prompt: str,
    max_tokens: int = 8192,
    reserve_for_output: int = 2048,
) -> dict:
    """
    Returns a context package dict:
      prompt            user query
      files             list of (possibly trimmed) summaries
      estimated_tokens  total token estimate
      trimmed           whether any summaries were truncated
      file_count        number of files included
    """
    budget = max_tokens - reserve_for_output - 500  # 500 for system prompt overhead
    used = count_tokens(user_prompt)
    assembled_files = []
    trimmed = False

    for summary in preprocessed_files:
        raw_tokens = count_tokens(_serialize(summary))
        if used + raw_tokens <= budget:
            assembled_files.append(summary)
            used += raw_tokens
        else:
            remaining = budget - used
            if remaining > 100:
                trimmed_s = _trim_summary(summary, remaining)
                assembled_files.append(trimmed_s)
                used += count_tokens(_serialize(trimmed_s))
            trimmed = True

    return {
        'prompt':           user_prompt,
        'files':            assembled_files,
        'estimated_tokens': used,
        'trimmed':          trimmed,
        'file_count':       len(assembled_files),
    }


def _trim_summary(summary: dict, token_budget: int) -> dict:
    """
    Drop lower-priority fields until the summary fits within token_budget.
    Drop order (lowest priority first):
      categorical_summary -> sample_rows -> notes -> numeric_summary
    Core fields (filename, shape, data_type, columns) are never dropped.
    """
    drop_order = ['analytics', 'categorical_summary', 'sample_rows', 'notes', 'numeric_summary']
    result = dict(summary)
    result['_trimmed'] = True
    for field in drop_order:
        if count_tokens(_serialize(result)) <= token_budget:
            break
        result.pop(field, None)
    return result

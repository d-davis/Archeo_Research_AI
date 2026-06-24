"""
Output formatting and file export -- Phase 3.5 update.

Phase 3.5 additions:
  - append_supplementary(): produces an updated report file with the
    Supplementary Findings section appended to the original narrative.
    Always creates a NEW file (timestamped); never modifies the original.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def save_output(
    narrative: str,
    user_prompt: str,
    files: List[str],
    model: str,
    original_narrative: Optional[str] = None,
    critique_result: Optional[dict] = None,
    phase1_results: Optional[List[dict]] = None,
    output_path: Optional[str] = None,
    fmt: str = 'markdown',
) -> Path:
    """Save the final interpretation to disk."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_path is None:
        out_dir = Path('./output')
        out_dir.mkdir(exist_ok=True)
        ext = '.md' if fmt == 'markdown' else '.txt'
        output_path = out_dir / f'interpretation_{timestamp}{ext}'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    critique_note = ''
    if critique_result:
        n_flags  = len(critique_result.get('flagged_claims', []))
        assessed = critique_result.get('overall_assessment', 'unknown')
        critique_note = f'\n**Critique:** {assessed} ({n_flags} flag(s) addressed)'

    header = (
        f'# Archaeological Interpretation Report\n\n'
        f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        f'**Model:** {model}\n'
        f'**Files analyzed:** {", ".join(files)}\n'
        f'**Query:** {user_prompt}'
        f'{critique_note}\n\n'
        f'---\n\n'
    )
    content = header + narrative

    if original_narrative:
        content += '\n\n---\n\n## Appendix A: Pre-Revision Draft\n\n'
        content += original_narrative

    if critique_result:
        content += '\n\n---\n\n## Appendix B: Critique Report\n\n```json\n'
        content += json.dumps(critique_result, indent=2, default=str)
        content += '\n```\n'

    if phase1_results:
        content += '\n\n---\n\n## Appendix C: Phase 1 Per-File Analyses\n\n'
        for result in phase1_results:
            fname = result.get('filename', 'unknown')
            content += f'### {fname}\n\n```json\n'
            content += json.dumps(result, indent=2, default=str)
            content += '\n```\n\n'

    output_path.write_text(content, encoding='utf-8')
    return output_path


def append_supplementary(
    session: dict,
    supplementary: str,
    new_files: List[str],
    output_path: Optional[str] = None,
    fmt: str = 'markdown',
) -> Path:
    """
    Create an updated report file with the Supplementary Findings appended.

    Always creates a new file -- never modifies the original report.
    The original narrative is reproduced in full so the updated file is
    self-contained.

    Args:
        session:        Current session object (contains original narrative)
        supplementary:  The revised supplementary findings section (Markdown)
        new_files:      List of newly added filenames
        output_path:    Override output path
        fmt:            'markdown' or 'txt'

    Returns:
        Path to the new updated report file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_path is None:
        out_dir = Path('./output')
        out_dir.mkdir(exist_ok=True)
        ext = '.md' if fmt == 'markdown' else '.txt'
        name = f"{session['session_name']}_updated_{timestamp}{ext}"
        output_path = out_dir / name
    output_path = Path(output_path)

    content = (
        f'# Archaeological Interpretation Report (Updated)\n\n'
        f'**Original session:** {session["session_name"]}\n'
        f'**Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        f'**All files:** {", ".join(session["files_analyzed"])}\n'
        f'**New files added this session:** {", ".join(new_files)}\n\n'
        f'---\n\n'
    )
    content += session['final_narrative']
    content += '\n\n---\n\n'
    content += supplementary

    # If there were prior supplementary sections, include them too
    for i, supp in enumerate(session.get('supplementary_sections', []), 1):
        if supp.get('content') and supp['content'] != supplementary:
            added = ', '.join(supp.get('added_files', []))
            content += (
                f'\n\n---\n\n'
                f'<!-- Session update {i}: added {added} on {supp.get("timestamp","")} -->\n\n'
            )
            content += supp['content']

    output_path.write_text(content, encoding='utf-8')
    return output_path

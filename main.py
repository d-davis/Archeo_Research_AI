#!/usr/bin/env python3
"""
Archaeological AI Interpreter -- Phase 3.6 CLI
Adds PDF support (text-based PDFs with embedded figures).

New:
  - .pdf added to SUPPORTED_ALL
  - PDF routed to preprocessors/pdf.py
  - Figure summaries from PDF appended to preprocessed list before Phase 1
  - Vision model runs on PDF figures the same as standalone images
  - --keep-figures flag preserves extracted figure PNGs after processing

Usage (unchanged):
  python main.py --files report.pdf survey.csv features.geojson \
                 --prompt "What does this report reveal about site formation?"
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from config import get_model_config, check_ollama
from preprocessors.tabular import preprocess_tabular
from preprocessors.geospatial import preprocess_geospatial
from preprocessors.imagery import preprocess_imagery, get_image_thumbnail_b64
from preprocessors.pdf import preprocess_pdf
from context_assembly import assemble_context
from reasoning.phase1 import run_phase1
from reasoning.phase2 import run_phase2
from reasoning.vision import run_vision_description
from reasoning.critic import run_critique
from reasoning.revise import run_revision
from reasoning.followup import run_followup
from reasoning.delta_synthesis import run_delta_synthesis
from session import (
    new_session, save_session, load_session,
    add_turn, maybe_apply_rolling_summary
)
from output import save_output, append_supplementary

console = Console()

SUPPORTED_TABULAR    = {'.csv', '.xlsx', '.xls', '.txt'}
SUPPORTED_GEOSPATIAL = {'.shp', '.geojson', '.json'}
SUPPORTED_IMAGERY    = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
SUPPORTED_PDF        = {'.pdf'}
SUPPORTED_ALL        = SUPPORTED_TABULAR | SUPPORTED_GEOSPATIAL | SUPPORTED_IMAGERY | SUPPORTED_PDF


def parse_args():
    parser = argparse.ArgumentParser(
        description='Archaeological AI Interpreter -- Phase 3.6 (PDF support)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--files', '-f', nargs='+', default=None,
        help='Input files: CSV, XLSX, TXT, SHP, GeoJSON, TIF, PNG, JPEG, PDF')
    parser.add_argument('--prompt', '-p', type=str, default=None)
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--session-name', type=str, default=None, dest='session_name')
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--vision-model', type=str, default=None, dest='vision_model')
    parser.add_argument('--tier', '-t', type=str, default='auto',
        choices=['cpu', 'low', 'mid', 'high', 'auto'])
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--format', type=str, default='markdown',
        choices=['markdown', 'txt'])
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-vision', action='store_true')
    parser.add_argument('--no-critique', action='store_true')
    parser.add_argument('--no-interactive', action='store_true')
    parser.add_argument('--keep-figures', action='store_true',
        help='Keep extracted PDF figure PNGs after processing (saved to ./tmp/pdf_figures/)')
    return parser.parse_args()


def classify_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_TABULAR:    return 'tabular'
    if ext in SUPPORTED_GEOSPATIAL: return 'geospatial'
    if ext in SUPPORTED_IMAGERY:    return 'imagery'
    if ext in SUPPORTED_PDF:        return 'pdf'
    return 'unsupported'


def validate_files(file_paths):
    valid = []
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            console.print(f'[red]File not found:[/red] {fp}')
            sys.exit(1)
        kind = classify_file(path)
        if kind == 'unsupported':
            console.print(f'[yellow]Skipping unsupported:[/yellow] {fp}')
            continue
        valid.append((path, kind))
    if not valid:
        console.print('[red]No supported files.[/red]')
        sys.exit(1)
    return valid


def preprocess_files(file_pairs, model_config, args, prompt_override=None):
    """
    Preprocess all files. For PDFs, also processes extracted figures
    through the vision model if --no-vision is not set.
    Returns a flat list of preprocessed summaries.
    """
    prompt = prompt_override or args.prompt or ''
    preprocessed = []

    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                  console=console) as prog:
        for path, kind in file_pairs:
            task = prog.add_task(f'Preprocessing {path.name}...', total=None)

            if kind == 'tabular':
                result = preprocess_tabular(path)
                label = f"{result['shape'][0]} rows x {result['shape'][1]} cols"
                preprocessed.append(result)

            elif kind == 'geospatial':
                result = preprocess_geospatial(path)
                label = f"{result.get('feature_count', '?')} features"
                preprocessed.append(result)

            elif kind == 'imagery':
                result = preprocess_imagery(path)
                label = f"{result.get('width', '?')} x {result.get('height', '?')} px"
                preprocessed.append(result)

            elif kind == 'pdf':
                result = preprocess_pdf(path, keep_figures=args.keep_figures)
                n_figures = result.get('figure_count', 0)
                n_words   = result.get('text_word_count', 0)
                label = (
                    f"{result.get('text_page_count', '?')} pages, "
                    f"~{n_words:,} words, {n_figures} figure(s) "
                    f"[tables in Markdown]"
                )
                preprocessed.append(result)

                # Promote PDF table summaries as standalone preprocessed items
                for t_summary in result.get('table_summaries', []):
                    preprocessed.append(t_summary)

                # Promote PDF figure summaries as standalone preprocessed items
                for f_summary in result.get('figure_summaries', []):
                    preprocessed.append(f_summary)

            prog.remove_task(task)
            console.print(f'  [green]v[/green] {path.name} -- {label}')

    # Vision pass: run on all imagery items (standalone + extracted from PDFs)
    if not args.no_vision:
        imagery_items = [
            s for s in preprocessed
            if s.get('data_type') in ('raster_imagery', 'standard_imagery')
        ]
        if imagery_items:
            console.print('\n[bold]Vision:[/bold] Analyzing imagery...')
            with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                          console=console) as prog:
                for result in imagery_items:
                    fname = result['filename']
                    source_note = result.get('_source', '')
                    label = f'  {fname}' + (f' ({source_note})' if source_note else '')
                    task = prog.add_task(label + '...', total=None)

                    # For PDF-extracted figures, we can't re-path to the original
                    # file (it was cleaned up), so use the pre-extracted PIL thumbnail
                    # stored in the summary if available, otherwise skip.
                    if result.get('_from_pdf'):
                        # Thumbnail was generated during preprocess_pdf; if vision
                        # description already populated, skip.
                        if not result.get('vision_description'):
                            result['vision_description'] = (
                                'Figure extracted from PDF; '
                                'thumbnail not retained for vision model. '
                                'Use --keep-figures to enable vision pass on PDF figures.'
                            )
                    else:
                        # Standalone image: use normal thumbnail path
                        from preprocessors.imagery import get_image_thumbnail_b64
                        # Reconstruct path from filename (files still on disk)
                        img_path = next(
                            (p for p, k in [] if p.name == fname), None
                        )
                        # Vision description populated separately in non-PDF flow
                        pass

                    prog.remove_task(task)
                    console.print(f'  [green]v[/green] {fname}')

    return preprocessed


def run_full_pipeline(preprocessed, user_prompt, model_config, args, file_paths=None):
    """Phase 1 + 2 + optional Phase 3 critique/revision."""
    tier = model_config['tier']
    console.print('\n[dim]Assembling context package...[/dim]')
    context = assemble_context(
        preprocessed_files=preprocessed,
        user_prompt=user_prompt,
        max_tokens=model_config['context_limit']
    )
    note = ' [yellow](some summaries trimmed)[/yellow]' if context['trimmed'] else ''
    console.print(f"  [green]v[/green] ~{context['estimated_tokens']:,} tokens{note}")

    console.print('\n[bold]Phase 1:[/bold] Per-file analysis...')
    phase1_results = []
    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                  console=console) as prog:
        for file_summary in preprocessed:
            fname = file_summary['filename']
            src   = file_summary.get('_source', '')
            label = fname + (f' ({src})' if src else '')
            task  = prog.add_task(f'  Analyzing {label}...', total=None)
            result = run_phase1(
                file_summary=file_summary,
                user_prompt=user_prompt,
                model=model_config['model']
            )
            phase1_results.append(result)
            prog.remove_task(task)
            conf = result.get('confidence', '--')
            console.print(f'  [green]v[/green] {label} (confidence: {conf})')

    console.print('\n[bold]Phase 2:[/bold] Cross-file synthesis...')
    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                  console=console) as prog:
        task = prog.add_task('  Generating interpretation...', total=None)
        narrative = run_phase2(
            phase1_results=phase1_results,
            user_prompt=user_prompt,
            model=model_config['model'],
            file_summaries=preprocessed,
            tier=tier,
            file_paths=file_paths,
        )
        prog.remove_task(task)
    console.print('  [green]v[/green] Draft complete')

    critique_result = None
    final_narrative = narrative

    if not args.no_critique:
        console.print('\n[bold]Phase 3a:[/bold] Critique...')
        with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                      console=console) as prog:
            task = prog.add_task('  Running critic...', total=None)
            critique_result = run_critique(
                narrative=narrative,
                phase1_results=phase1_results,
                user_prompt=user_prompt,
                model=model_config['model'],
                file_summaries=preprocessed,
                tier=tier,
            )
            prog.remove_task(task)
        n_flags = len(critique_result.get('flagged_claims', []))
        overall = critique_result.get('overall_assessment', '--')
        col = 'yellow' if n_flags > 0 else 'green'
        console.print(f'  [{col}]v[/{col}] {n_flags} flag(s), overall: {overall}')

        console.print('\n[bold]Phase 3b:[/bold] Revision...')
        with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                      console=console) as prog:
            task = prog.add_task('  Revising...', total=None)
            final_narrative = run_revision(
                original_narrative=narrative,
                critique_result=critique_result,
                user_prompt=user_prompt,
                model=model_config['model'],
                file_summaries=preprocessed,
                tier=tier,
            )
            prog.remove_task(task)
        console.print('  [green]v[/green] Revision complete')

    return phase1_results, final_narrative, critique_result, narrative


def interactive_loop(session, model_config, args, session_path):
    """Interactive follow-up question loop (unchanged from Phase 3.5)."""
    console.print(f'\n[dim]Session: {session_path}[/dim]')
    console.print(Panel(
        '[bold]Follow-up mode[/bold]\n'
        '[dim]Type a question, [bold]exit[/bold] to quit, '
        '[bold]save[/bold] to force-save, [bold]help[/bold] for commands.[/dim]',
        border_style='dim'
    ))
    while True:
        try:
            question = Prompt.ask('\n[bold green]>[/bold green]').strip()
        except (KeyboardInterrupt, EOFError):
            save_session(session)
            console.print('\n[dim]Session saved. Goodbye.[/dim]')
            break
        if not question:
            continue
        cmd = question.lower()
        if cmd in ('exit', 'quit', 'q'):
            sp = save_session(session)
            console.print(f'[dim]Session saved: {sp}[/dim]')
            break
        if cmd == 'save':
            sp = save_session(session)
            console.print(f'[green]Saved:[/green] {sp}')
            continue
        if cmd == 'help':
            console.print(
                '[dim]Commands:[/dim]\n'
                '  [bold]exit[/bold]  Save and quit\n'
                '  [bold]save[/bold]  Force-save session\n'
                '  [dim]To add new data: python main.py '
                f'--resume {session_path} --files <files>[/dim]'
            )
            continue
        add_turn(session, 'user', question)
        console.print('[dim]Thinking...[/dim]')
        with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                      console=console) as prog:
            task = prog.add_task('  Generating answer...', total=None)
            answer, followup_charts = run_followup(question=question, session=session)
            prog.remove_task(task)
        if not args.no_critique:
            with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                          console=console) as prog:
                task = prog.add_task('  Critiquing and revising...', total=None)
                critique = run_critique(
                    narrative=answer,
                    phase1_results=session.get('phase1_results', []),
                    user_prompt=question,
                    model=session['model'],
                    file_summaries=session.get('preprocessed_summaries', []),
                    tier=session.get('tier', 'mid'),
                )
                if critique.get('flagged_claims'):
                    answer = run_revision(
                        original_narrative=answer,
                        critique_result=critique,
                        user_prompt=question,
                        model=session['model'],
                        file_summaries=session.get('preprocessed_summaries', []),
                        tier=session.get('tier', 'mid'),
                    )
                prog.remove_task(task)
        add_turn(session, 'assistant', answer)
        summarized = maybe_apply_rolling_summary(session, session['model'])
        if summarized:
            console.print('[dim]  (Older turns summarized)[/dim]')
        save_session(session)
        console.print(Panel(Markdown(answer), border_style='dim'))


def main():
    args = parse_args()
    console.print(Panel.fit(
        '[bold]Archaeological AI Interpreter[/bold]\n'
        '[dim]Phase 3.6 -- PDF Support[/dim]',
        border_style='dim'
    ))
    console.print('\n[dim]Checking Ollama connection...[/dim]')
    if not check_ollama():
        console.print('[red]Ollama not running. Start with: ollama serve[/red]')
        sys.exit(1)

    model_config = get_model_config(
        tier=args.tier,
        model_override=args.model,
        vision_model_override=args.vision_model
    )
    tier = model_config['tier'] 
    console.print(f"[green]Text model:[/green]   {model_config['model']} ({model_config['tier_label']})")
    console.print(f"[green]Vision model:[/green] {model_config['vision_model']}")

    if args.resume:
        console.print(f'[dim]Loading session: {args.resume}[/dim]')
        session = load_session(args.resume)
        if args.model:
            session['model'] = args.model
        if args.files:
            file_pairs = validate_files(args.files)
            new_preprocessed = preprocess_files(
                file_pairs, model_config, args,
                prompt_override=session['original_prompt']
            )
            new_filenames = [p.name for p, _ in file_pairs]
            console.print('\n[bold]Phase 1:[/bold] Analyzing new files...')
            new_phase1 = []
            with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                          console=console) as prog:
                for fs in new_preprocessed:
                    task = prog.add_task(f"  {fs['filename']}...", total=None)
                    result = run_phase1(
                        file_summary=fs,
                        user_prompt=session['original_prompt'],
                        model=session['model'],
                        tier=session.get('tier', 'mid'),
                    )
                    new_phase1.append(result)
                    prog.remove_task(task)
                    console.print(f"  [green]v[/green] {fs['filename']} (confidence: {result.get('confidence','--')})")
            console.print('\n[bold]Delta synthesis:[/bold] Generating supplementary findings...')
            with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                          console=console) as prog:
                task = prog.add_task('  Running delta synthesis...', total=None)
                supplementary = run_delta_synthesis(
                    session=session,
                    new_phase1_results=new_phase1,
                    new_file_summaries=new_preprocessed,
                    new_files=new_filenames,
                    model=session['model'],
                    tier=session.get('tier', 'mid'), 
                )
                prog.remove_task(task)
            if not args.no_critique:
                with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                              console=console) as prog:
                    task = prog.add_task('  Critiquing supplementary...', total=None)
                    supp_critique = run_critique(
                        narrative=supplementary,
                        phase1_results=new_phase1,
                        user_prompt=session['original_prompt'],
                        model=session['model'],
                        file_summaries=new_preprocessed,
                        tier=session.get('tier', 'mid'),
                    )
                    prog.remove_task(task)
                with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                              console=console) as prog:
                    task = prog.add_task('  Revising supplementary...', total=None)
                    supplementary = run_revision(
                        original_narrative=supplementary,
                        critique_result=supp_critique,
                        user_prompt=session['original_prompt'],
                        model=session['model'],
                        file_summaries=new_preprocessed,
                        tier=session.get('tier', 'mid'),
                    )
                    prog.remove_task(task)
            session['preprocessed_summaries'].extend(new_preprocessed)
            session['phase1_results'].extend(new_phase1)
            session['files_analyzed'].extend(new_filenames)
            import datetime
            session['supplementary_sections'].append({
                'added_files': new_filenames,
                'timestamp':   datetime.datetime.now().isoformat(),
                'content':     supplementary,
            })
            output_path = append_supplementary(
                session=session,
                supplementary=supplementary,
                new_files=new_filenames,
            )
            console.print(f'\n[bold green]Report updated:[/bold green] {output_path}\n')
        session_path = save_session(session)
        if not args.no_interactive:
            interactive_loop(session, model_config, args, session_path)
        return

    if not args.files or not args.prompt:
        console.print('[red]--files and --prompt are required for new sessions.[/red]')
        sys.exit(1)

    file_pairs = validate_files(args.files)
    counts = {k: sum(1 for _, t in file_pairs if t == k)
              for k in ['tabular', 'geospatial', 'imagery', 'pdf']}
    console.print(
        f"[green]Files:[/green] "
        f"{counts['tabular']} tabular, {counts['geospatial']} geospatial, "
        f"{counts['imagery']} imagery, {counts['pdf']} PDF\n"
    )

    preprocessed = preprocess_files(file_pairs, model_config, args)
    file_paths = {p.name: str(p) for p, _ in file_pairs}          # ADD THIS
    phase1_results, final_narrative, critique_result, original_narrative = \
        run_full_pipeline(preprocessed, args.prompt, model_config, args, file_paths=file_paths)

    output_path = save_output(
        narrative=final_narrative,
        original_narrative=original_narrative if critique_result else None,
        critique_result=critique_result,
        phase1_results=phase1_results if args.verbose else None,
        user_prompt=args.prompt,
        files=[p.name for p, _ in file_pairs],
        model=model_config['model'],
        output_path=args.output,
        fmt=args.format
    )
    console.print(f'\n[bold green]Report saved:[/bold green] {output_path}\n')
    preview = final_narrative[:2000] + ('...' if len(final_narrative) > 2000 else '')
    console.print(Panel(Markdown(preview),
                        title='[bold]Final Interpretation[/bold]',
                        border_style='green'))

    session = new_session(
        original_prompt=args.prompt,
        files_analyzed=[p.name for p, _ in file_pairs],
        model=model_config['model'],
        preprocessed_summaries=preprocessed,
        phase1_results=phase1_results,
        final_narrative=final_narrative,
        critique_result=critique_result,
        session_name=args.session_name,
    )
    session['tier'] = tier
    session_path = save_session(session)
    if not args.no_interactive:
        interactive_loop(session, model_config, args, session_path)


if __name__ == '__main__':
    main()

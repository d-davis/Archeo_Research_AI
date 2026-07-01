"""
Archaeological AI Interpreter -- Streamlit GUI
Replaces the Gradio version entirely.

Install:  pip install streamlit
Launch:   streamlit run app.py
Opens at: http://localhost:8501
"""
import json
import datetime
import traceback
import tempfile
import zipfile as zipmod
from pathlib import Path

import streamlit as st

from config import get_model_config, check_ollama
from preprocessors.tabular import preprocess_tabular
from preprocessors.geospatial import preprocess_geospatial
from preprocessors.imagery import preprocess_imagery, get_image_thumbnail_b64
from preprocessors.pdf import preprocess_pdf
from preprocessors.zip_handler import extract_zip, cleanup_zip_temp
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
    add_turn, maybe_apply_rolling_summary,
)
from output import save_output, append_supplementary
from visualize import generate_charts

SUPPORTED_TABULAR    = {'.csv', '.xlsx', '.xls', '.txt'}
SUPPORTED_GEOSPATIAL = {'.shp', '.geojson', '.json'}
SUPPORTED_IMAGERY    = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
SUPPORTED_PDF        = {'.pdf'}
TIER_CHOICES         = ['auto', 'cpu', 'low', 'mid', 'high']


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Archaeological AI Interpreter',
    page_icon='🏺',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ── Session state init ─────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        'session_path': None,
        'report':       None,
        'chart_paths':  [],
        'chat_history': [],
        'no_critique':  False,
        'log':          [],
        'resumed': False,
        '_session_cache': None,   # backup: store session object directly
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ────────────────────────────────────────────────────────────────────
def classify_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_TABULAR:    return 'tabular'
    if ext in SUPPORTED_GEOSPATIAL: return 'geospatial'
    if ext in SUPPORTED_IMAGERY:    return 'imagery'
    if ext in SUPPORTED_PDF:        return 'pdf'
    return 'unsupported'


def save_uploaded(uploaded_file) -> Path:
    """Save to a persistent session uploads folder, preserving original filename."""
    uploads_dir = Path('./uploads')
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / uploaded_file.name
    path.write_bytes(uploaded_file.read())
    return path


def preprocess_all(saved_paths, model_config, no_vision, keep_figures, log):
    """Preprocess all files. ZIPs are extracted first."""
    zip_dirs  = []
    expanded  = []

    for path in saved_paths:
        if path.suffix.lower() == '.zip':
            log.append(f'Extracting ZIP: {path.name}...')
            try:
                found, extract_dir = extract_zip(str(path))
                zip_dirs.append(extract_dir)
                if found:
                    log.append(f' Extracted: {", ".join(f.name for f in found)}')
                    # Copy extracted files to uploads/ for persistent access
                    uploads_dir = Path('./uploads')
                    uploads_dir.mkdir(parents=True, exist_ok=True)
                    all_files_in_zip = list(Path(extract_dir).rglob('*'))
                    for f in all_files_in_zip:
                        if f.is_file():
                            dest = uploads_dir / f.name
                            dest.write_bytes(f.read_bytes())

                    # Only add supported file types to the processing queue
                    persistent_found = []
                    for f in found:
                        dest = uploads_dir / f.name
                        st.session_state.setdefault('_zip_extracted_paths', {})[f.name] = str(dest)
                        persistent_found.append(dest)
                    expanded.extend(persistent_found)
                else:
                    log.append(f' No supported files in {path.name}')
            except Exception as e:
                log.append(f' ZIP error: {e}')
        else:
            expanded.append(path)

    preprocessed = []
    for path in expanded:
        kind = classify_file(path)
        if kind == 'unsupported':
            log.append(f'Skipped: {path.name}')
            continue
        log.append(f'Preprocessing {path.name}...')
        try:
            if kind == 'tabular':
                result = preprocess_tabular(path)
            elif kind == 'geospatial':
                result = preprocess_geospatial(path)
            elif kind == 'imagery':
                result = preprocess_imagery(path)
            elif kind == 'pdf':
                result = preprocess_pdf(path, keep_figures=keep_figures)
                for t in result.get('table_summaries', []):
                    preprocessed.append(t)
                for fig in result.get('figure_summaries', []):
                    preprocessed.append(fig)
            preprocessed.append(result)
        except Exception as e:
            log.append(f'  Error: {e}')

    if not no_vision:
        for result in preprocessed:
            if result.get('data_type') in ('raster_imagery', 'standard_imagery'):
                thumb = result.get('_thumbnail_b64')
                if thumb:
                    try:
                        result['vision_description'] = run_vision_description(
                            image_b64=thumb,
                            user_prompt='Describe with archaeological relevance.',
                            model=model_config['vision_model'],
                            filename=result['filename'],
                        )
                    except Exception as e:
                        result['vision_description'] = f'Vision error: {e}'

    for d in zip_dirs:
        cleanup_zip_temp(d)

    return preprocessed


def run_pipeline(preprocessed, prompt, model_config, no_critique, log):
    """Phase 1 + 2 + optional critique/revision."""
    log.append('Assembling context...')
    assemble_context(preprocessed_files=preprocessed, user_prompt=prompt,
                     max_tokens=model_config['context_limit'])

    log.append('Phase 1: Per-file analysis...')
    phase1_results = []
    for fs in preprocessed:
        result = run_phase1(file_summary=fs, user_prompt=prompt,
                            model=model_config['model'])
        phase1_results.append(result)
        log.append(f"  {fs['filename']} -- confidence: {result.get('confidence','?')}")

    log.append('Phase 2: Cross-file synthesis...')
    narrative = run_phase2(phase1_results=phase1_results, user_prompt=prompt,
                           model=model_config['model'], file_summaries=preprocessed)
    log.append('  Draft complete.')

    critique_result    = None
    original_narrative = narrative

    if not no_critique:
        log.append('Phase 3a: Critique...')
        critique_result = run_critique(narrative=narrative,
                                       phase1_results=phase1_results,
                                       user_prompt=prompt,
                                       model=model_config['model'],
                                       file_summaries=preprocessed)
        n = len(critique_result.get('flagged_claims', []))
        log.append(f"  {n} flag(s), overall: {critique_result.get('overall_assessment','?')}")

        log.append('Phase 3b: Revision...')
        narrative = run_revision(original_narrative=original_narrative,
                                 critique_result=critique_result,
                                 user_prompt=prompt,
                                 model=model_config['model'],
                                 file_summaries=preprocessed)
        log.append('  Revision complete.')

    return phase1_results, narrative, critique_result, original_narrative


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🏺 Archaeological AI')
    st.caption('On-premises multi-model interpreter')
    st.divider()

    ollama_ok = check_ollama()
    if ollama_ok:
        st.success('Ollama connected', icon='✅')
    else:
        st.error('Ollama not running. Run: `ollama serve`', icon='🔴')

    st.divider()
    st.subheader('Settings')
    tier         = st.selectbox('Hardware tier', TIER_CHOICES, index=0)
    model_ovr    = st.text_input('Text model override', placeholder='llama3.1:8b-instruct-q4_K_M')
    vision_ovr   = st.text_input('Vision model override', placeholder='llava:7b')
    no_vision    = st.checkbox('Skip vision model', value=False)
    no_critique  = st.checkbox('Skip critique (faster)', value=False)
    keep_figures = st.checkbox('Keep PDF figures', value=False)

    st.divider()
    if st.session_state.session_path:
        st.caption(f"Active session:\n`{Path(st.session_state.session_path).name}`")
    else:
        st.caption('No active session')


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['New Analysis', 'Follow-up', 'Resume Session'])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: New Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header('New Analysis')

    uploaded = st.file_uploader(
        'Upload files (CSV, XLSX, TXT, SHP, GeoJSON, TIF, PNG, JPEG, PDF, or ZIP)',
        accept_multiple_files=True,
        type=['csv','xlsx','xls','txt','shp','geojson','json',
              'tif','tiff','png','jpg','jpeg','pdf','zip'],
    )

    st.caption(
    "⚠️ Images containing primarily text (tables, scanned forms, annotated maps) "
    "should be pre-processed with an OCR tool (e.g. Tesseract or EasyOCR) and "
    "uploaded as .txt files. The vision model describes visual content but is not "
    "a reliable text extractor."
    )
    prompt = st.text_area('Research question',
                          placeholder='What patterns in this data suggest habitation activity?',
                          height=80)
    session_name = st.text_input('Session name (optional)', placeholder='site_42_initial')

    run_btn = st.button('Run Analysis', type='primary', disabled=not ollama_ok)

    if run_btn:
        if not uploaded:
            st.warning('Upload at least one file.')
        elif not prompt.strip():
            st.warning('Enter a research question.')
        else:
            log = []
            model_config = get_model_config(
                tier=tier,
                model_override=model_override if (model_override := model_ovr.strip()) else None,
                vision_model_override=vision_override if (vision_override := vision_ovr.strip()) else None,
            )
            log.append(f"Model: {model_config['model']} ({model_config['tier_label']})")
            log.append(f"Vision: {model_config['vision_model']}")

            saved_paths = [save_uploaded(f) for f in uploaded]

            with st.status('Running pipeline...', expanded=True) as status:
                try:
                    preprocessed = preprocess_all(
                        saved_paths, model_config, no_vision, keep_figures, log
                    )
                    if not preprocessed:
                        st.error('No files could be preprocessed.')
                        st.stop()

                    for msg in log:
                        st.write(msg)

                    phase1, narrative, critique, original = run_pipeline(
                        preprocessed, prompt, model_config, no_critique, log
                    )
                    for msg in log[len(log)-10:]:
                        st.write(msg)

                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    chart_paths = generate_charts(preprocessed, session_id=ts)

                    output_path = save_output(
                        narrative=narrative,
                        original_narrative=original if critique else None,
                        critique_result=critique,
                        user_prompt=prompt,
                        files=[f.name for f in uploaded],
                        model=model_config['model'],
                    )

                    session = new_session(
                        original_prompt=prompt,
                        files_analyzed=[f.name for f in uploaded],
                        model=model_config['model'],
                        preprocessed_summaries=preprocessed,
                        phase1_results=phase1,
                        final_narrative=narrative,
                        critique_result=critique,
                        session_name=session_name or None,
                        file_paths={
                        **{f.name: str(p) for f, p in zip(uploaded, saved_paths)},
                        **st.session_state.pop('_zip_extracted_paths', {}),
                    },
                    )
                    session_path = save_session(session)

                    st.session_state.session_path  = str(session_path)
                    st.session_state.report        = narrative
                    st.session_state.chart_paths   = [str(p) for p in chart_paths]
                    st.session_state.no_critique   = no_critique
                    st.session_state.log           = log

                    status.update(label='Analysis complete', state='complete')

                except Exception as e:
                    status.update(label='Error', state='error')
                    st.error(f'{e}\n\n{traceback.format_exc()}')

    # Results
    if st.session_state.report:
        st.divider()
        res_tab1, res_tab2, res_tab3 = st.tabs(['Report', 'Charts', 'Download'])

        with res_tab1:
            st.markdown(st.session_state.report)

        with res_tab2:
            charts = st.session_state.chart_paths
            if charts:
                cols = st.columns(2)
                for i, cp in enumerate(charts):
                    if Path(cp).exists():
                        cols[i % 2].image(cp, width='stretch')
            else:
                st.info('No charts generated for these file types.')

        with res_tab3:
            if st.session_state.report:
                report_bytes = st.session_state.report.encode('utf-8')
                st.download_button('Download report (.md)', data=report_bytes,
                                   file_name='interpretation.md', mime='text/markdown')

            if st.session_state.chart_paths:
                buf = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
                with zipmod.ZipFile(buf.name, 'w') as zf:
                    for cp in st.session_state.chart_paths:
                        if Path(cp).exists():
                            zf.write(cp, Path(cp).name)
                with open(buf.name, 'rb') as f:
                    st.download_button('Download charts (.zip)', data=f.read(),
                                       file_name='charts.zip', mime='application/zip')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Follow-up
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header('Follow-up')

    _has_session = bool(st.session_state.session_path) or bool(st.session_state._session_cache)
    if not _has_session:
        st.info('Run an analysis in Tab 1, or resume a session in Tab 3.')
    else:
        st.caption(f"Active session: `{Path(st.session_state.session_path).name}`")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

        question = st.chat_input('Ask a follow-up question...')

        if question and st.session_state.session_path:
            with st.chat_message('user'):
                st.markdown(question)
            st.session_state.chat_history.append({'role': 'user', 'content': question})

            with st.chat_message('assistant'):
                with st.spinner('Thinking...'):
                    try:
                        if st.session_state.session_path:
                            session = load_session(st.session_state.session_path)
                        else:
                            session = st.session_state._session_cache
                        add_turn(session, 'user', question)

                        answer, followup_charts = run_followup(question=question, session=session)
                          
                        if not st.session_state.no_critique:
                            critique = run_critique(
                                narrative=answer,
                                phase1_results=session.get('phase1_results', []),
                                user_prompt=question,
                                model=session['model'],
                                file_summaries=session.get('preprocessed_summaries', []),
                            )
                            if critique.get('flagged_claims'):
                                answer = run_revision(
                                    original_narrative=answer,
                                    critique_result=critique,
                                    user_prompt=question,
                                    model=session['model'],
                                    file_summaries=session.get('preprocessed_summaries', []),
                                )

                        add_turn(session, 'assistant', answer)
                        maybe_apply_rolling_summary(session, session['model'])
                        save_session(session)

                        st.markdown(answer)
                        # Display any charts returned by follow-up
                        if followup_charts:
                            cols = st.columns(min(2, len(followup_charts)))
                            for idx, cp in enumerate(followup_charts):
                                if Path(cp).exists():
                                    cols[idx % 2].image(str(cp), width='stretch')
                        st.session_state.chat_history.append({'role': 'assistant', 'content': answer})

                    except Exception as e:
                        st.error(f'Error: {e}')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Resume Session
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header('Resume Session')

    sess_file = st.file_uploader('Session file (.json)', type=['json'], key='sess_upload')
    new_files = st.file_uploader(
        'New data files (optional)',
        accept_multiple_files=True,
        type=['csv','xlsx','xls','txt','shp','geojson','json',
              'tif','tiff','png','jpg','jpeg','pdf','zip'],
        key='new_files_upload',
    )
    res_no_crit = st.checkbox('Skip critique on supplementary section', value=False, key='res_crit')
    res_btn     = st.button('Resume', type='primary', key='res_btn', disabled=not ollama_ok)

    if res_btn:
        if not sess_file:
            st.warning('Upload a session .json file.')
        else:
            tmp_sess = save_uploaded(sess_file)
            log      = []

            with st.status('Resuming session...', expanded=True) as status:
                try:
                    session = load_session(str(tmp_sess))
                    log.append(f"Loaded: {session['session_name']}")
                    log.append(f"Original query: {session['original_prompt']}")
                    
                    model_config = get_model_config(
                        tier=tier,
                        model_override=session.get('model')
                    )

                    if new_files:
                        new_saved = [save_uploaded(f) for f in new_files]
                        new_preproc = preprocess_all(new_saved, model_config, False, False, log)
                        new_fnames  = [f.name for f in new_files]

                        log.append('Phase 1: Analyzing new files...')
                        new_phase1 = []
                        for fs in new_preproc:
                            r = run_phase1(file_summary=fs,
                                           user_prompt=session['original_prompt'],
                                           model=session['model'])
                            new_phase1.append(r)
                            log.append(f"  {fs['filename']} (confidence: {r.get('confidence','?')})")

                        log.append('Delta synthesis...')
                        supplementary = run_delta_synthesis(
                            session=session, new_phase1_results=new_phase1,
                            new_file_summaries=new_preproc, new_files=new_fnames,
                            model=session['model']
                        )

                        if not res_no_crit:
                            log.append('Critiquing supplementary...')
                            sc = run_critique(narrative=supplementary,
                                              phase1_results=new_phase1,
                                              user_prompt=session['original_prompt'],
                                              model=session['model'],
                                              file_summaries=new_preproc)
                            supplementary = run_revision(
                                original_narrative=supplementary, critique_result=sc,
                                user_prompt=session['original_prompt'],
                                model=session['model'], file_summaries=new_preproc
                            )
                            log.append(f"  {len(sc.get('flagged_claims',[]))} flag(s) addressed.")

                        session['preprocessed_summaries'].extend(new_preproc)
                        session['phase1_results'].extend(new_phase1)
                        session['files_analyzed'].extend(new_fnames)
                        session['supplementary_sections'].append({
                            'added_files': new_fnames,
                            'timestamp':   datetime.datetime.now().isoformat(),
                            'content':     supplementary,
                        })

                        append_supplementary(session=session, supplementary=supplementary,
                                             new_files=new_fnames)

                        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        new_charts = generate_charts(new_preproc, session_id=f'resume_{ts}')
                        st.session_state.chart_paths = [str(p) for p in new_charts]

                        combined = session['final_narrative'] + '\n\n---\n\n' + supplementary
                        st.session_state.report = combined
                        log.append('Supplementary section added.')
                    else:
                        st.session_state.report = session['final_narrative']

                    sp = save_session(session)
                    st.session_state.session_path  = str(sp)
                    st.session_state._session_cache = session  # backup
                    st.session_state.chat_history  = []
                    st.session_state.no_critique   = res_no_crit

                    for msg in log:
                        st.write(msg)

                    st.session_state.resumed = True
                    status.update(label='Session resumed', state='complete')
                    st.rerun()

                except Exception as e:
                    status.update(label='Error', state='error')
                    st.error(f'{e}\n\n{traceback.format_exc()}')

    if st.session_state.report and st.session_state.resumed:
        st.divider()
        st.markdown(st.session_state.report)
        st.info("Session resumed. Switch to the **Follow-up** tab to ask questions.")

        if st.session_state.chart_paths:
            st.subheader('New Charts')
            cols = st.columns(2)
            for i, cp in enumerate(st.session_state.chart_paths):
                if Path(cp).exists():
                    cols[i % 2].image(cp, width='stretch')

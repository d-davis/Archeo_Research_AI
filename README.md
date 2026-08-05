# Archaeological Interpretive Synthesizer

An on-premises, multi-model pipeline for analyzing geospatial, tabular, imagery, and document data and generating structured archaeological interpretations. Runs entirely locally using open-source LLMs via [Ollama](https://ollama.com). No cloud API keys. No data egress. Available as both a command-line tool and a Streamlit web interface.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported File Types](#supported-file-types)
3. [System Requirements](#system-requirements)
4. [Installation: Conda (Recommended)](#installation-conda-recommended)
5. [Installation: pip (Alternative)](#installation-pip-alternative)
6. [Ollama Setup](#ollama-setup)
7. [Model Selection by Hardware Tier](#model-selection-by-hardware-tier)
8. [Project Structure](#project-structure)
9. [Analytics Module](#analytics-module)
10. [Usage: GUI](#usage-gui)
11. [Usage: CLI](#usage-cli)
12. [CLI Reference](#cli-reference)
13. [How It Works](#how-it-works)
14. [Sessions and Interactive Mode](#sessions-and-interactive-mode)
15. [Charts and Visualizations](#charts-and-visualizations)
16. [Output Format](#output-format)
17. [Troubleshooting](#troubleshooting)
18. [Build Phases](#build-phases)

---

## Overview

The system ingests heterogeneous archaeological data files, converts them to structured summaries, and routes them through a two-phase language model reasoning loop. A critic pass reviews the draft for unsupported claims and overreach; a revision pass produces the final narrative. After the initial report, users can ask follow-up questions grounded in the analyzed data. Sessions are persisted to disk so prior analysis can be resumed with new data at any time.

AI acts as analyst and interpreter, not detector. The system reads data that users and traditional GIS tools have already produced and generates reasoned, revisable interpretations from it.

Core design principles:

- **Preprocess before reasoning.** No raw file data reaches a language model directly. Every input passes through a domain-specific handler first.
- **Two-phase reasoning.** Each file is analyzed individually (Phase 1), then all results are synthesized together (Phase 2). This reduces hallucination and prevents any single large file from dominating the output.
- **Critique and revision loop.** A critic pass reviews the Phase 2 draft for unsupported claims, overconfidence, missing alternatives, and overreach. A revision pass addresses every flagged issue and appends an audit-trail Revision Log.
- **Interactive follow-up.** After the initial report, users can ask follow-up questions in context. Every answer goes through the full two-pass critique loop before being shown.
- **Persistent sessions.** Sessions save all context to disk. A prior session can be resumed with new data, which produces a Supplementary Findings section appended to the original report.
- **Hardware-tiered deployment.** The same codebase runs from CPU-only to high-end GPU by selecting appropriate quantized models and token caps at startup. CPU tier applies additional context trimming and disables chart generation to stay within strict context window limits.
- **Dual interface.** A Streamlit web GUI (`app.py`) and a full-featured CLI (`main.py`) share the same backend. Use whichever fits your workflow.

---

## Supported File Types

| Category    | Extensions                               |
|-------------|-------------------------------------------|
| Tabular     | `.csv`, `.xlsx`, `.xls`, `.txt`          |
| Geospatial  | `.shp`, `.geojson`, `.json`              |
| Imagery     | `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` |
| Document    | `.pdf`                                   |
| Archive     | `.zip` (GUI only — for shapefile bundles and multi-file uploads) |

**Note on images containing text (tables, forms, scanned records):** The vision model describes image content but is not a reliable OCR engine. Images where the primary analytical value is textual (e.g., tables saved as PNG, scanned field forms, annotated maps with dense labelling) should be converted to text first using a dedicated OCR tool such as [tesseract-ocr](https://github.com/tesseract-ocr/tesseract) (free, open-source) or [EasyOCR](https://github.com/JaidedAI/EasyOCR) (pip-installable, GPU-accelerated). Save the extracted text as a `.txt` file and upload that alongside or instead of the image for accurate content interpretation.

**Note on shapefiles:** A shapefile is a bundle of files. In the CLI, pass the `.shp` directly with `.dbf`, `.shx`, and `.prj` in the same folder. In the GUI, ZIP all four files together and upload the ZIP.

**Note on PDFs:** Only text-based PDFs are supported (not scanned/image-only PDFs). Text is extracted as Markdown including tables via `pymupdf4llm`. The `TEXT_CHAR_LIMIT` constant in `preprocessors/pdf.py` controls how many characters are passed per document; adjust it for your hardware tier (see model tier table).

---

## System Requirements

- **OS:** Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python:** 3.10 or newer
- **RAM:** 16 GB minimum; 32 GB recommended for large datasets
- **GPU:** Optional but strongly recommended. NVIDIA GPU with CUDA support for best performance.
- **Ollama:** Required. Must be running locally before the pipeline is executed.
- **Conda:** Recommended for managing GDAL and geospatial dependencies.

---

## Installation: Conda (Recommended)

### Step 1: Install Miniconda or Anaconda

- **Miniconda (minimal):** https://docs.conda.io/en/latest/miniconda.html
- **Anaconda (full):** https://www.anaconda.com/download

### Step 2: Create a new conda environment

```bash
conda create -n archaeo_ai python=3.11
conda activate archaeo_ai
```

### Step 3: Install geospatial dependencies via conda-forge

```bash
conda install -c conda-forge geopandas rasterio pyproj shapely
```

### Step 4: Install remaining Python dependencies via pip

```bash
pip install ollama pandas openpyxl tiktoken rich requests Pillow \
    pymupdf pymupdf4llm scipy \
    streamlit matplotlib
```

### Step 5: (Optional) Install PyTorch for CLIP semantic embeddings

Visit https://pytorch.org/get-started/locally/ for the correct install command for your platform and CUDA version, then:

```bash
pip install open-clip-torch>=2.24.0
```

### Step 6: Clone or download the project

```bash
git clone https://github.com/your-org/archaeo_ai.git
cd archaeo_ai
```

### Step 7: Verify installation

```bash
python -c "
import geopandas, rasterio, pandas, ollama, \
    pymupdf4llm, streamlit, matplotlib
print('All dependencies OK')
"
```

---

## Installation: pip (Alternative)

Use this path only if you cannot use conda. GDAL must be installed at the system level first.

### macOS

```bash
brew install gdal
pip install GDAL==$(gdal-config --version) geopandas rasterio
pip install ollama pandas openpyxl tiktoken rich requests Pillow \
    pymupdf pymupdf4llm streamlit matplotlib scipy
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal
pip install GDAL==$(gdal-config --version) geopandas rasterio
pip install ollama pandas openpyxl tiktoken rich requests Pillow \
    pymupdf pymupdf4llm streamlit matplotlib scipy
```

### Windows (pip only, not recommended)

Use conda instead. If you must use pip, install GDAL from a pre-compiled wheel:

https://github.com/cgohlke/geospatial-wheels/releases

```bash
pip install GDAL-<version>-cp311-cp311-win_amd64.whl
pip install geopandas rasterio
pip install ollama pandas openpyxl tiktoken rich requests Pillow \
    pymupdf pymupdf4llm streamlit matplotlib scipy
```

---

## Ollama Setup

**Install Ollama**

Download from https://ollama.com

**Start the Ollama server**

```bash
ollama serve
```

Keep this running in a separate terminal.

**Pull your models**

```bash
# Mid GPU (recommended)
ollama pull llama3.3:70b-instruct-q4_K_M
ollama pull llama3.2-vision:11b

# Low GPU / 8 GB VRAM
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull llava:7b

# CPU only
ollama pull phi3:mini
ollama pull llava:7b
```

---

## Model Selection by Hardware Tier

| Tier      | Hardware       | Text Model                        | Vision Model             | Quality   |
|-----------|----------------|------------------------------------|---------------------------|-----------|
| CPU Only  | No GPU         | phi3:mini                         | llava:7b                  | Basic     |
| Low GPU   | 6-8 GB VRAM    | llama3.1:8b-instruct-q4_K_M       | llava:7b                  | Good      |
| Mid GPU   | 16-24 GB VRAM  | llama3.3:70b-instruct-q4_K_M      | llama3.2-vision:11b       | Excellent |
| High GPU  | 48 GB+ VRAM    | llama3.3:70b-instruct-fp16        | llama3.2-vision:11b       | Best      |

**PDF text limit by tier** (set `TEXT_CHAR_LIMIT` in `preprocessors/pdf.py`):

| Tier      | Recommended TEXT_CHAR_LIMIT |
|-----------|------------------------------|
| CPU only  | 8,000                        |
| Low GPU   | 16,000                       |
| Mid GPU   | 50,000 (default)             |
| High GPU  | 80,000                       |

**Token generation caps by tier:** Each pipeline phase has a maximum output token cap (`num_predict`) that scales with hardware tier. These are set automatically based on detected hardware and stored in `config.py`. CPU and Low GPU tiers use tighter caps to prevent runaway generation loops on smaller models. Mid and High GPU tiers use higher caps to allow full narrative depth. No manual configuration is required.

**CPU tier limitations:** When running on CPU (phi3:mini), the following restrictions apply automatically to stay within the 4096-token context window:

- Chart generation is disabled in follow-up mode. The model will answer questions in prose only.
- Phase 1 per-file analyses are excluded from follow-up context to save tokens.
- The final narrative is truncated to approximately 400 words when injected into follow-up prompts.
- A repetition detection and truncation guard is active for delta synthesis (resumed sessions with new data).

These restrictions do not affect the initial analysis pipeline (Phases 1-3), only the interactive follow-up session.

---

## Project Structure

```
archaeo_ai/
├── app.py                  Streamlit web GUI
├── main.py                 CLI entry point and interactive loop
├── config.py               Hardware detection, model tier configuration, and per-tier token caps
├── context_assembly.py     Token budget management and context packaging
├── analytics.py            Pre-computation analytics engine
├── output.py               Report formatting, file export, supplementary append
├── session.py              Session creation, save/load, rolling summary
├── visualize.py            Chart generation from preprocessed summaries
├── requirements.txt        Python dependencies
│
├── preprocessors/
│   ├── __init__.py
│   ├── tabular.py          CSV, XLSX, TXT handler
│   ├── geospatial.py       SHP, GeoJSON handler
│   ├── imagery.py          TIF, PNG, JPEG handler + thumbnail export
│   ├── pdf.py              PDF handler via pymupdf4llm (Markdown + figures)
│   └── zip_handler.py      ZIP extraction for multi-file GUI uploads
│
├── reasoning/
│   ├── __init__.py
│   ├── phase1.py           Per-file analysis (structured JSON output)
│   ├── phase2.py           Cross-file synthesis (Markdown narrative draft)
│   ├── vision.py           Vision model image description
│   ├── critic.py           Structured critique pass
│   ├── revise.py           Revision pass with Revision Log
│   ├── followup.py         Follow-up question answering with session context
│   └── delta_synthesis.py  Supplementary findings for resumed sessions
│
├── sessions/               Auto-created: session .json files
├── output/
│   └── charts/             Auto-created: PNG charts per session
└── tmp/
    └── pdf_figures/        Temporary extracted PDF figures
```

---

## Usage: GUI

```bash
conda activate archaeo_ai
ollama serve    # separate terminal
streamlit run app.py
```

Opens at http://localhost:8501

**Tab 1: New Analysis**

Upload files individually or as a ZIP (required for shapefiles). Enter your research question. Configure optional settings in the sidebar (hardware tier, model overrides, skip flags). Click Run Analysis. The report renders as Markdown; charts display inline. Download report (.md) or all charts as a ZIP from the Downloads sub-tab.

**Tab 2: Follow-up**

Chat interface connected to the active session. Type your question and press Enter. Every answer goes through the full two-pass critique loop. Session saves automatically after every turn. Chart generation is available on Low GPU tier and above; CPU tier returns prose-only answers.

**Tab 3: Resume Session**

Upload a prior `.json` session file. Optionally add new data files. Click Resume. Delta synthesis produces a Supplementary Findings section, critique and revision are applied, and the updated report is displayed.

---

## Usage: CLI

```bash
conda activate archaeo_ai
ollama serve    # separate terminal

# New session
python main.py \
    --files survey.csv features.geojson site_photo.tif field_report.pdf \
    --prompt "What spatial and material patterns indicate site significance?"

# Named session, no follow-up loop
python main.py --files survey.csv features.geojson \
    --prompt "Describe feature distribution." \
    --session-name site_42_initial \
    --no-interactive

# Verbose output
python main.py --files survey.csv --prompt "..." --verbose

# Skip critique (faster, recommended for CPU tier)
python main.py --files data.csv --prompt "..." --no-critique

# PDF with figure retention
python main.py --files report.pdf --prompt "..." --keep-figures

# Resume session with new data
python main.py --resume sessions/site_42_initial.json \
    --files new_survey.csv new_features.shp

# Resume session, follow-ups only
python main.py --resume sessions/site_42_initial.json
```

**Interactive prompt commands (CLI)**

| Command   | Action                          |
|-----------|----------------------------------|
| `exit`    | Save session and quit            |
| `save`    | Force-save session                |
| `help`    | Show available commands           |
| Any text  | Ask a follow-up question          |

---

## CLI Reference

| Argument           | Short | Default   | Description |
|--------------------|-------|-----------|--------------|
| `--files`          | `-f`  | required  | Input files. Required for new sessions. New files to add with `--resume`. |
| `--prompt`         | `-p`  | required  | Analytical question. Required for new sessions. |
| `--resume`         | `-r`  | none      | Path to a session `.json` file to resume |
| `--session-name`   |       | auto      | Name for the session file (default: `session_<timestamp>`) |
| `--model`          | `-m`  | auto      | Ollama text model name override |
| `--vision-model`   |       | auto      | Ollama vision model name override |
| `--tier`           | `-t`  | auto      | Hardware tier: `cpu`, `low`, `mid`, `high`, `auto` |
| `--output`         | `-o`  | auto      | Output file path |
| `--format`         |       | markdown  | Output format: `markdown` or `txt` |
| `--verbose`        | `-v`  | off       | Include pre-revision draft, critique JSON, and Phase 1 analyses in appendices |
| `--no-vision`      |       | off       | Skip vision model step |
| `--no-critique`    |       | off       | Skip critique and revision loop |
| `--no-interactive` |       | off       | Run analysis only; skip the interactive follow-up loop |
| `--keep-figures`   |       | off       | Retain extracted PDF figure PNGs; also enables vision model pass on them |

---

## How It Works

**Preprocessing layer**

- **Tabular (CSV, XLSX, TXT):** Schema, statistics, value frequencies, 5-row sample, coordinate and temporal column detection.
- **Geospatial (SHP, GeoJSON):** CRS, geometry types, feature count, bounding box, attribute schema, archaeological column heuristics.
- **Analytics (`analytics.py`):** After each tabular or geospatial file is preprocessed, a full statistical suite runs in Python before the LLM sees the data. Results are injected into the Phase 1 context as pre-verified facts. Includes: descriptive statistics (mean, median, SD, IQR, skewness, kurtosis), Shapiro-Wilk normality testing, Z-score and Tukey fence outlier detection, rate of change, column ratios, Pearson and Spearman correlation, cross-tabulation, linear trend (slope, R², p-value), and for geospatial layers: feature density, bounding box area, and mean nearest-neighbour distance for point features.
- **Imagery (TIF, PNG, JPEG):** Per-band or per-channel statistics, resolution, CRS for rasters, base64 thumbnail for vision model.
- **PDF:** `pymupdf4llm.to_markdown()` converts the full document to Markdown including pipe-delimited tables. Embedded figures extracted via `pymupdf` and processed through imagery pipeline.

**Phase 1: Per-file analysis**

Each preprocessed summary analyzed independently. Returns structured JSON: observations, patterns, tables identified, confidence, limitations.

**Phase 2: Cross-file synthesis**

All Phase 1 outputs synthesized into a Markdown draft. Cites files per claim, surfaces cross-dataset patterns, distinguishes evidence from inference.

**Vision step**

Imagery files passed to a vision-capable model before Phase 1. Returns a 200-300 word archaeological description appended to the imagery summary.

**Phase 3a: Critique**

Same text model, isolated context, peer-reviewer prompt. Returns structured JSON flagging claims by type (unsupported, overconfident, missing_alternative, overreach, untraceable).

**Phase 3b: Revision**

Rewrites only flagged sections. Appends a Revision Log as permanent audit trail.

**Token management**

All model calls include a `repeat_penalty` (1.15) to prevent runaway generation loops and a `num_predict` cap scaled to the active hardware tier and pipeline phase. Caps are defined in `config.py` and selected automatically. This prevents small models (CPU/Low GPU tiers) from filling their context window and hallucinating, while allowing larger models (Mid/High GPU) full output depth.

**Follow-up**

Full session context injected per call. Full two-pass critique on every answer. Auto-save after every turn. Rolling summary at 8 turns to manage token usage.

**CPU tier follow-up:** On CPU tier, the follow-up prompt is aggressively trimmed to fit within the 4096-token context limit. Phase 1 analyses are excluded, the final narrative is truncated to approximately 400 words, and chart generation is disabled entirely. Answers are prose-only. For richer follow-up capability, use Low GPU tier or above.

**Delta synthesis**

New files preprocessed and analyzed via Phase 1 only. Produces Supplementary Findings answering: what does the new data corroborate, contradict, and extend? Full critique and revision applied. Output is checked for repetition loops before being saved to the session; truncated output is flagged with a note in the report.

---

## Sessions and Interactive Mode

Sessions saved to `./sessions/` as `.json` files. Contains all context: preprocessed summaries, Phase 1 results, final narrative, critique output, conversation history, rolling summary, supplementary sections, and hardware tier. Tier is saved to the session so resumed sessions use the correct token caps for the model that originally ran the analysis.

---

## Charts and Visualizations

Auto-generated from preprocessed summaries using matplotlib. Saved to `./output/charts/<session_id>/` and displayed inline in the GUI.

| File type        | Charts generated |
|-------------------|-------------------|
| Tabular            | Numeric column ranges (min/mean/max), categorical frequency bars |
| Geospatial         | Bounding box map, categorical attribute bars |
| Raster imagery     | Per-band statistics (mean +/- std) |
| Standard image     | Per-channel distribution (R/G/B) |
| PDF                | Word count and page count bars |

Follow-up charts are generated on demand during conversation (Tab 2 / interactive mode). Unlike session charts above, follow-up charts are produced in response to specific analytical questions and reflect exactly what the LLM determined was most useful to visualize. They are saved to `./output/charts/<session_id>_followup/`.

**Note:** Chart generation in follow-up mode is disabled on CPU tier (phi3:mini). The model is too small to reliably emit the structured `CHART_REQUEST:` format, and the context window is too limited to include chart instructions alongside session data. Use Low GPU tier or above for follow-up charts.

---

## Output Format

```
./output/
    interpretation_<timestamp>.md           Initial report
    <session_name>_updated_<timestamp>.md   Resumed session report
    charts/<session_id>/                    PNG chart files
./sessions/
    <session_name>.json                     Session file
./tmp/pdf_figures/<pdf_basename>/           Temporary PDF figures
```

Report structure:

```
# Archaeological Interpretation Report
[metadata header]
[Final revised narrative]
## Revision Log
## Supplementary Findings     [resumed sessions]
## Appendix A: Pre-Revision Draft   [--verbose]
## Appendix B: Critique Report      [--verbose]
## Appendix C: Phase 1 Analyses     [--verbose]
```

---

## Troubleshooting

**"Ollama not running"**

Run: `ollama serve`

**"geopandas not installed" or GDAL errors**

Run: `conda install -c conda-forge geopandas rasterio`

**"scipy not installed" or analytics not computing**

Run: `pip install scipy>=1.11`

Shapiro-Wilk normality testing, Pearson/Spearman correlations, and linear trend analysis all require scipy. The system will warn on startup if it is missing and skip those analytics gracefully.

**Vision model not found**

Pull it: `ollama pull llava:7b` (low/CPU) or `ollama pull llama3.2-vision:11b` (mid/high)

**"pymupdf4llm not installed"**

Run: `pip install pymupdf4llm`

**Text in imagery not being read**

The vision model (LLaVA, Llama 3.2 Vision) describes visual content but is not a reliable OCR engine. Images containing primarily text — tables, scanned forms, annotated maps — should be pre-processed with a dedicated OCR tool before upload. Recommended options: [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) (requires OS-level install; `pip install pytesseract`) or [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) (`pip install easyocr`, no external install needed, downloads model weights on first run). Save OCR output as `.txt` and upload that file instead.

**PDF tables not being read**

Check `TEXT_CHAR_LIMIT` in `pdf.py`. The notes field in `--verbose` output shows what percentage of the document was passed. Increase the limit for longer documents.

**PDF has no extractable text**

The file is likely a scanned PDF. Convert to text-based PDF first, or provide a `.txt` transcript.

**Streamlit app not launching**

Verify: `pip install streamlit>=1.35.0`. Run: `streamlit run app.py`. Opens at http://localhost:8501.

**Shapefile error in GUI ("cannot open .shx")**

ZIP all shapefile components (`.shp`, `.dbf`, `.shx`, `.prj`) together and upload the ZIP. The system extracts them to a shared temp folder so geopandas can read them.

**Critique returns parse error**

Re-running typically resolves it. Use `--no-critique` if it persists.

**Follow-up context seems degraded**

Rolling summary has engaged (at 8 turns). This is expected. Key conclusions are retained but exact phrasing from older turns is summarized.

**Session file not found on --resume**

Use `--no-interactive` on future runs to write the session immediately after the initial report.

**CPU tier: model generates repetitive or nonsensical output**

This occurs when the context window fills during generation, typically during delta synthesis (resumed sessions with new data). The system includes automatic repetition detection that truncates output and appends a warning note when this is detected. If it occurs on initial analysis, reduce the number of files per session or use `--no-critique` to reduce model call count. Upgrading to Low GPU tier with llama3.1:8b significantly improves output quality and context handling.

**CPU tier: follow-up answers reference charts but nothing appears**

Chart generation is disabled on CPU tier. The model may still describe charts in prose despite instructions not to. This is a known limitation of phi3:mini's instruction-following capability. Answers remain analytically valid; only visualization is unavailable. Use Low GPU tier or above for follow-up chart generation.

**CPU tier: follow-up answers seem to lack detail**

On CPU tier, Phase 1 analyses are excluded from the follow-up prompt and the narrative is truncated to approximately 400 words to fit within the 4096-token context limit. For more detailed follow-up capability, upgrade to Low GPU tier (llama3.1:8b, 8192 token context) or higher.

**Slow follow-ups**

Use `--no-critique` to skip the two-pass critique on follow-ups, or upgrade to the mid-GPU tier.

---

## Build Phases

| Phase   | Status   | Description |
|---------|----------|--------------|
| Phase 1 | Complete | CLI, Ollama integration, tabular preprocessing, single-model reasoning |
| Phase 2 | Complete | Geospatial and imagery preprocessing, vision model, multi-file synthesis |
| Phase 3 | Complete | Critique and revision loop, Streamlit GUI, session persistence |
| Phase 4 | Complete | Interactive follow-up, delta synthesis for resumed sessions |
| Phase 5 | Complete | Analytics module (descriptive stats, normality, correlations, outliers) |
| Phase 6 | Complete | PDF support, ZIP handling, tier-aware token caps, CPU tier optimizations |

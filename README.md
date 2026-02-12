# AssemblyAI Transcriber

Upload audio — get text with speaker labels, ready to export as TXT, MD, SRT, or JSON.

> Runs locally. Transcription via [AssemblyAI](https://www.assemblyai.com/) cloud API.

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

### 2. Set your API key

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```

```env
ASSEMBLYAI_API_KEY=your_key_here
OPENAI_API_KEY=your_openai_key_here  # optional, for LLM post-processing
```

Get keys at [assemblyai.com/dashboard](https://www.assemblyai.com/dashboard) and [platform.openai.com](https://platform.openai.com/api-keys).

### 3. Run

```bash
uv run python -m src
```

On Windows: double-click **run.bat**.
Opens at [http://127.0.0.1:7860](http://127.0.0.1:7860).

## Features

| Feature | Details |
|---------|---------|
| **Diarization** | Who said what |
| **Multichannel** | Separate channel per speaker |
| **Models** | Universal-3 Pro, Universal-2, combo with fallback |
| **Languages** | 11 languages + auto-detection |
| **Word boost** | Custom vocabulary for domain terms |
| **Speaker names** | Replace "Speaker A/B/C" with real names |
| **LLM post-processing** | Fix duplications, punctuation, ASR errors via OpenAI |
| **Profanity filter** | Masks profane words |
| **Disfluencies** | Keeps filler words ("um", "uh") |
| **Context prompt** | Hint for U3-Pro accuracy |
| **Export** | TXT, Markdown, SRT, JSON |

## Project Structure

```
src/
├── __main__.py      — entry point
├── config.py        — settings from .env
├── transcriber.py   — AssemblyAI SDK wrapper, export
├── postprocessor.py — LLM text refinement (OpenAI)
├── prompts.yaml     — editable prompts and API params
└── app.py           — Gradio web UI
```

## Tech Stack

**Python** 3.11+ · **Gradio** 6.x · **AssemblyAI SDK** · **OpenAI SDK** · **pydantic-settings**

## Development

```bash
uv run ruff check .           # lint
uv run ruff format .          # format
uv add <package>              # add dependency
```

## Roadmap

- [ ] **Diff-view** — highlight LLM changes before/after

## License

Personal and educational use.
API terms: [AssemblyAI ToS](https://www.assemblyai.com/terms).

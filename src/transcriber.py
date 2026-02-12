"""AssemblyAI transcription wrapper with speaker diarization and export."""

__all__ = (
    "LANGUAGE_OPTIONS",
    "MODEL_OPTIONS",
    "TranscriptionResult",
    "Utterance",
    "export_json",
    "export_md",
    "export_srt",
    "export_txt",
    "format_speaker_text",
    "transcribe_file",
)

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import assemblyai as aai

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Utterance:
    """A single utterance from a speaker."""

    speaker: str
    text: str
    start_ms: int
    end_ms: int


@dataclass
class TranscriptionResult:
    """Structured transcription output."""

    full_text: str
    utterances: list[Utterance] = field(default_factory=list)
    raw_response: dict | None = None
    warnings: list[str] = field(default_factory=list)


def _init_client() -> None:
    """Set AssemblyAI API key from settings."""
    settings = get_settings()
    aai.settings.api_key = settings.assemblyai_api_key


LANGUAGE_OPTIONS: dict[str, str | None] = {
    "Автоопределение": None,
    "Русский": "ru",
    "English": "en",
    "Deutsch": "de",
    "Français": "fr",
    "Español": "es",
    "Italiano": "it",
    "Português": "pt",
    "中文": "zh",
    "日本語": "ja",
    "한국어": "ko",
    "Українська": "uk",
}

MODEL_OPTIONS: dict[str, list[str]] = {
    "Universal-3 Pro + Universal-2 (fallback)": ["universal-3-pro", "universal-2"],
    "Universal-3 Pro": ["universal-3-pro"],
    "Universal-2": ["universal-2"],
}

# Languages that support specific features (from AssemblyAI docs).
_ENGLISH_CODES = {"en", "en_us", "en_uk", "en_au"}
_PROFANITY_LANGS = _ENGLISH_CODES | {
    "es", "fr", "de", "it", "pt", "nl", "hi", "ja",
}
_DISFLUENCIES_LANGS = _ENGLISH_CODES


def transcribe_file(
    file_path: str | Path,
    *,
    language: str | None = None,
    speakers_expected: int | None = None,
    model_key: str | None = None,
    filter_profanity: bool = False,
    disfluencies: bool = False,
    prompt: str | None = None,
    multichannel: bool = False,
    word_boost: list[str] | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file with speaker diarization.

    Args:
        file_path: Path to the audio file.
        language: Language code (e.g. 'ru', 'en') or None for auto-detection.
        speakers_expected: Expected number of speakers, or None for auto.
        model_key: Key from MODEL_OPTIONS, or None for default.
        filter_profanity: Replace profanity with asterisks.
        disfluencies: Keep filler words (um, uh, er).
        prompt: Context prompt for Universal-3 Pro (up to 1500 words).
        multichannel: Transcribe each audio channel separately.
            Note: Incompatible with speaker diarization (speaker_labels).
        word_boost: List of domain-specific terms to boost recognition.
            Up to 200 words (U2) or 1000 words (U3-Pro).

    Returns:
        Structured transcription result with utterances.

    Raises:
        RuntimeError: If transcription fails.
    """
    _init_client()

    default_key = "Universal-3 Pro + Universal-2 (fallback)"
    fallback = ["universal-3-pro", "universal-2"]
    speech_models = MODEL_OPTIONS.get(
        model_key or default_key, fallback
    )

    warnings: list[str] = []

    config_kwargs: dict = {
        "speech_models": speech_models,
    }

    # Multichannel and speaker_labels are mutually exclusive.
    if multichannel:
        config_kwargs["multichannel"] = True
        config_kwargs["speaker_labels"] = False
        if speakers_expected:
            warnings.append(
                "⚠ Multichannel: число спикеров игнорируется — "
                "multichannel несовместим с диаризацией."
            )
    else:
        config_kwargs["speaker_labels"] = True

    if language:
        config_kwargs["language_code"] = language
    else:
        config_kwargs["language_detection"] = True

    # Check language-dependent features.
    # When auto-detect, we don't know the language upfront,
    # so skip features that might cause 400 errors.
    lang_known = language is not None
    lang_set = {language} if lang_known else set()

    if disfluencies:
        if not lang_known:
            warnings.append(
                "⚠ Слова-паразиты (disfluencies): пропущено — "
                "недоступно при автоопределении языка. "
                "Выберите English вручную."
            )
        elif lang_set & _DISFLUENCIES_LANGS:
            config_kwargs["disfluencies"] = True
        else:
            warnings.append(
                "⚠ Слова-паразиты (disfluencies): пропущено — "
                "доступно только для English."
            )

    if filter_profanity:
        if not lang_known:
            warnings.append(
                "⚠ Фильтр мата: пропущено — "
                "недоступно при автоопределении языка. "
                "Выберите язык вручную."
            )
        elif lang_set & _PROFANITY_LANGS:
            config_kwargs["filter_profanity"] = True
        else:
            warnings.append(
                f"⚠ Фильтр мата: пропущено — "
                f"не поддерживается для '{language}'."
            )

    if speakers_expected and speakers_expected > 0:
        config_kwargs["speakers_expected"] = speakers_expected

    if prompt and prompt.strip():
        config_kwargs["prompt"] = prompt.strip()

    if word_boost:
        config_kwargs["word_boost"] = word_boost

    config = aai.TranscriptionConfig(**config_kwargs)

    logger.info("Starting transcription for %s", file_path)
    transcript = aai.Transcriber().transcribe(str(file_path), config)

    if transcript.status == aai.TranscriptStatus.error:
        msg = f"Transcription failed: {transcript.error}"
        logger.error(msg)
        raise RuntimeError(msg)

    utterances = [
        Utterance(
            speaker=u.speaker,
            text=u.text,
            start_ms=u.start,
            end_ms=u.end,
        )
        for u in (transcript.utterances or [])
    ]

    raw = {
        "id": transcript.id,
        "text": transcript.text,
        "utterances": [
            {
                "speaker": u.speaker,
                "text": u.text,
                "start_ms": u.start_ms,
                "end_ms": u.end_ms,
            }
            for u in utterances
        ],
    }

    logger.info("Transcription complete: %d utterances", len(utterances))
    return TranscriptionResult(
        full_text=transcript.text or "",
        utterances=utterances,
        raw_response=raw,
        warnings=warnings,
    )


def format_speaker_text(result: TranscriptionResult) -> str:
    """Format transcription with speaker labels for display.

    Args:
        result: Transcription result.

    Returns:
        Human-readable text with speaker labels.
    """
    if not result.utterances:
        return result.full_text

    lines: list[str] = []
    for u in result.utterances:
        lines.append(f"[Speaker {u.speaker}]: {u.text}")
    return "\n\n".join(lines)


def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def export_txt(result: TranscriptionResult, output_path: Path) -> Path:
    """Export transcription as plain text with speaker labels.

    Args:
        result: Transcription result.
        output_path: Path to save the file.

    Returns:
        Path to the saved file.
    """
    output_path.write_text(format_speaker_text(result), encoding="utf-8")
    return output_path


def export_md(result: TranscriptionResult, output_path: Path) -> Path:
    """Export transcription as Markdown with speaker headers.

    Args:
        result: Transcription result.
        output_path: Path to save the file.

    Returns:
        Path to the saved file.
    """
    lines: list[str] = ["# Транскрипция", ""]

    if not result.utterances:
        lines.append(result.full_text)
    else:
        current_speaker: str | None = None
        for u in result.utterances:
            if u.speaker != current_speaker:
                current_speaker = u.speaker
                lines.append(f"## Speaker {u.speaker}")
                lines.append("")
            lines.append(u.text)
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_srt(result: TranscriptionResult, output_path: Path) -> Path:
    """Export transcription as SRT subtitles with speaker labels.

    Args:
        result: Transcription result.
        output_path: Path to save the file.

    Returns:
        Path to the saved file.
    """
    lines: list[str] = []
    for idx, u in enumerate(result.utterances, start=1):
        start = _ms_to_srt_time(u.start_ms)
        end = _ms_to_srt_time(u.end_ms)
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(f"[Speaker {u.speaker}]: {u.text}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_json(result: TranscriptionResult, output_path: Path) -> Path:
    """Export full transcription data as JSON.

    Args:
        result: Transcription result.
        output_path: Path to save the file.

    Returns:
        Path to the saved file.
    """
    output_path.write_text(
        json.dumps(result.raw_response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path

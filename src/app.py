"""Gradio web UI for AssemblyAI speech-to-text transcription."""

__all__ = ("main",)

import logging
import socket
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import gradio as gr

from src.postprocessor import (
    PostProcessingTask,
    get_available_tasks,
    process_text,
)
from src.transcriber import (
    LANGUAGE_OPTIONS,
    MODEL_OPTIONS,
    TranscriptionResult,
    Utterance,
    export_json,
    export_md,
    export_srt,
    export_txt,
    format_speaker_text,
    transcribe_file,
)

logger = logging.getLogger(__name__)

_last_result: TranscriptionResult | None = None
_speaker_mapping: dict[str, str] = {}


# ============================================================================
# Operation History
# ============================================================================


@dataclass
class OperationHistoryItem:
    """Single operation in the history stack.

    Attributes:
        task_label: Human-readable task label (e.g., "Удаление двоений").
        text_before: Text before the operation.
        text_after: Text after the operation.
        timestamp: When the operation was applied.
    """

    task_label: str
    text_before: str
    text_after: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


def _format_history_html(history: list[OperationHistoryItem]) -> str:
    """Format operation history as HTML tags with remove buttons.

    Args:
        history: List of operation history items.

    Returns:
        HTML string with styled tags.
    """
    if not history:
        return ""

    tags = []
    for i, item in enumerate(history):
        # Each tag shows task label and a remove button (✕).
        # Button ID encodes the index for removal.
        tag = (
            f'<span class="history-tag" style="'
            f"display: inline-flex; align-items: center; "
            f"background: #3b82f6; color: white; padding: 4px 8px; "
            f"border-radius: 4px; margin: 2px 4px 2px 0; font-size: 13px;"
            f'">'
            f"{item.task_label} "
            f'<span style="opacity: 0.7; margin-left: 4px;">({item.timestamp})</span>'
            f"</span>"
        )
        tags.append(tag)

    return "".join(tags)


# Maximum number of speaker name fields in UI.
_MAX_SPEAKERS = 10


def _transcribe(
    audio_file: str | None,
    language_label: str,
    speakers: int,
    model_label: str,
    filter_profanity: bool,
    disfluencies: bool,
    multichannel: bool,
    word_boost_str: str,
    prompt: str,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Handle transcription and return text + export file paths.

    Args:
        audio_file: Path to uploaded audio file.
        language_label: Human-readable language name from dropdown.
        speakers: Expected number of speakers (0 = auto).
        model_label: Human-readable model name from dropdown.
        filter_profanity: Filter profanity flag.
        disfluencies: Keep filler words flag.
        multichannel: Transcribe channels separately flag.
        word_boost_str: Comma-separated domain terms.
        prompt: Context prompt for Universal-3 Pro.

    Returns:
        Tuple of (text, txt_path, md_path, srt_path, json_path).
    """
    global _last_result

    if not audio_file:
        return (
            "Загрузите аудиофайл для начала транскрипции.",
            None,
            None,
            None,
            None,
        )

    language_code = LANGUAGE_OPTIONS.get(language_label)
    speakers_expected = speakers if speakers > 0 else None

    # Parse word boost from comma-separated string.
    word_boost: list[str] | None = None
    if word_boost_str and word_boost_str.strip():
        word_boost = [w.strip() for w in word_boost_str.split(",") if w.strip()]
        if not word_boost:
            word_boost = None

    try:
        result = transcribe_file(
            audio_file,
            language=language_code,
            speakers_expected=speakers_expected,
            model_key=model_label,
            filter_profanity=filter_profanity,
            disfluencies=disfluencies,
            prompt=prompt or None,
            multichannel=multichannel,
            word_boost=word_boost,
        )
        _last_result = result
        text = format_speaker_text(result)

        if result.warnings:
            prefix = "\n".join(result.warnings) + "\n\n"
            text = prefix + text

        return (
            text,
            _make_export("txt"),
            _make_export("md"),
            _make_export("srt"),
            _make_export("json"),
        )
    except RuntimeError:
        logger.exception("Transcription failed")
        return (
            "Ошибка транскрипции. Проверьте файл и API-ключ.",
            None,
            None,
            None,
            None,
        )
    except Exception:
        logger.exception("Unexpected error during transcription")
        return (
            "Непредвиденная ошибка. Подробности в логах.",
            None,
            None,
            None,
            None,
        )


def _make_export(fmt: str) -> str | None:
    """Create an export file for the last transcription result.

    Args:
        fmt: Export format — 'txt', 'md', 'srt', or 'json'.

    Returns:
        Path to the exported file, or None if unavailable.
    """
    if _last_result is None:
        return None

    suffix = f".{fmt}"
    tmp = Path(tempfile.mktemp(suffix=suffix, prefix="transcript_"))

    exporters = {
        "txt": export_txt,
        "md": export_md,
        "srt": export_srt,
        "json": export_json,
    }
    exporters[fmt](_last_result, tmp)
    return str(tmp)


def _get_unique_speakers() -> list[str]:
    """Get list of unique speaker IDs from last transcription."""
    if _last_result is None or not _last_result.utterances:
        return []
    seen: set[str] = set()
    speakers: list[str] = []
    for u in _last_result.utterances:
        if u.speaker not in seen:
            seen.add(u.speaker)
            speakers.append(u.speaker)
    return speakers


def _apply_speaker_names(
    text: str, *name_inputs: str
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Apply custom speaker names to the transcription text.

    Args:
        text: Current text in the result field.
        *name_inputs: Up to _MAX_SPEAKERS name inputs from UI.

    Returns:
        Tuple of (updated_text, txt_path, md_path, srt_path, json_path).
    """
    global _speaker_mapping

    if _last_result is None:
        return text, None, None, None, None

    speakers = _get_unique_speakers()
    if not speakers:
        return text, None, None, None, None

    # Build mapping from speaker IDs to custom names.
    _speaker_mapping = {}
    for i, speaker in enumerate(speakers):
        if i < len(name_inputs) and name_inputs[i] and name_inputs[i].strip():
            _speaker_mapping[speaker] = name_inputs[i].strip()
        else:
            _speaker_mapping[speaker] = f"Speaker {speaker}"

    # Apply replacement to text.
    updated_text = text
    for speaker, name in _speaker_mapping.items():
        updated_text = updated_text.replace(f"[Speaker {speaker}]", f"[{name}]")

    # Regenerate exports with updated names.
    return (
        updated_text,
        _make_export_with_names("txt"),
        _make_export_with_names("md"),
        _make_export_with_names("srt"),
        _make_export_with_names("json"),
    )


def _postprocess_text(
    text: str,
    task_label: str,
    word_boost_str: str,
    history: list[OperationHistoryItem],
) -> tuple[str, gr.update, list[OperationHistoryItem], str]:
    """Apply LLM post-processing to transcription text.

    Args:
        text: Current text in the result field (or processed field if history exists).
        task_label: Human-readable task label from dropdown.
        word_boost_str: Comma-separated domain terms for LLM protection.
        history: Current operation history stack.

    Returns:
        Tuple of (processed_text, field_update, history, history_html).
    """
    if not text or not text.strip():
        return (
            "Нет текста для обработки.",
            gr.update(visible=True),
            history,
            _format_history_html(history),
        )

    # Find task by label.
    task: PostProcessingTask | None = None
    for label, t in get_available_tasks():
        if label == task_label:
            task = t
            break

    if task is None:
        return (
            f"Неизвестная задача: {task_label}",
            gr.update(visible=True),
            history,
            _format_history_html(history),
        )

    # Parse word boost from comma-separated string.
    word_boost: list[str] | None = None
    if word_boost_str and word_boost_str.strip():
        word_boost = [w.strip() for w in word_boost_str.split(",") if w.strip()]
        if not word_boost:
            word_boost = None

    result = process_text(text, task, word_boost=word_boost)

    if not result.success:
        error_msg = f"⚠ Ошибка: {result.error_message}"
        return (
            error_msg,
            gr.update(visible=True),
            history,
            _format_history_html(history),
        )

    # Add to history.
    new_item = OperationHistoryItem(
        task_label=task_label,
        text_before=text,
        text_after=result.processed_text,
    )
    new_history = [*history, new_item]

    return (
        result.processed_text,
        gr.update(visible=True),
        new_history,
        _format_history_html(new_history),
    )


def _restore_original(
    original_text: str,
) -> tuple[str, gr.update, list[OperationHistoryItem], str]:
    """Restore original text, clear history, hide processed field.

    Args:
        original_text: The original transcription text.

    Returns:
        Tuple to clear processed field, clear history.
    """
    return (
        "",
        gr.update(visible=False),
        [],  # Clear history
        "",  # Clear history HTML
    )


def _undo_last_operation(
    history: list[OperationHistoryItem],
) -> tuple[str, gr.update, list[OperationHistoryItem], str]:
    """Undo the last operation from history.

    Args:
        history: Current operation history stack.

    Returns:
        Tuple of (restored_text, visibility_update, updated_history, history_html).
    """
    if not history:
        return (
            "",
            gr.update(visible=False),
            [],
            "",
        )

    # Get the last operation and restore text_before.
    last_op = history[-1]
    new_history = history[:-1]

    if not new_history:
        # No more operations — hide the processed field.
        return (
            "",
            gr.update(visible=False),
            [],
            "",
        )

    # There are still operations — show the previous result.
    return (
        last_op.text_before,
        gr.update(visible=True),
        new_history,
        _format_history_html(new_history),
    )


def _make_export_with_names(fmt: str) -> str | None:
    """Create export with custom speaker names applied.

    Args:
        fmt: Export format — 'txt', 'md', 'srt', or 'json'.

    Returns:
        Path to the exported file, or None if unavailable.
    """
    if _last_result is None:
        return None

    suffix = f".{fmt}"
    tmp = Path(tempfile.mktemp(suffix=suffix, prefix="transcript_"))

    # Create a modified result with renamed speakers.
    from dataclasses import replace

    renamed_utterances = []
    for u in _last_result.utterances:
        new_speaker = _speaker_mapping.get(u.speaker, u.speaker)
        renamed_utterances.append(
            Utterance(
                speaker=new_speaker,
                text=u.text,
                start_ms=u.start_ms,
                end_ms=u.end_ms,
            )
        )

    modified_result = replace(
        _last_result,
        utterances=renamed_utterances,
    )

    # Update raw_response with new speaker names.
    if modified_result.raw_response:
        modified_result.raw_response = {
            **modified_result.raw_response,
            "utterances": [
                {
                    "speaker": u.speaker,
                    "text": u.text,
                    "start_ms": u.start_ms,
                    "end_ms": u.end_ms,
                }
                for u in renamed_utterances
            ],
        }

    exporters = {
        "txt": export_txt,
        "md": export_md,
        "srt": export_srt,
        "json": export_json,
    }
    exporters[fmt](modified_result, tmp)
    return str(tmp)


def _build_ui() -> gr.Blocks:
    """Construct the Gradio interface."""
    custom_css = """
    .dark-gray-btn {
        background-color: #4a4a4a !important;
        border-color: #4a4a4a !important;
        color: white !important;
    }
    .dark-gray-btn:hover {
        background-color: #3a3a3a !important;
        border-color: #3a3a3a !important;
    }
    .dark-red-btn {
        background-color: #8b2020 !important;
        border-color: #8b2020 !important;
        color: white !important;
    }
    .dark-red-btn:hover {
        background-color: #6b1515 !important;
        border-color: #6b1515 !important;
    }
    """
    with gr.Blocks(css=custom_css) as app:
        # State for operation history.
        history_state = gr.State(value=[])

        gr.Markdown("# AssemblyAI — Распознавание речи")
        gr.Markdown("Загрузите аудиофайл для транскрипции с определением спикеров.")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Аудиофайл",
                    type="filepath",
                )
                language_input = gr.Dropdown(
                    choices=list(LANGUAGE_OPTIONS.keys()),
                    value="Автоопределение",
                    label="Язык аудио",
                )
                model_input = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value="Universal-3 Pro + Universal-2 (fallback)",
                    label="Модель",
                )
                speakers_input = gr.Number(
                    value=0,
                    label="Кол-во спикеров (0 = авто)",
                    precision=0,
                    minimum=0,
                    maximum=20,
                )

                with gr.Accordion("Дополнительные параметры", open=False):
                    profanity_input = gr.Checkbox(
                        value=False,
                        label="Фильтр ненормативной лексики",
                    )
                    disfluencies_input = gr.Checkbox(
                        value=False,
                        label="Сохранять слова-паразиты (ум, э-э)",
                    )
                    multichannel_input = gr.Checkbox(
                        value=False,
                        label="Multichannel (раздельные каналы)",
                    )
                    word_boost_input = gr.Textbox(
                        label="Word boost (доменные термины)",
                        placeholder="термин1, термин2, термин3",
                        lines=2,
                    )
                    prompt_input = gr.Textbox(
                        label="Контекстная подсказка (U3-Pro)",
                        placeholder="Пример: Это звонок в медклинику.",
                        lines=3,
                    )

                transcribe_btn = gr.Button(
                    "Транскрибировать",
                    variant="primary",
                )

            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Результат транскрипции (оригинал)",
                    lines=12,
                    interactive=False,
                )

                with gr.Row():
                    processed_text = gr.Textbox(
                        label="После обработки",
                        lines=12,
                        visible=False,
                    )

                with gr.Accordion("Замена имён спикеров", open=False):
                    gr.Markdown(
                        "Введите имена для замены `[Speaker A]`, `[Speaker B]` и т.д."
                    )
                    speaker_inputs: list[gr.Textbox] = []
                    for i in range(_MAX_SPEAKERS):
                        speaker_inputs.append(
                            gr.Textbox(
                                label=f"Спикер {chr(65 + i)}",
                                placeholder=f"Имя для Speaker {chr(65 + i)}",
                                visible=(i < 2),  # Show first 2 by default
                            )
                        )
                    apply_names_btn = gr.Button(
                        "Применить имена",
                        variant="secondary",
                    )

                with gr.Accordion("LLM-постобработка (OpenAI)", open=False):
                    gr.Markdown(
                        "Применение LLM для улучшения текста. "
                        "Требует OPENAI_API_KEY в .env"
                    )
                    postprocess_task = gr.Dropdown(
                        choices=[label for label, _ in get_available_tasks()],
                        value=get_available_tasks()[0][0],
                        label="Задача постобработки",
                    )
                    with gr.Row():
                        postprocess_btn = gr.Button(
                            "Применить постобработку",
                            variant="primary",
                            scale=3,
                        )
                        undo_btn = gr.Button(
                            "↶ Отменить",
                            variant="secondary",
                            scale=1,
                            elem_classes=["dark-gray-btn"],
                        )
                        restore_btn = gr.Button(
                            "↩ Сбросить всё",
                            variant="secondary",
                            scale=1,
                            elem_classes=["dark-red-btn"],
                        )

                    # History tags display.
                    history_html = gr.HTML(
                        value="",
                        label="История операций",
                        visible=True,
                    )

                gr.Markdown("### Экспорт результата")
                with gr.Row():
                    dl_txt = gr.File(
                        label="TXT",
                        interactive=False,
                    )
                    dl_md = gr.File(
                        label="MD",
                        interactive=False,
                    )
                    dl_srt = gr.File(
                        label="SRT",
                        interactive=False,
                    )
                    dl_json = gr.File(
                        label="JSON",
                        interactive=False,
                    )

        # Handler for applying speaker names.
        apply_names_btn.click(
            fn=_apply_speaker_names,
            inputs=[output_text, *speaker_inputs],
            outputs=[output_text, dl_txt, dl_md, dl_srt, dl_json],
        )

        # Handler for LLM post-processing.
        # Use original text if no operations yet, otherwise processed.
        def _process_from_original(
            original: str,
            processed: str,
            task_label: str,
            word_boost_str: str,
            history: list[OperationHistoryItem],
        ) -> tuple[str, gr.update, list[OperationHistoryItem], str]:
            """Use original text if no operations yet, otherwise processed."""
            text_to_process = processed if processed and history else original
            return _postprocess_text(
                text_to_process, task_label, word_boost_str, history
            )

        postprocess_btn.click(
            fn=_process_from_original,
            inputs=[
                output_text,
                processed_text,
                postprocess_task,
                word_boost_input,
                history_state,
            ],
            outputs=[
                processed_text,
                processed_text,
                history_state,
                history_html,
            ],
        )

        # Handler for restoring original text (clear all).
        restore_btn.click(
            fn=_restore_original,
            inputs=[output_text],
            outputs=[
                processed_text,
                processed_text,
                history_state,
                history_html,
            ],
        )

        # Handler for undoing last operation.
        undo_btn.click(
            fn=_undo_last_operation,
            inputs=[history_state],
            outputs=[
                processed_text,
                processed_text,
                history_state,
                history_html,
            ],
        )

        transcribe_btn.click(
            fn=_transcribe,
            inputs=[
                audio_input,
                language_input,
                speakers_input,
                model_input,
                profanity_input,
                disfluencies_input,
                multichannel_input,
                word_boost_input,
                prompt_input,
            ],
            outputs=[output_text, dl_txt, dl_md, dl_srt, dl_json],
        )

    return app


_DEFAULT_PORT = 7860


def _is_port_in_use(port: int) -> bool:
    """Check if a port is already in use.

    Args:
        port: Port number to check.

    Returns:
        True if port is in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
        return result == 0


def main() -> None:
    """Launch the Gradio application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if _is_port_in_use(_DEFAULT_PORT):
        logger.error(
            "Порт %d уже занят. Возможно, приложение уже запущено.",
            _DEFAULT_PORT,
        )
        print(
            f"\n❌ Ошибка: порт {_DEFAULT_PORT} уже занят.\n"
            "   Возможно, приложение уже запущено в другом окне.\n"
            "   Закройте другой экземпляр или используйте другой порт.\n"
        )
        sys.exit(1)

    app = _build_ui()
    app.launch(share=False, server_port=_DEFAULT_PORT)


if __name__ == "__main__":
    main()

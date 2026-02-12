"""LLM-based text post-processing for transcription results.

Uses OpenAI API for text refinement tasks:
- Formal corrections (duplicates, punctuation, typography)
- Context-based error correction
- SRBS-style semantic extraction (not summarization)

Architecture:
    postprocessor.py → prompts.yaml → openai_smart_client → OpenAI API

    Prompts and parameters are loaded from src/prompts.yaml.
    Edit that file to customize LLM behavior without code changes.
"""

__all__ = (
    "AVAILABLE_TASKS",
    "MODEL_SEMANTIC",
    "MODEL_SIMPLE",
    "PostProcessingResult",
    "PostProcessingTask",
    "get_available_tasks",
    "load_prompts_config",
    "process_text",
)

import logging
import sys
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to prompts configuration file.
_PROMPTS_CONFIG_PATH = Path(__file__).parent / "prompts.yaml"

# Add openai_smart_client to path.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# OpenAI client setup via openai_smart_client.
_call_openai = None

try:
    from openai_smart_client import call_openai as _smart_call_openai

    def _call_openai_impl(
        prompt: str,
        *,
        system: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> str:
        """Call OpenAI API via openai_smart_client.

        Args:
            prompt: User message.
            system: System prompt (instructions).
            model: Model to use.

        Returns:
            Model response text.
        """
        return _smart_call_openai(
            prompt,
            system=system,
            model=model,
        )

    _call_openai = _call_openai_impl
    logger.info("openai_smart_client initialized")

except ImportError as e:
    logger.warning("openai_smart_client not available: %s", e)


# ============================================================================
# Configuration loading
# ============================================================================


@lru_cache(maxsize=1)
def load_prompts_config() -> dict[str, Any]:
    """Load prompts configuration from YAML file.

    Returns:
        Configuration dictionary with models, api_params, tasks.

    Raises:
        FileNotFoundError: If prompts.yaml doesn't exist.
        yaml.YAMLError: If YAML is malformed.
    """
    if not _PROMPTS_CONFIG_PATH.exists():
        logger.error("Prompts config not found: %s", _PROMPTS_CONFIG_PATH)
        raise FileNotFoundError(f"Prompts config not found: {_PROMPTS_CONFIG_PATH}")

    with _PROMPTS_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded prompts config from %s", _PROMPTS_CONFIG_PATH)
    return config


def reload_prompts_config() -> dict[str, Any]:
    """Reload prompts configuration (clears cache).

    Use when prompts.yaml has been modified at runtime.

    Returns:
        Fresh configuration dictionary.
    """
    load_prompts_config.cache_clear()
    return load_prompts_config()


def _get_config() -> dict[str, Any]:
    """Get prompts config, falling back to hardcoded defaults on error."""
    try:
        return load_prompts_config()
    except Exception as e:
        logger.warning("Failed to load prompts config, using defaults: %s", e)
        return _DEFAULT_CONFIG


# Model constants (GPT-4.1 family, available without org verification).
# These are defaults, actual values come from prompts.yaml.
MODEL_SIMPLE = "gpt-4.1-mini"  # For formal, near-deterministic tasks
MODEL_SEMANTIC = "gpt-4.1"  # For interpretive, context-dependent tasks


class PostProcessingTask(StrEnum):
    """Available post-processing task types."""

    # Formal, near-deterministic tasks (use MODEL_SIMPLE)
    REMOVE_DUPLICATES = auto()  # "более более", "все. все"
    FIX_PUNCTUATION = auto()  # Punctuation and capitalization
    NORMALIZE_TYPOGRAPHY = auto()  # Quotes, dashes, ellipses
    FIX_ASR_ERRORS = auto()  # Obvious ASR mistakes
    NUMBERS_TO_TEXT = auto()  # Digit-to-text conversion
    REMOVE_DISFLUENCIES = auto()  # "ум", "э-э", fillers

    # Semi-interpretive tasks (use MODEL_SEMANTIC)
    CONTEXT_ERROR_FIX = auto()  # Context-based error correction
    AUTOCOMPLETE = auto()  # Complete truncated endings
    SEMANTIC_CONSOLIDATION = auto()  # Merge semantic duplicates

    # SRBS extraction (use MODEL_SEMANTIC)
    SRBS_EXTRACTION = auto()  # Extract main content, remove secondary


# Mapping from task enum to YAML task key.
_TASK_TO_KEY: dict[PostProcessingTask, str] = {
    PostProcessingTask.REMOVE_DUPLICATES: "remove_duplicates",
    PostProcessingTask.FIX_PUNCTUATION: "fix_punctuation",
    PostProcessingTask.NORMALIZE_TYPOGRAPHY: "normalize_typography",
    PostProcessingTask.FIX_ASR_ERRORS: "fix_asr_errors",
    PostProcessingTask.NUMBERS_TO_TEXT: "numbers_to_text",
    PostProcessingTask.REMOVE_DISFLUENCIES: "remove_disfluencies",
    PostProcessingTask.CONTEXT_ERROR_FIX: "context_error_fix",
    PostProcessingTask.AUTOCOMPLETE: "autocomplete",
    PostProcessingTask.SEMANTIC_CONSOLIDATION: "semantic_consolidation",
    PostProcessingTask.SRBS_EXTRACTION: "srbs_extraction",
}


def _get_task_labels() -> dict[PostProcessingTask, str]:
    """Get task labels from config or use defaults."""
    config = _get_config()
    tasks = config.get("tasks", {})

    labels = {}
    for task, key in _TASK_TO_KEY.items():
        task_config = tasks.get(key, {})
        labels[task] = task_config.get("label", key.replace("_", " ").title())

    return labels


# Human-readable task labels for UI.
# Loaded dynamically from prompts.yaml.
TASK_LABELS: dict[PostProcessingTask, str] = _get_task_labels()


def get_available_tasks() -> list[tuple[str, PostProcessingTask]]:
    """Get available tasks with labels for UI selection.

    Returns:
        List of (label, task) tuples.
    """
    labels = _get_task_labels()
    return [(label, task) for task, label in labels.items()]


# For backward compatibility.
AVAILABLE_TASKS: list[tuple[str, PostProcessingTask]] = get_available_tasks()


@dataclass
class PostProcessingResult:
    """Result of post-processing operation."""

    original_text: str
    processed_text: str
    task: PostProcessingTask
    success: bool
    error_message: str | None = None
    changes_made: list[str] = field(default_factory=list)


# ============================================================================
# Default configuration (fallback if YAML fails to load)
# ============================================================================

_DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        "simple": "gpt-4.1-mini",
        "semantic": "gpt-4.1",
    },
    "api_params": {
        "timeout": 60,
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    "word_boost_instruction": (
        "ДОМЕННЫЕ ТЕРМИНЫ (сохранять без изменений, НЕ исправлять):\n{terms}\n"
    ),
    "speaker_preserve": (
        "\n\nВАЖНО: Сохраняй разметку спикеров в квадратных скобках "
        "([Speaker A], [Speaker B], [Иван], [Мария] и т.д.) без изменений!"
    ),
    "tasks": {},  # Will use hardcoded fallback prompts
}

# Fallback prompts (used if YAML doesn't have a task).
_FALLBACK_PROMPTS: dict[str, str] = {
    "remove_duplicates": "Удали повторяющиеся слова подряд. Верни только текст.",
    "fix_punctuation": "Исправь пунктуацию. Верни только текст.",
    "normalize_typography": "Нормализуй типографику. Верни только текст.",
    "fix_asr_errors": "Исправь ошибки распознавания. Верни только текст.",
    "numbers_to_text": "Преобразуй числа в слова. Верни только текст.",
    "remove_disfluencies": "Удали слова-паразиты. Верни только текст.",
    "context_error_fix": "Исправь ошибки по контексту. Верни только текст.",
    "autocomplete": "Дополни оборванные слова. Верни только текст.",
    "semantic_consolidation": "Объедини смысловые повторы. Верни только текст.",
    "srbs_extraction": "Извлеки главное по методу SRBS. Верни связный текст.",
}


# ============================================================================
# Prompt building
# ============================================================================


def build_system_prompt(
    task: PostProcessingTask,
    *,
    word_boost: list[str] | None = None,
) -> str:
    """Build system prompt for a task, optionally with word_boost terms.

    Args:
        task: The post-processing task type.
        word_boost: List of domain-specific terms to protect from "correction".

    Returns:
        Complete system prompt with all instructions.
    """
    config = _get_config()
    task_key = _TASK_TO_KEY.get(task)

    if not task_key:
        return "Обработай текст. Верни только результат."

    # Get task prompt from config or fallback.
    tasks = config.get("tasks", {})
    task_config = tasks.get(task_key, {})
    base_prompt = task_config.get("prompt", _FALLBACK_PROMPTS.get(task_key, ""))

    # Build prompt parts.
    parts = []

    # Add word_boost instruction if terms provided.
    if word_boost:
        word_boost_template = config.get(
            "word_boost_instruction",
            _DEFAULT_CONFIG["word_boost_instruction"],
        )
        terms_str = ", ".join(word_boost)
        word_boost_instruction = word_boost_template.format(terms=terms_str)
        parts.append(word_boost_instruction.strip())

    # Add main prompt.
    parts.append(base_prompt.strip())

    # Add speaker preservation instruction.
    speaker_preserve = config.get(
        "speaker_preserve",
        _DEFAULT_CONFIG["speaker_preserve"],
    )
    parts.append(speaker_preserve.strip())

    return "\n\n".join(parts)


def _get_model_for_task(task: PostProcessingTask) -> str:
    """Get the appropriate model for a task from config.

    Args:
        task: The post-processing task type.

    Returns:
        Model name string.
    """
    config = _get_config()
    task_key = _TASK_TO_KEY.get(task)

    if not task_key:
        return MODEL_SIMPLE

    tasks = config.get("tasks", {})
    task_config = tasks.get(task_key, {})
    model_type = task_config.get("model", "simple")  # "simple" or "semantic"

    models = config.get("models", _DEFAULT_CONFIG["models"])
    return models.get(model_type, MODEL_SIMPLE)


def _build_user_prompt(text: str) -> str:
    """Build user prompt with the text to process.

    Args:
        text: The text to process.

    Returns:
        Formatted user prompt.
    """
    return f"Обработай следующий текст:\n\n{text}"


def process_text(
    text: str,
    task: PostProcessingTask,
    *,
    model: str | None = None,
    word_boost: list[str] | None = None,
) -> PostProcessingResult:
    """Process text using LLM for the specified task.

    Args:
        text: Input text to process.
        task: Post-processing task type.
        model: OpenAI model override. If None, auto-selects based on task
            from prompts.yaml configuration.
        word_boost: List of domain-specific terms to protect from "correction".
            These terms are added to the system prompt to prevent LLM from
            changing specialized vocabulary.

    Returns:
        PostProcessingResult with original and processed text.
    """
    if _call_openai is None:
        return PostProcessingResult(
            original_text=text,
            processed_text=text,
            task=task,
            success=False,
            error_message="OpenAI client не доступен. Проверьте установку.",
        )

    if not text or not text.strip():
        return PostProcessingResult(
            original_text=text,
            processed_text=text,
            task=task,
            success=True,
            error_message=None,
        )

    # Build system prompt with optional word_boost.
    system_prompt = build_system_prompt(task, word_boost=word_boost)

    if not system_prompt:
        return PostProcessingResult(
            original_text=text,
            processed_text=text,
            task=task,
            success=False,
            error_message=f"Неизвестная задача: {task}",
        )

    # Auto-select model from config or use override.
    selected_model = model or _get_model_for_task(task)

    user_prompt = _build_user_prompt(text)

    try:
        result = _call_openai(
            user_prompt,
            system=system_prompt,
            model=selected_model,
        )

        if not result or not result.strip():
            return PostProcessingResult(
                original_text=text,
                processed_text=text,
                task=task,
                success=False,
                error_message="LLM вернул пустой результат.",
            )

        return PostProcessingResult(
            original_text=text,
            processed_text=result.strip(),
            task=task,
            success=True,
        )

    except Exception as e:
        logger.exception("Post-processing failed for task %s", task)
        return PostProcessingResult(
            original_text=text,
            processed_text=text,
            task=task,
            success=False,
            error_message=str(e),
        )

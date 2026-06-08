"""Build the Whisper `initial_prompt` glossary.

Whisper only consumes roughly the last 224 tokens of the prompt and uses it as
lexical bias (not as instructions), so we keep it short and, if it must be
trimmed, keep the *tail* (where the rarest, highest-value terms should live).
"""
from __future__ import annotations

# Conservative word budget; ~1.3 tokens/word keeps us under the 224-token window.
_MAX_WORDS = 160


def build_initial_prompt(text: str | None) -> str | None:
    if not text:
        return None
    words = text.split()
    if len(words) <= _MAX_WORDS:
        return " ".join(words)
    # Keep the tail: Whisper biases on the last tokens.
    return " ".join(words[-_MAX_WORDS:])

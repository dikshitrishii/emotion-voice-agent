"""Domain-specific exceptions.

Each maps to a specific HTTP status in main.py's exception handlers, so
service code raises intent ("audio too large") instead of generic
Exceptions that all collapse to an opaque 500.
"""


class VoiceAgentError(Exception):
    """Base class for all expected, handled errors in this service."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AudioTooLargeError(VoiceAgentError):
    status_code = 413


class InvalidAudioError(VoiceAgentError):
    status_code = 422


class TranscriptionError(VoiceAgentError):
    status_code = 502


class LLMError(VoiceAgentError):
    status_code = 502


class SynthesisError(VoiceAgentError):
    status_code = 502

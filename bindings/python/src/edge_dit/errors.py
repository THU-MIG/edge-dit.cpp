from __future__ import annotations

STATUS_OK = 0
STATUS_ERROR = 1
STATUS_INVALID_ARGUMENT = 2
STATUS_MODEL_LOAD_FAILED = 3
STATUS_GENERATION_FAILED = 4
STATUS_OUT_OF_MEMORY = 5
STATUS_UNSUPPORTED = 6


class EdgeDitError(RuntimeError):
    """Base error for edge-dit Python bindings."""


class EdgeDitClosedError(EdgeDitError):
    """Raised when an operation is attempted on a closed engine."""


class EdgeDitLibraryError(EdgeDitError):
    """Raised when the native library cannot be loaded."""


class InvalidArgumentError(EdgeDitError):
    """Raised for invalid user-provided arguments."""


class ModelLoadError(EdgeDitError):
    """Raised when the native engine cannot be created."""


class GenerationError(EdgeDitError):
    """Raised when image generation fails."""


class UnsupportedError(EdgeDitError):
    """Raised when a requested feature is unsupported."""


class UnsupportedImageFormatError(UnsupportedError):
    """Raised when the native output cannot be converted into a Python image."""


def exception_for_status(status: int, message: str) -> EdgeDitError:
    if status == STATUS_INVALID_ARGUMENT:
        return InvalidArgumentError(message)
    if status == STATUS_MODEL_LOAD_FAILED:
        return ModelLoadError(message)
    if status in (STATUS_GENERATION_FAILED, STATUS_OUT_OF_MEMORY):
        return GenerationError(message)
    if status == STATUS_UNSUPPORTED:
        return UnsupportedError(message)
    return EdgeDitError(message)


def get_last_error_message(lib: object, ctx: object | None) -> str | None:
    if lib is None or ctx is None:
        return None

    getter = getattr(lib, "ed_get_last_error", None)
    if getter is None:
        return None

    raw = getter(ctx)
    if not raw:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def raise_for_status(
    status: int,
    *,
    lib: object | None = None,
    ctx: object | None = None,
    default_message: str = "native edge-dit call failed",
) -> None:
    if status == STATUS_OK:
        return

    message = get_last_error_message(lib, ctx) or default_message
    raise exception_for_status(status, message)


from __future__ import annotations

import os


class CStringPool:
    """Keeps UTF-8 bytes alive for the duration of native calls."""

    def __init__(self) -> None:
        self._keepalive: list[bytes] = []

    def add_optional(self, value: str | bytes | os.PathLike[str] | None) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            encoded = value
        else:
            encoded = os.fspath(value).encode("utf-8")
        self._keepalive.append(encoded)
        return encoded


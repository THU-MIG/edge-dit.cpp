#!/usr/bin/env python3
"""Repository hygiene checks for public source releases.

The checker only scans Git-tracked files in the current checkout. It avoids
third-party source trees and never modifies repository contents.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_PREFIXES = (
    "third_party/",
    ".git/",
)

LARGE_TOKENIZER_HEADERS = (
    "src/dit_models/components/text_encoders/tokenizers/vocab/",
)

BINARY_SUFFIXES = {
    ".png": "test generated image",
    ".jpg": "test generated image",
    ".jpeg": "test generated image",
    ".webp": "test generated image",
    ".mp4": "test generated video",
    ".mov": "test generated video",
    ".webm": "test generated video",
    ".gif": "test generated image",
    ".zip": "large archive",
    ".tar": "large archive",
    ".tgz": "large archive",
    ".gz": "large archive",
    ".7z": "large archive",
    ".rar": "large archive",
    ".safetensors": "model weight",
    ".gguf": "model weight",
    ".ckpt": "model weight",
    ".pt": "model weight",
    ".pth": "model weight",
    ".onnx": "model weight",
    ".log": "log file",
    ".pyc": "Python cache",
    ".o": "build artifact",
    ".a": "build artifact",
    ".so": "build artifact",
    ".dylib": "build artifact",
    ".dll": "build artifact",
    ".exe": "build artifact",
}

SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |)PRIVATE KEY-----")),
    ("GitHub access token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("generic API key assignment", re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret[_-]?key)\b\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{20,}['\"]")),
    ("bearer token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_./+=-]{24,}\b")),
)

INTERNAL_PATH_PATTERNS = (
    ("internal absolute path /export/home", re.compile(r"/export/home/[A-Za-z0-9._-]+")),
    ("internal absolute path /root/autodl-tmp", re.compile(r"/root/autodl-tmp(?:/[^\s)\"']*)?")),
    ("personal /home path", re.compile(r"/home/(?!runner\b|github\b|user\b|ubuntu\b|linuxbrew\b)[A-Za-z0-9._-]+(?:/[^\s)\"']*)?")),
    ("personal conda path", re.compile(r"/(?:home|export/home)/[A-Za-z0-9._-]+/[^\s)\"']*(?:conda|anaconda|miniconda)[^\s)\"']*")),
)

HOSTNAME_PATTERN = re.compile(
    r"\b[A-Za-z0-9-]+\.(?:corp|internal|intranet|lan|localdomain|jd)(?:\.[A-Za-z0-9-]+)*\b",
    re.IGNORECASE,
)

PRIVATE_IP_PATTERN = re.compile(r"\b(?:10|172\.(?:1[6-9]|2[0-9]|3[0-1])|192\.168)\.(?:\d{1,3}\.){1,2}\d{1,3}\b")
MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


class Finding:
    def __init__(self, path: str, line: int, reason: str) -> None:
        self.path = path
        self.line = line
        self.reason = reason

    def __str__(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"{location}: {self.reason}"


def git_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if line]


def is_excluded(path: str) -> bool:
    return path.startswith(EXCLUDED_PREFIXES)


def is_large_tokenizer_header(path: str) -> bool:
    return path.startswith(LARGE_TOKENIZER_HEADERS) and path.endswith((".h", ".hpp"))


def check_path(path: str) -> Iterator[Finding]:
    parts = Path(path).parts
    if "node_modules" in parts:
        yield Finding(path, 0, "node_modules should not be tracked")
    if "__pycache__" in parts:
        yield Finding(path, 0, "Python cache should not be tracked")
    suffix = Path(path).suffix.lower()
    if suffix in BINARY_SUFFIXES and not is_large_tokenizer_header(path):
        yield Finding(path, 0, f"{BINARY_SUFFIXES[suffix]} should not be tracked")


def valid_private_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_private
    except ValueError:
        return False


def check_text_file(root: Path, path: str) -> Iterator[Finding]:
    full_path = root / path
    try:
        text = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    for line_no, line in enumerate(text.splitlines(), 1):
        for reason, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                yield Finding(path, line_no, reason)
        for reason, pattern in INTERNAL_PATH_PATTERNS:
            if pattern.search(line):
                yield Finding(path, line_no, reason)
        for match in HOSTNAME_PATTERN.finditer(line):
            token = match.group(0)
            if token not in {"localhost"}:
                yield Finding(path, line_no, f"possible internal hostname: {token}")
        for match in PRIVATE_IP_PATTERN.finditer(line):
            if valid_private_ip(match.group(0)):
                yield Finding(path, line_no, f"private IP address: {match.group(0)}")

        if path.endswith((".md", ".markdown")):
            yield from check_markdown_links(root, path, line_no, line)


def check_markdown_links(root: Path, path: str, line_no: int, line: str) -> Iterator[Finding]:
    base = root / path
    for match in MARKDOWN_LINK_PATTERN.finditer(line):
        target = match.group(1).strip()
        if (
            not target
            or target.startswith(("#", "http://", "https://", "mailto:", "tel:"))
            or "://" in target
        ):
            continue
        target_path = target.split("#", 1)[0]
        if not target_path:
            continue
        candidate = (base.parent / target_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            yield Finding(path, line_no, f"Markdown link escapes repository: {target}")
            continue
        if not candidate.exists():
            yield Finding(path, line_no, f"broken relative Markdown link: {target}")


def check_submodules(root: Path) -> Iterator[Finding]:
    if not (root / "third_party/ggml/CMakeLists.txt").is_file():
        yield Finding("third_party/ggml", 0, "missing submodule; run git submodule update --init --recursive")


def run(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in git_files(root):
        if is_excluded(path):
            continue
        findings.extend(check_path(path))
        findings.extend(check_text_file(root, path))
    findings.extend(check_submodules(root))
    return findings


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    args = parser.parse_args(argv)

    findings = run(args.root.resolve())
    if findings:
        for finding in findings:
            print(finding)
        return 1
    print("repository hygiene scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

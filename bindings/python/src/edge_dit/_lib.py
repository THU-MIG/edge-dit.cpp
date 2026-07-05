from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Iterable

from .errors import EdgeDitLibraryError

EDGE_DIT_LIBRARY_ENV = "EDGE_DIT_LIBRARY"
EDGE_DIT_DEPENDENCY_DIRS_ENV = "EDGE_DIT_DEPENDENCY_DIRS"


def _library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("edgedit.dll",)
    if os.name == "posix" and os.uname().sysname == "Darwin":
        return ("libedgedit.dylib",)
    return ("libedgedit.so",)


def _repo_root() -> Path | None:
    path = Path(__file__).resolve()
    parents = path.parents
    if len(parents) < 5:
        return None
    return parents[4]


def _iter_nvidia_python_lib_dirs(base_dir: Path) -> Iterable[Path]:
    site_packages = base_dir.glob("python*/site-packages/nvidia/*/lib")
    for path in sorted(site_packages):
        if path.is_dir():
            yield path


def _common_dependency_dirs() -> list[Path]:
    candidates: list[Path] = []

    env_dirs = os.environ.get(EDGE_DIT_DEPENDENCY_DIRS_ENV)
    if env_dirs:
        for raw in env_dirs.split(os.pathsep):
            if raw:
                candidates.append(Path(raw).expanduser())

    for env_name in ("CUDNN_ROOT", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_name)
        if value:
            root = Path(value).expanduser()
            candidates.extend((root, root / "lib", root / "lib64", root / "targets/x86_64-linux/lib"))

    home = Path.home()
    candidates.extend(_iter_nvidia_python_lib_dirs(home / ".local" / "lib"))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(_iter_nvidia_python_lib_dirs(Path(conda_prefix) / "lib"))

    candidates.append(Path("/usr/local/cuda/lib64"))
    candidates.append(Path("/usr/local/cuda/targets/x86_64-linux/lib"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def candidate_library_paths() -> list[Path]:
    candidates: list[Path] = []

    env_value = os.environ.get(EDGE_DIT_LIBRARY_ENV)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    repo_root = _repo_root()
    if repo_root is not None:
        build_dirs = sorted(repo_root.glob("build*"))
        for build_dir in build_dirs:
            if not build_dir.is_dir():
                continue
            for name in _library_names():
                candidates.append(build_dir / name)
                candidates.append(build_dir / "lib" / name)
                candidates.append(build_dir / "bin" / name)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _load_cdll(path: Path) -> ctypes.CDLL:
    mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(ctypes, "RTLD_LOCAL", 0)
    return ctypes.CDLL(str(path), mode=mode)


def _dependency_search_dirs(library_path: Path) -> list[Path]:
    candidates = [library_path.parent]
    candidates.extend(_common_dependency_dirs())

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _preload_dependency_libraries(library_path: Path) -> None:
    patterns = (
        "libcudart.so*",
        "libnvrtc.so*",
        "libcudnn*.so*",
        "libggml-base.so*",
        "libggml-cpu.so*",
        "libggml-cuda.so*",
        "libggml.so*",
    )

    loaded: set[str] = set()
    for directory in _dependency_search_dirs(library_path):
        for pattern in patterns:
            for candidate in sorted(directory.glob(pattern)):
                key = str(candidate.resolve())
                if key in loaded or not candidate.is_file():
                    continue
                try:
                    _load_cdll(candidate)
                except OSError:
                    continue
                loaded.add(key)


def load_library(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    errors: list[str] = []

    if path is not None:
        explicit_path = Path(path).expanduser()
        try:
            _preload_dependency_libraries(explicit_path)
            return _load_cdll(explicit_path)
        except OSError as exc:
            raise EdgeDitLibraryError(
                f"failed to load edge-dit library from {explicit_path}: {exc}"
            ) from exc

    for candidate in candidate_library_paths():
        try:
            _preload_dependency_libraries(candidate)
            return _load_cdll(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    found_name = ctypes.util.find_library("edgedit")
    if found_name:
        try:
            return ctypes.CDLL(found_name, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError as exc:
            errors.append(f"{found_name}: {exc}")

    hint_lines = "\n".join(f"  - {item}" for item in errors[:5])
    if hint_lines:
        hint_lines = f"\nTried:\n{hint_lines}"

    extra_hint = ""
    if any("GLIBCXX_" in item for item in errors):
        extra_hint = (
            "\nDetected a libstdc++ ABI mismatch while loading native dependencies. "
            "If you are inside Conda, try /usr/bin/python3 or use a newer libstdc++ runtime."
        )

    raise EdgeDitLibraryError(
        "Could not load libedgedit. "
        f"Set {EDGE_DIT_LIBRARY_ENV}=/absolute/path/to/libedgedit.so "
        f"and optionally {EDGE_DIT_DEPENDENCY_DIRS_ENV}=/path/one:/path/two "
        "or install a wheel that bundles the native library."
        f"{extra_hint}"
        f"{hint_lines}"
    )

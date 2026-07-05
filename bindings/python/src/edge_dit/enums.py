from __future__ import annotations

from .errors import InvalidArgumentError


def _normalize_enum_name(value: str) -> str:
    normalized = value.strip().lower()
    for old, new in (("_", "-"), (".", "-"), (" ", "-")):
        normalized = normalized.replace(old, new)
    return normalized


DTYPE_VALUES = {
    "auto": -1,
    "f32": 0,
    "f16": 1,
    "bf16": 30,
    "q4-0": 2,
    "q4-1": 3,
    "q5-0": 6,
    "q5-1": 7,
    "q8-0": 8,
    "q2-k": 10,
    "q3-k": 11,
    "q4-k": 12,
    "q5-k": 13,
    "q6-k": 14,
}

SAMPLER_VALUES = {
    "auto": -1,
    "euler": 0,
    "euler-a": 1,
    "heun": 2,
    "dpm2": 3,
    "dpm-plus-plus-2s-a": 4,
    "dpm++-2s-a": 4,
    "dpm-plus-plus-2m": 5,
    "dpm++-2m": 5,
    "dpm-plus-plus-2m-v2": 6,
    "dpm++-2m-v2": 6,
    "ipndm": 7,
    "ipndm-v": 8,
    "lcm": 9,
    "ddim-trailing": 10,
    "ddim": 10,
    "tcd": 11,
    "res-multistep": 12,
    "res-2s": 13,
    "er-sde": 14,
}

SCHEDULER_VALUES = {
    "auto": -1,
    "discrete": 0,
    "karras": 1,
    "exponential": 2,
    "ays": 3,
    "gits": 4,
    "sgm-uniform": 5,
    "simple": 6,
    "smoothstep": 7,
    "kl-optimal": 8,
    "lcm": 9,
    "bong-tangent": 10,
}

CACHE_MODE_VALUES = {
    "disabled": 0,
    "disable": 0,
    "off": 0,
    "none": 0,
    "0": 0,
    "easycache": 1,
    "easy": 1,
    "ucache": 2,
    "u": 2,
    "dbcache": 3,
    "db": 3,
    "taylorseer": 4,
    "taylor-seer": 4,
    "taylor": 4,
    "cache-dit": 5,
    "cachedit": 5,
}


def _resolve_enum(mapping: dict[str, int], value: int | str, label: str) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, str) or not value.strip():
        raise InvalidArgumentError(f"{label} must be a non-empty string or integer")

    normalized = _normalize_enum_name(value)
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise InvalidArgumentError(f"unsupported {label}: {value}") from exc


def resolve_dtype(value: int | str) -> int:
    return _resolve_enum(DTYPE_VALUES, value, "weight_type")


def resolve_sampler(value: int | str) -> int:
    return _resolve_enum(SAMPLER_VALUES, value, "sampler")


def resolve_scheduler(value: int | str) -> int:
    return _resolve_enum(SCHEDULER_VALUES, value, "scheduler")


def resolve_cache_mode(value: int | str) -> int:
    return _resolve_enum(CACHE_MODE_VALUES, value, "cache_mode")


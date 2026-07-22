#!/usr/bin/env python3
"""Build the stable-diffusion.cpp C API e2e benchmark wrapper.

This helper links against an already configured stable-diffusion.cpp build tree
so the benchmark harness can use the C API without reporting process-level
sd-cli timings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd-cpp-repo", type=Path, required=True)
    parser.add_argument("--sd-cpp-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.sd_cpp_repo.resolve()
    build = args.sd_cpp_build.resolve()
    output = args.output.resolve()
    source = Path(__file__).resolve().with_name("sd_cpp_e2e.cpp")
    link_txt = build / "examples" / "cli" / "CMakeFiles" / "sd-cli.dir" / "link.txt"
    stable_lib = build / "libstable-diffusion.a"

    if not source.exists():
        raise SystemExit(f"missing wrapper source: {source}")
    if not link_txt.exists() or not stable_lib.exists():
        raise SystemExit(
            "stable-diffusion.cpp build is incomplete; build the sd-cli target first"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    object_path = output.parent / "sd_cpp_e2e.o"
    compile_cmd = [
        "/usr/bin/c++",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-fPIE",
        "-I",
        str(repo / "include"),
        "-I",
        str(repo / "examples" / "common"),
        "-c",
        str(source),
        "-o",
        str(object_path),
    ]
    subprocess.run(compile_cmd, check=True)

    tokens = shlex.split(link_txt.read_text(encoding="utf-8").strip())
    try:
        output_flag = tokens.index("-o")
    except ValueError as exc:
        raise SystemExit(f"could not parse linker command: {link_txt}") from exc
    libraries = tokens[output_flag + 2 :]
    if not any("libstable-diffusion.a" in token for token in libraries):
        raise SystemExit(f"linker command does not include stable-diffusion library: {link_txt}")

    # Object files listed before "-o" contain the CLI helpers our wrapper depends on
    # (media_io.cpp.o -> load_sd_image_from_file / create_video_from_sd_images,
    # common.cpp.o, log.cpp.o, image_metadata.cpp.o, zip.c.o). We must link them in,
    # but drop main.cpp.o because our wrapper provides its own main().
    helper_objects = [
        token
        for token in tokens[1:output_flag]
        if (token.endswith(".o") and "main.cpp.o" not in token)
    ]

    linker = tokens[0]
    link_cmd = [
        linker,
        "-O3",
        "-DNDEBUG",
        "-no-pie",
        str(object_path),
        *helper_objects,
        "-o",
        str(output),
        *libraries,
    ]
    subprocess.run(link_cmd, cwd=build / "examples" / "cli", check=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

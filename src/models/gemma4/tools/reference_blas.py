"""Build an isolated BLAS reference DLL from the pinned Windows Ninja build.

Only the BLAS dispatch and feature-reporting objects are recompiled. An isolated
dispatch overlay honors explicit GGML_PREC_F32 before the fork's BF16 WMMA path,
which otherwise ignores that precision request at batches >=512. All numerical
operators remain llama/ggml implementations. Other objects and dependencies
retain the reference build's compiler flags. No existing source/build is changed.
Keep this arithmetic-matched oracle separate from performance comparisons.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def fields(lines, start):
    result = {}
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        key, value = line.strip().split(" = ", 1)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--llama-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-tool", type=Path, required=True)
    args = parser.parse_args()
    build, output = args.llama_build.resolve(), args.output.resolve()
    if output == build or build in output.parents:
        raise ValueError("output must be outside the existing reference build")
    output.mkdir(parents=True, exist_ok=True)
    source = args.llama_source.resolve()
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code",
                    "1d13fa1c5d6ed9f8cfdc924744b204070c32466d", "--", "src", "include", "ggml"], check=True)
    lines = (build / "build.ninja").read_text().splitlines()
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    compiler = next(line.split("=", 1)[1] for line in cache if line.startswith("CMAKE_CXX_COMPILER:"))
    original_source = next(line.split("=", 1)[1] for line in cache if line.startswith("CMAKE_HOME_DIRECTORY:"))
    replacements = {}
    commands = []
    for name in ("ggml-cuda", "mmq"):
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("build ") and f"/{name}.cu.obj:" in line)
        line = lines[start]
        original = line.split(": CXX_", 1)[0][6:]
        settings = fields(lines, start)
        obj = output / f"{name}.cu.obj"
        unit = source / "ggml/src/ggml-cuda" / f"{name}.cu"
        if name == "ggml-cuda":
            original_text = unit.read_text(encoding="utf-8")
            signature = "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {"
            if original_text.count(signature) != 1:
                raise ValueError("pinned precision dispatch location changed")
            overlay = signature + "\n#ifdef GGML_CUDA_FORCE_CUBLAS\n    if (dst->op_params[0] == GGML_PREC_F32) {\n        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);\n        return;\n    }\n#endif"
            unit = output / "ggml-cuda.cu"
            unit.write_text(original_text.replace(signature, overlay), encoding="utf-8")
        command = [compiler, "-I" + str(Path(__file__).resolve().parents[4] / "compat/win32/hip_include"), *shlex.split(settings["DEFINES"]),
                   "-I" + str(source / "ggml/src/ggml-cuda"),
                   *shlex.split(settings["INCLUDES"].replace(original_source, source.as_posix())), *shlex.split(settings["FLAGS"]),
                   "-DGGML_CUDA_FORCE_CUBLAS", "-o", str(obj), "-c",
                   str(unit)]
        commands.append(command)
        subprocess.run(command, check=True)
        replacements[original] = str(obj)
    start = next(i for i, line in enumerate(lines) if line.startswith("build bin/ggml-hip.dll "))
    settings = fields(lines, start)
    objects = lines[start].split(": CXX_SHARED_LIBRARY_LINKER__ggml-hip_Release ", 1)[1].split(" | ", 1)[0].split()
    objects = [replacements.get(item, str(build / item)) for item in objects]
    libraries = shlex.split(settings["LINK_LIBRARIES"])
    libraries = [str(build / item) if item.endswith(".lib") and not Path(item).is_absolute() else item for item in libraries]
    command = [compiler, "-nostartfiles", "-nostdlib", *shlex.split(settings["LANGUAGE_COMPILE_FLAGS"]),
               *shlex.split(settings["LINK_FLAGS"]), "-o", str(output / "ggml-hip.dll"),
               "-Xlinker", "/MANIFEST:EMBED", "-Xlinker", f"/implib:{output / 'ggml-hip.lib'}",
               *objects, *libraries]
    commands.append(command)
    subprocess.run(command, check=True)
    for dll in (build / "bin").glob("*.dll"):
        if dll.name != "ggml-hip.dll":
            shutil.copy2(dll, output / dll.name)
    shutil.copy2(args.reference_tool, output / "llama-reference.exe")
    (output / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(f"Built isolated BLAS numerical reference at {output}")


if __name__ == "__main__":
    main()

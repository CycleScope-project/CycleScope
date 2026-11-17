#!/usr/bin/env python3
"""Build vendored FA3 SM90 flash_attn_3._C from flash_api.cpp (traditional TORCH_LIBRARY).

Official hopper/setup.py switches to flash_api_stable.cpp on torch>=2.9 because
that path uses py_limited_api.  This builder keeps flash_api.cpp and compiles it
as a regular torch cpp extension so flash_attn_interface.py can import
flash_attn_3._C and dispatch torch.ops.flash_attn_3.  SM8x TUs are omitted;
split/paged/append/local/softcap/pack/varlen/backward stay enabled.

Compile is sharded: 4 ninja jobs write distinct objects into a shared artifact
directory, then a separate link step produces flash_attn_3/_C.so.
cpu_simulation jobs have GPU0; nvcc is pinned to sm90a without device query.
Default --workers 4 matches that 32CPU profile.
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
HOPPER_ROOT = SUITE_ROOT / "upstream" / "hopper"
CUTLASS_INCLUDE = SUITE_ROOT / "upstream" / "cutlass" / "include"
GLOBAL_TIMER_CU = REPO_ROOT / "delivery_tests" / "common" / "globaltimer_timer.cu"
DEFAULT_BUILD_ROOT = Path(
    os.environ.get(
        "FLASHATTENTION3_FULL_GPU_BUILD_ROOT",
        "/path/to/hopper-prototype/artifacts/fa3_full_gpu_100_20260920",
    )
)
SOURCE_REVISION = "98eb7998a0eba4047c7a30375522569d4b8efb20"
SHARD_COUNT = 4
FEATURE_MACRO = "-DFLASHATTENTION_DISABLE_SM8x"
SM90A_GENCODE = ["-gencode", "arch=compute_90a,code=sm_90a"]

# CPU compile shards have GPU0. Pin arch before any torch import can probe.
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"




def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def official_sm90_sources() -> list[str]:
    """Same product as hopper/setup.py with DISABLE_SM8x and flash_api.cpp forced."""

    head_bwd = [64, 96, 128, 192, 256]
    head_diff64 = ["64_256", "64_512"]
    head_diff192 = ["192_128"]
    dtype_fwd = ["bf16", "fp16", "e4m3"]
    half_fwd = ["bf16", "fp16"]
    dtype_bwd = ["bf16", "fp16"]
    split = ["", "_split"]
    paged = ["", "_paged"]
    softcap = ["", "_softcap"]
    softcap_all = ["_softcapall"]
    pack = ["", "_packgqa"]
    fwd = [
        f"instantiations/flash_fwd_hdim{hdim}_{dtype}{page}{spl}{cap}{gqa}_sm90.cu"
        for hdim, dtype, spl, page, cap, gqa in itertools.product(
            head_bwd, dtype_fwd, split, paged, softcap, pack
        )
        if not (gqa and (page or spl))
    ]
    fwd += [
        f"instantiations/flash_fwd_hdim{hdim}_{dtype}{page}{spl}{cap}{gqa}_sm90.cu"
        for hdim, dtype, spl, page, cap, gqa in itertools.product(
            head_diff64, half_fwd, split, paged, softcap, pack
        )
        if not (gqa and (page or spl))
    ]
    fwd += [
        f"instantiations/flash_fwd_hdim{hdim}_{dtype}{page}{spl}{cap}{gqa}_sm90.cu"
        for hdim, dtype, spl, page, cap, gqa in itertools.product(
            head_diff192, dtype_fwd, split, paged, softcap, pack
        )
        if not (gqa and (page or spl))
    ]
    bwd = [
        f"instantiations/flash_bwd_hdim{hdim}_{dtype}{cap}_sm90.cu"
        for hdim, dtype, cap in itertools.product(head_bwd, dtype_bwd, softcap_all)
    ]
    return ["flash_api.cpp", *fwd, *bwd, "flash_fwd_combine.cu", "flash_prepare_scheduler.cu"]


def resolve_sources() -> list[Path]:
    missing = []
    resolved = []
    for rel in official_sm90_sources():
        path = HOPPER_ROOT / rel
        if not path.is_file():
            missing.append(str(path))
        else:
            resolved.append(path)
    if missing:
        die("missing FA3 sources:\n" + "\n".join(missing[:20]))
    if not CUTLASS_INCLUDE.is_dir():
        die(f"missing vendored CUTLASS include: {CUTLASS_INCLUDE}")
    return resolved


def shard_sources(sources: list[Path]) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(SHARD_COUNT)]
    for index, source in enumerate(sources):
        buckets[index % SHARD_COUNT].append(source)
    return buckets


def object_path(objects_dir: Path, source: Path) -> Path:
    suffix = ".cuda.o" if source.suffix == ".cu" else ".o"
    return objects_dir / f"{source.stem}{suffix}"


def write_config(site: Path) -> Path:
    config = {
        "build_flags": {
            "FLASHATTENTION_DISABLE_BACKWARD": False,
            "FLASHATTENTION_DISABLE_SPLIT": False,
            "FLASHATTENTION_DISABLE_PAGEDKV": False,
            "FLASHATTENTION_DISABLE_APPENDKV": False,
            "FLASHATTENTION_DISABLE_LOCAL": False,
            "FLASHATTENTION_DISABLE_SOFTCAP": False,
            "FLASHATTENTION_DISABLE_PACKGQA": False,
            "FLASHATTENTION_DISABLE_FP16": False,
            "FLASHATTENTION_DISABLE_FP8": False,
            "FLASHATTENTION_DISABLE_VARLEN": False,
            "FLASHATTENTION_DISABLE_CLUSTER": False,
            "FLASHATTENTION_DISABLE_HDIM64": False,
            "FLASHATTENTION_DISABLE_HDIM96": False,
            "FLASHATTENTION_DISABLE_HDIM128": False,
            "FLASHATTENTION_DISABLE_HDIM192": False,
            "FLASHATTENTION_DISABLE_HDIM256": False,
            "FLASHATTENTION_DISABLE_SM8x": True,
            "FLASHATTENTION_ENABLE_VCOLMAJOR": False,
            "FLASH_ATTENTION_DISABLE_HDIMDIFF64": False,
            "FLASH_ATTENTION_DISABLE_HDIMDIFF192": False,
        }
    }
    path = site / "flash_attn_config.py"
    path.write_text(
        "# Generated by suites/flashattention3/harness/build_full_gpu.py\n"
        f"CONFIG = {repr(config)}\n\n"
        "def show():\n"
        "    from pprint import pprint\n"
        "    pprint(CONFIG)\n",
        encoding="utf-8",
    )
    return path


def install_python_package(site: Path) -> None:
    pkg = site / "flash_attn_3"
    pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HOPPER_ROOT / "flash_attn_interface.py", site / "flash_attn_interface.py")
    shutil.copy2(HOPPER_ROOT / "flash_attn_3" / "__init__.py", pkg / "__init__.py")
    shutil.copy2(
        HOPPER_ROOT / "flash_attn_3" / "flash_attn_interface.py",
        pkg / "flash_attn_interface.py",
    )
    (pkg / "flash_attn_config.py").write_text(
        "from flash_attn_config import *  # noqa: F403\n", encoding="utf-8"
    )
    write_config(site)




def extension_flags() -> tuple[list[str], list[str], list[str]]:
    import torch
    from torch.utils.cpp_extension import (
        COMMON_NVCC_FLAGS,
        CUDA_HOME,
        include_paths,
        _prepare_ldflags,
    )

    cxx11 = bool(torch.compiled_with_cxx11_abi())
    cuda_home = Path(CUDA_HOME or os.environ.get("CUDA_HOME") or "/usr/local/cuda")
    nvcc = cuda_home / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ.setdefault("PYTORCH_NVCC", str(nvcc))
        os.environ["PATH"] = str(nvcc.parent) + os.pathsep + os.environ.get("PATH", "")
    user_includes = [str(HOPPER_ROOT), str(CUTLASS_INCLUDE)]
    # Same env var as upstream/hopper/setup.py: FA3_DELIVERY_MARKER_OVERRIDE=1 prepends the
    # harness/override device_kernel.h marker override ahead of CUTLASS on the include path.
    if os.environ.get("FA3_DELIVERY_MARKER_OVERRIDE") == "1":
        user_includes.insert(0, str(SUITE_ROOT / "harness" / "override"))

    cccl = cuda_home / "include" / "cccl"
    if cccl.is_dir():
        user_includes.append(str(cccl))
    system_includes = [str(path) for path in include_paths(device_type="cuda")]
    python_include = sysconfig.get_path("include", scheme="posix_prefix")
    if python_include:
        system_includes.append(python_include)
    common = [
        "-DTORCH_EXTENSION_NAME=_C",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        *[f"-I{path}" for path in user_includes],
        *[f"-isystem {path}" for path in system_includes],
        f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx11)}",
        FEATURE_MACRO,
    ]
    wheel_includes = sorted((Path(torch.__file__).resolve().parent.parent / "nvidia").glob("*/include"))
    cxx = [*common, *[f"-isystem {path}" for path in wheel_includes], "-fPIC", "-std=c++17", "-O3"]
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    cuda = [
        *common,
        *COMMON_NVCC_FLAGS,
        "--compiler-options",
        "'-fPIC'",
        "-O3",
        "-std=c++17",
        "--ftemplate-backtrace-limit=0",
        "--use_fast_math",
        "--resource-usage",
        "-lineinfo",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",
        *SM90A_GENCODE,
        "--threads",
        nvcc_threads,
        FEATURE_MACRO,
    ]
    ldflags = _prepare_ldflags([], with_cuda=True, with_sycl=False, verbose=True, is_standalone=False)
    torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
    if not any(flag.startswith("-Wl,-rpath,") for flag in ldflags):
        ldflags.append(f"-Wl,-rpath,{torch_lib}")
    return cxx, cuda, ldflags


def compile_shard(build_root: Path, shard: int, workers: int) -> list[Path]:
    sources = shard_sources(resolve_sources())[shard]
    objects_dir = build_root / "objects"
    shard_dir = build_root / "shards" / str(shard)
    objects_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    objects = [object_path(objects_dir, source) for source in sources]
    cxx, cuda, _ = extension_flags()
    os.environ["MAX_JOBS"] = str(workers)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from torch.utils.cpp_extension import _write_ninja_file_and_compile_objects

    _write_ninja_file_and_compile_objects(
        sources=[str(path) for path in sources],
        objects=[str(path) for path in objects],
        cflags=cxx,
        post_cflags=None,
        cuda_cflags=cuda,
        cuda_post_cflags=None,
        cuda_dlink_post_cflags=None,
        sycl_cflags=None,
        sycl_post_cflags=None,
        sycl_dlink_post_cflags=None,
        build_directory=str(shard_dir),
        verbose=True,
        with_cuda=True,
        with_sycl=False,
    )
    missing = [str(path) for path in objects if not path.is_file()]
    if missing:
        die(f"shard {shard} did not produce objects:\n" + "\n".join(missing[:20]))
    (shard_dir / "objects.json").write_text(
        json.dumps({"shard": shard, "objects": [str(path) for path in objects]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return objects


def link_extension(build_root: Path) -> Path:
    from torch.utils.cpp_extension import get_cxx_compiler

    sources = resolve_sources()
    objects = [object_path(build_root / "objects", source) for source in sources]
    missing = [str(path) for path in objects if not path.is_file()]
    if missing:
        die(f"link missing {len(missing)} objects; run all four shards first:\n" + "\n".join(missing[:20]))
    _, _, ldflags = extension_flags()
    site = build_root / "site-packages"
    pkg = site / "flash_attn_3"
    pkg.mkdir(parents=True, exist_ok=True)
    target = pkg / "_C.so"
    cmd = [get_cxx_compiler(), *[str(path) for path in objects], "-shared", *ldflags, "-o", str(target)]
    print("link:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    if not target.is_file():
        die(f"link produced no {target}")
    tagged = pkg / f"_C{sysconfig.get_config_var('EXT_SUFFIX') or '.so'}"
    if tagged != target:
        shutil.copy2(target, tagged)
    return target


def compile_globaltimer(build_root: Path) -> Path:
    if not GLOBAL_TIMER_CU.is_file():
        die(f"missing globaltimer source: {GLOBAL_TIMER_CU}")
    target = build_root / "libdelivery_globaltimer_timer.so"
    cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-arch=sm_90a",
        "-shared",
        "-Xcompiler=-fPIC",
        str(GLOBAL_TIMER_CU),
        "-o",
        str(target),
    ]
    print("globaltimer:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    if not target.is_file():
        die(f"globaltimer build produced no {target}")
    return target


def write_manifest(build_root: Path, sources: list[Path]) -> Path:
    shards = shard_sources(sources)
    payload = {
        "flash_api": "flash_api.cpp",
        "disable_sm8x": True,
        "disabled_features": [],
        "source_revision": SOURCE_REVISION,
        "source_count": len(sources),
        "shard_count": SHARD_COUNT,
        "workers_per_shard": 4,
        "objects_dir": str(build_root / "objects"),
        "site_packages": str(build_root / "site-packages"),
        "shards": [
            {
                "shard": index,
                "source_count": len(group),
                "sources": [str(path.relative_to(HOPPER_ROOT)) for path in group],
            }
            for index, group in enumerate(shards)
        ],
    }
    path = build_root / "fa3_full_gpu_sources.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (build_root / "source_revision.txt").write_text(SOURCE_REVISION + "\n", encoding="utf-8")
    return path


def prepare(build_root: Path) -> dict:
    sources = resolve_sources()
    build_root.mkdir(parents=True, exist_ok=True)
    (build_root / "objects").mkdir(parents=True, exist_ok=True)
    (build_root / "shards").mkdir(parents=True, exist_ok=True)
    site = build_root / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    install_python_package(site)
    manifest = write_manifest(build_root, sources)
    return {
        "status": "prepared",
        "build_root": str(build_root),
        "source_count": len(sources),
        "flash_api": "flash_api.cpp",
        "disable_sm8x": True,
        "manifest": str(manifest),
        "shard_commands": [
            f"python3 {Path(__file__).resolve()} --shard {index} --workers 4 --build-root {build_root}"
            for index in range(SHARD_COUNT)
        ],
        "link_command": f"python3 {Path(__file__).resolve()} --link --build-root {build_root}",
    }


def ensure_prepared(build_root: Path) -> None:
    marker = build_root / "site-packages" / "flash_attn_config.py"
    if marker.is_file():
        return
    build_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(build_root / ".prepare.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not marker.is_file():
            prepare(build_root)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full SM90 FA3 extension (flash_api.cpp, no feature stubs) via 4 object shards."
    )
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--shard", type=int, choices=range(SHARD_COUNT), metavar="N")
    parser.add_argument("--link", action="store_true")
    parser.add_argument("--all", action="store_true", help="prepare, four sequential shards, then link")
    parser.add_argument("--workers", type=int, default=4, help="ninja -j per shard; 4 fits cpu_simulation 32CPU")
    parser.add_argument("--list-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        die("--workers must be >= 1")
    if args.list_sources:
        sources = resolve_sources()
        print(
            json.dumps(
                {
                    "source_count": len(sources),
                    "flash_api": "flash_api.cpp",
                    "disable_sm8x": True,
                    "shards": [
                        [str(path.relative_to(HOPPER_ROOT)) for path in group]
                        for group in shard_sources(sources)
                    ],
                },
                indent=2,
            )
        )
        return
    if not (args.prepare or args.shard is not None or args.link or args.all):
        die("specify --prepare, --shard N, --link, --all, or --list-sources", 2)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    build_root = args.build_root.expanduser().resolve()
    result: dict = {}
    if args.all or args.prepare:
        result = prepare(build_root)
    if args.all:
        for shard in range(SHARD_COUNT):
            objects = compile_shard(build_root, shard, args.workers)
            print(json.dumps({"shard": shard, "object_count": len(objects)}), file=sys.stderr)
        so = link_extension(build_root)
        timer = compile_globaltimer(build_root)
        result.update(status="built", extension=str(so), globaltimer=str(timer))
    else:
        if args.shard is not None:
            ensure_prepared(build_root)
            objects = compile_shard(build_root, args.shard, args.workers)
            result = {
                "status": "shard_compiled",
                "shard": args.shard,
                "object_count": len(objects),
                "objects_dir": str(build_root / "objects"),
            }
        if args.link:
            so = link_extension(build_root)
            timer = compile_globaltimer(build_root)
            result = {
                "status": "linked",
                "extension": str(so),
                "globaltimer": str(timer),
                "source_count": len(resolve_sources()),
            }
            (build_root / "build_ok.json").write_text(
                json.dumps(
                    {
                        **result,
                        "flash_api": "flash_api.cpp",
                        "source_revision": SOURCE_REVISION,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

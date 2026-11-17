#!/usr/bin/env python3
"""Copy and host-patch the pinned FlashInfer site for effective SM sizing."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = Path("/path/to/hopper-prototype/artifacts")
V = ART / "rebuild/c450v2"
FAR_ENV = REPO / "scripts/pipeline/far_pod_env.sh"
DEST = V / "build/fi_site_effsm"
MANIFEST = "EFFSM_PATCH.json"
IN_PROGRESS = ".EFFSM_PATCH_IN_PROGRESS.json"
PATCH_VERSION = 2
SCAN_ROOTS = (
    "flashinfer",
    "flashinfer/data/include",
    "flashinfer/data/csrc",
    "flashinfer/data/cutlass",
)
SOURCE_FILE = "flashinfer/utils.py"
UTILS_CUH = "flashinfer/data/include/flashinfer/utils.cuh"
CUTLASS_HW = "flashinfer/data/cutlass/include/cutlass/kernel_hardware_info.h"
CPP_SUFFIXES = {
    ".c", ".h", ".hh", ".hpp", ".hxx", ".inl", ".cuh", ".cc", ".cpp", ".cu",
}
SM_READ = re.compile(
    r"cudaDevAttrMultiProcessorCount|multiProcessorCount|"
    r"multi_processor_count|query_device_multiprocessor_count|"
    r"get_device_sm_count"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_site_from_env(path):
    text = path.read_text()
    match = re.search(r'^FI_SITE="([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise ValueError(f"FI_SITE assignment missing from {path}")
    return Path(match.group(1))


def replace_exact(text, before, after, path):
    count = text.count(before)
    if count != 1:
        raise ValueError(
            f"{path}: expected one patch anchor, found {count}: {before!r}")
    return text.replace(before, after, 1)


def inject_include(text, include):
    lines = text.splitlines(keepends=True)
    directive = f'#include "{include}"'
    lines = [line for line in lines if line.strip() != directive]
    depth = 0
    index = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#endif"):
            depth = max(0, depth - 1)
        if depth == 0 and stripped.startswith("#include"):
            index = i
            break
        if stripped.startswith(("#if ", "#ifdef ", "#ifndef ")):
            depth += 1
    if index is None:
        guard_index = None
        for i, line in enumerate(lines[:40]):
            match = re.match(r"\s*#ifndef\s+([A-Za-z_]\w*)", line)
            if match:
                guard = match.group(1)
                for j in range(i + 1, min(i + 5, len(lines))):
                    if re.match(rf"\s*#define\s+{re.escape(guard)}(?:\s|$)",
                                lines[j]):
                        guard_index = j + 1
                        break
                if guard_index is not None:
                    break
        if guard_index is not None:
            index = guard_index
        else:
            index = next((i for i, line in enumerate(lines)
                          if line.lstrip().startswith("#if")), 0)
    lines.insert(index, directive + "\n")
    return "".join(lines)


def patch_python_utils(path):
    text = path.read_text()
    if "HOPPER_EFFECTIVE_SM_COUNT_PATCH = True" in text:
        return text
    anchor = "logger = logging.getLogger(__name__)\n"
    patch = '''\
logger = logging.getLogger(__name__)


def _hopper_effective_sm_count():
    try:
        count = int(os.environ.get("FLASHINFER_EFFECTIVE_SM_COUNT", "0"))
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


class _HopperEffectiveDeviceProperties:
    def __init__(self, properties, sm_count):
        self._properties = properties
        self.multi_processor_count = sm_count

    def __getattr__(self, name):
        return getattr(self._properties, name)


if not getattr(torch.cuda.get_device_properties,
               "_hopper_effective_sm_count_patch", False):
    _hopper_get_device_properties = torch.cuda.get_device_properties

    def _hopper_get_device_properties_patched(device=None):
        properties = _hopper_get_device_properties(device)
        sm_count = _hopper_effective_sm_count()
        if sm_count is None:
            return properties
        return _HopperEffectiveDeviceProperties(properties, sm_count)

    _hopper_get_device_properties_patched._hopper_effective_sm_count_patch = True
    torch.cuda.get_device_properties = _hopper_get_device_properties_patched

HOPPER_EFFECTIVE_SM_COUNT_PATCH = True
'''
    if anchor in text:
        text = replace_exact(text, anchor, patch, path)
    elif "int effective_sm_count = flashinfer_effective_sm_count(0);" not in text:
        raise ValueError(f"{path}: max active clusters anchor missing")
    before = '''def get_device_sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
'''
    after = '''def get_device_sm_count(device: torch.device) -> int:
    sm_count = _hopper_effective_sm_count()
    if sm_count is not None:
        return sm_count
    return torch.cuda.get_device_properties(device).multi_processor_count
'''
    return replace_exact(text, before, after, path)


def helper_source():
    return '''#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

inline int flashinfer_effective_sm_count(int actual_count) {
  const char* raw = std::getenv("FLASHINFER_EFFECTIVE_SM_COUNT");
  if (raw == nullptr) {
    return actual_count;
  }
  char* end = nullptr;
  long count = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\\0' || count <= 0 || count > 2147483647L) {
    return actual_count;
  }
  return static_cast<int>(count);
}

template <typename T>
inline cudaError_t flashinfer_effective_cudaDeviceGetAttribute(
    T* value, cudaDeviceAttr attr, int device) {
  cudaError_t status = ::cudaDeviceGetAttribute(value, attr, device);
  if (status == cudaSuccess && attr == cudaDevAttrMultiProcessorCount) {
    *value = static_cast<T>(flashinfer_effective_sm_count(
        static_cast<int>(*value)));
  }
  return status;
}
'''


def patch_cutlass_hardware_info(path):
    text = path.read_text()
    text = inject_include(text, "flashinfer/effective_sm_count.cuh")
    before = "    return multiprocessor_count;\n"
    after = "    return flashinfer_effective_sm_count(multiprocessor_count);\n"
    if before in text:
        text = replace_exact(text, before, after, path)
    elif after not in text:
        raise ValueError(f"{path}: SM-count return anchor missing")
    anchor = '''      uint32_t threads_per_block,
      void const* kernel_ptr,
      cudaStream_t stream = nullptr) {
    int max_active_clusters = 0;
'''
    patch = '''      uint32_t threads_per_block,
      void const* kernel_ptr,
      cudaStream_t stream = nullptr) {
    int effective_sm_count = flashinfer_effective_sm_count(0);
    if (effective_sm_count > 0) {
      int cluster_volume = int(cluster_dims.x) * int(cluster_dims.y) *
                           int(cluster_dims.z);
      return std::max(1, effective_sm_count / std::max(cluster_volume, 1));
    }
    int max_active_clusters = 0;
'''
    if anchor in text:
        text = replace_exact(text, anchor, patch, path)
    elif "int effective_sm_count = flashinfer_effective_sm_count(0);" not in text:
        raise ValueError(f"{path}: max active clusters anchor missing")
    if "#include <algorithm>" not in text:
        text = text.replace(
            '#include "cuda_runtime.h"\n',
            '#include "cuda_runtime.h"\n#include <algorithm>\n', 1)
    return text


def scan_sites(site):
    files = set()
    for root_name in SCAN_ROOTS:
        root = site / root_name
        if root.is_dir():
            files.update(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix in CPP_SUFFIXES | {".py"}
            )
    sites = []
    for path in sorted(files):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        rel = path.relative_to(site).as_posix()
        for number, line in enumerate(lines, 1):
            if not SM_READ.search(line):
                continue
            sites.append({
                "file": rel,
                "line": number,
                "source": line.strip()[:240],
                "decision": classify_site(rel, line),
            })
    return sites


def classify_site(relative, line):
    if relative == SOURCE_FILE:
        return "patched: get_device_sm_count and torch.cuda.get_device_properties wrapper"
    if relative == CUTLASS_HW:
        return "patched: SM count and max active clusters use effective SM count"
    if relative.endswith(".py"):
        if any(part in relative for part in (
                "/comm/", "/moe_ep/", "collect_env.py", "sm120",
                "/blackwell/")):
            return "unreachable: not called by the c450v2 FlashInfer case families"
        return "patched: Python device-properties wrapper returns the effective count"
    if relative.startswith("flashinfer/data/include/flashinfer/"):
        return "patched: effective cudaDeviceGetAttribute wrapper"
    if relative.startswith("flashinfer/data/csrc/"):
        if any(token in relative for token in (
                "sm120", "alltoall", "cudnn", "blackwell",
                "low_latency", "batched_gemm")):
            return "unreachable: outside the c450v2 FlashInfer case families"
        return "patched: effective cudaDeviceGetAttribute wrapper"
    if relative.startswith("flashinfer/data/cutlass/"):
        if "examples/" in relative or "tools/" in relative:
            return "unreachable: example or reference utility is not launched by the suite"
        return "patched: effective cudaDeviceGetAttribute wrapper"
    if relative.startswith("flashinfer/data/"):
        return "patched: effective cudaDeviceGetAttribute wrapper"
    if "multiProcessorCount" in line:
        return "patched: effective device-properties wrapper"
    return "unreachable: outside the c450v2 FlashInfer case families"


def patch_site(site):
    changes = {}
    helper_rel = "flashinfer/data/include/flashinfer/effective_sm_count.cuh"
    helper = site / helper_rel
    changes[helper_rel] = helper_source()
    utils_cuh = site / UTILS_CUH
    text = utils_cuh.read_text()
    changes[UTILS_CUH] = inject_include(
        text, "flashinfer/effective_sm_count.cuh")

    utils_py = site / SOURCE_FILE
    changes[SOURCE_FILE] = patch_python_utils(utils_py)

    cutlass_hw = site / CUTLASS_HW
    changes[CUTLASS_HW] = patch_cutlass_hardware_info(cutlass_hw)

    for root_name in SCAN_ROOTS[:1]:
        root = site / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (not path.is_file() or path.suffix not in CPP_SUFFIXES
                    or path == cutlass_hw):
                continue
            text = path.read_text(errors="replace")
            if not re.search(
                    r"cudaDevAttrMultiProcessorCount|"
                    r"\.multiProcessorCount", text):
                continue
            rel = path.relative_to(site).as_posix()
            if rel in changes:
                continue
            changed = text
            if path.is_relative_to(site / "flashinfer/data/include"):
                if rel != UTILS_CUH:
                    changed = inject_include(
                        changed, "flashinfer/effective_sm_count.cuh")
            else:
                changed = inject_include(
                    changed, "flashinfer/effective_sm_count.cuh")
            changed = re.sub(
                r"\bcudaDeviceGetAttribute\(\s*([^,]+),\s*"
                r"cudaDevAttrMultiProcessorCount,\s*([^)]+)\)",
                r"flashinfer_effective_cudaDeviceGetAttribute("
                r"\1, cudaDevAttrMultiProcessorCount, \2)",
                changed,
            )
            changed = re.sub(
                r"(?<!flashinfer_effective_sm_count\()"
                r"(\b[A-Za-z_]\w*\.multiProcessorCount\b)",
                r"flashinfer_effective_sm_count(\1)",
                changed,
            )
            changes[rel] = changed

    before_after = {}
    for rel, changed in changes.items():
        path = site / rel
        before_sha = sha256(path) if path.is_file() else None
        encoded = changed.encode()
        if path.is_file() and path.read_bytes() == encoded:
            after_sha = before_sha
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
            after_sha = sha256(path)
        before_after[rel] = {
            "before_sha256": before_sha,
            "after_sha256": after_sha,
        }
    return before_after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--far-env", type=Path, default=FAR_ENV)
    args = parser.parse_args()
    source = args.source or source_site_from_env(args.far_env)
    source = source.resolve()
    dest = args.dest.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise SystemExit("refusing source/destination overlap")
    if not source.is_dir():
        raise SystemExit(f"FlashInfer source site is missing: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / MANIFEST
    if dest.exists() and manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if old.get("source_site") != str(source):
            raise SystemExit(
                f"existing patched site has another source: {manifest_path}")
        if old.get("patch_version") == PATCH_VERSION:
            print(json.dumps({
                "site": str(dest), "already_patched": True,
                "files": len(old.get("files", {})),
                "sites": len(old.get("sites", [])),
            }))
            return 0
    if not dest.exists():
        shutil.copytree(source, dest, symlinks=True, copy_function=shutil.copy2)
        (dest / IN_PROGRESS).write_text(json.dumps(
            {"source_site": str(source)}, sort_keys=True) + "\n")
    elif not manifest_path.is_file():
        progress_path = dest / IN_PROGRESS
        if not progress_path.is_file():
            raise SystemExit(
                f"refusing unmarked destination; inspect it first: {dest}")
        progress = json.loads(progress_path.read_text())
        if progress.get("source_site") != str(source):
            raise SystemExit(
                f"existing destination copy has another source: {progress_path}")
    changes = patch_site(dest)
    site_list = scan_sites(dest)
    previous_files = old.get("files", {}) if manifest_path.is_file() else {}
    for rel, record in changes.items():
        if rel in previous_files:
            record["before_sha256"] = previous_files[rel].get(
                "before_sha256", record["before_sha256"])
    result = {
        "patch_version": PATCH_VERSION,
        "source_site": str(source),
        "destination_site": str(dest),
        "files": changes,
        "sites": site_list,
        "summary": {
            "patched_files": len(changes),
            "patched_sites": sum(
                item["decision"].startswith("patched:")
                for item in site_list),
            "unreachable_sites": sum(
                item["decision"].startswith("unreachable:")
                for item in site_list),
        },
    }
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (dest / IN_PROGRESS).unlink(missing_ok=True)
    print(json.dumps({"site": str(dest), **result["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

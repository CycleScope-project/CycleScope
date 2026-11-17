#!/usr/bin/env bash
# blackwell-flashmla-build: FlashMLA cuda.abi3.so for B300 (sm_103a).
# evidence: deps/flashmla content (_build_info.py + cuda.abi3.so);
#   pip --no-build-isolation install of blackwell-flashmla-source
#   (api-fragment patched); source's own submodule cutlass v4.3.5
#   (4faf1a15) — NOT blackwell-cutlass-source/f614dc4 (old ninja
#   build.ninja identical to stock, only paths differ).
# spill check: old deps/flashmla/cuda.abi3.so has 40 STACK/LOCAL>0
#   entries (cuobjdump -res-usage), so the original build bypassed
#   setup.py's post-link reg-spill check -> SKIP_REG_SPILL_CHECK.
# input: blackwell-flashmla-source only.
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/blackwell-flashmla-build
S=$T/work/src
rm -rf "$T"
mkdir -p "$T" "$T/logs" "$T/work"
rm -rf "$T/work/src"
# build on a copy — source entries are verified read-only
cp -a "$CYCLESCOPE_ARTIFACT_ROOT/tools/blackwell-flashmla-source" "$T/work/src"
export FLASH_MLA_SKIP_REG_SPILL_CHECK=1
python3 -m pip install --no-build-isolation --no-deps \
  --target "$T" -v "$S" > "$T/logs/pip.log" 2>&1
test -f "$T/flash_mla/cuda.abi3.so" || \
  find "$T" -name 'cuda.abi3.so' | grep -q .
rm -rf "$T/work"

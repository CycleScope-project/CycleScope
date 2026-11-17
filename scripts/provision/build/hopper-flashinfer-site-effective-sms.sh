#!/usr/bin/env bash
# hopper-flashinfer-site-effective-sms: marker-injected + effective-SM
# patched flashinfer site-packages (fi_site_effsm equivalent).
# evidence: old fi_site_effsm files carry FI_DELIVERY_MARKER_TIMING
#   (27 files) and FI_DELIVERY_SM_CAP (3 files) -> the effsm --source
#   was a delivery site copy already injected by
#   harness/marker_mode.py:apply_marker_mode ->
#   suites/flashinfer/scripts/patch_wheel_marker.py --tree <site>
#   --apply --sm-cap --log <A>/build/marker_logs/fi_marker_patch_log.json.
#   Then scripts/rounds/c450v2_fi_effsm_patch.py --source <copy> --dest
#   (sha 8da45481, byte-identical in mirror).
# input: hopper-flashinfer-delivery (site-packages source dir).
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
H=$CYCLESCOPE_ROOT/suites/hopper/runners/harness
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-flashinfer-site-effective-sms
D=$CYCLESCOPE_ARTIFACT_ROOT/tools/hopper-flashinfer-delivery
SRC="$D/build/site-packages-fi0617-cutlass470-tvmffi0113"
rm -rf "$T"
mkdir -p "$T/work" "$T/logs"
# 1. copy delivery site to a scratch tree (never patch delivery in place)
cp -a "$SRC" "$T/work/source-site"
# 2. marker injection, two passes (evidence: hopper-prototype@65579254 changed
#    insert_include to file-top; the 9 mismatched files were patched by
#    the earlier "after last top-level include" version; mha.cu only got
#    coverage at 17986138, so the pre-17986138 version runs first and the
#    current version fills the rest — sentinel skips already-patched).
#    17986138^ copy: scripts/provision/flashinfer/marker-pre-mha/
#    (patch_wheel_marker.py sha 539efbb1, fi_delivery_marker.h).
python3 "$CYCLESCOPE_ROOT/scripts/provision/flashinfer/marker-pre-mha/scripts/patch_wheel_marker.py" \
  --tree "$T/work/source-site" --apply --sm-cap \
  --log "$T/logs/marker-pre-mha.json"
# second pass: the 55bc671f^-era version (file-top insert since
# 65579254; header still says delivery_tests/, matching the old site).
python3 "$CYCLESCOPE_ROOT/scripts/provision/flashinfer/marker-rename-era/scripts/patch_wheel_marker.py" \
  --tree "$T/work/source-site" --apply --sm-cap \
  --log "$T/logs/fi_marker_patch_log.json"
# 3. effsm patch; the script copies source -> dest itself (refuses a
#    pre-existing unmarked dest), so dest must not exist beforehand
python3 "$H/scripts/rounds/c450v2_fi_effsm_patch.py" \
  --source "$T/work/source-site" \
  --dest "$T/dest" \
  --far-env "$H/scripts/environment_far.sh" \
  > "$T/logs/patch.log" 2>&1
test -f "$T/dest/EFFSM_PATCH.json"
# jit-cache blocking stub: generated inline by
# delivery_tests/common/far_holdout/bin/run_flashinfer_suite.sh @c91ba913
# (heredoc; sha 9bdf54a4 matches old fi_site_effsm copy)
cp "$CYCLESCOPE_ROOT/scripts/provision/flashinfer/sitecustomize_jit_block.py" \
  "$T/dest/sitecustomize.py"
# flatten: dest is the tool root content; keep work/ + logs/ for audit
# (compare excludes work/ and logs/)
find "$T/dest" -mindepth 1 -maxdepth 1 -exec mv -t "$T" -- {} +
rmdir "$T/dest"
test -f "$T/flashinfer/__init__.py" -a -f "$T/EFFSM_PATCH.json" -a -f "$T/sitecustomize.py"

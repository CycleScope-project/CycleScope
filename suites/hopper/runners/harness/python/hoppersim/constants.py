from __future__ import annotations

VERSION = "0.1.0"

# Project-wide formal microbenchmark acceptance policy.  A value equal to the
# limit is rejected; callers must use strict less-than comparisons.
MICROBENCH_MAXIMUM_CV_PERCENT = 2.5
MICROBENCH_MAXIMUM_REPEAT_DRIFT_PERCENT = 2.5

BUNDLE_SCHEMA = "hopper.bundle.v1"
PROFILE_SCHEMA = "hopper.profile.v1"
RUN_SCHEMA = "hopper.run.v1"
CAPTURE_SCHEMA = "hopper.capture.v1"

WARP_MAGIC = b"HSWRPV1\0"
INSTRUCTION_MAGIC = b"HSINSV1\0"
REGISTER_MAGIC = b"HSREGV1\0"
STRING_MAGIC = b"HSSTRV1\0"
PAYLOAD_MAGIC = b"HSPAYV1\0"
PAYLOAD_CHUNK_MAGIC = b"HSPCHV1\0"

# Kind-31/35 currently carry mbarrier generation and initialization epoch in
# one byte each.  Values outside this development wire domain remain valid
# capture facts, but must use the established untyped/base lifecycle until a
# lossless relation-extension row is added.
TMA_TYPED_SIGNATURE_RELATION_FIELD_LIMIT = 1 << 8
TMA_TYPED_SIGNATURE_DOMAIN_MISS = (
    "tma_typed_producer_signature_generation_or_epoch_out_of_wire_domain"
)

# The single real simulation parameter family of the accuracy-evidence
# pipeline: the H100 family has one measured parameter set (historical
# readings normalized by hoppersim.timing_metric_normalization).
SIMULATION_PARAMETER_FAMILY = "h100"

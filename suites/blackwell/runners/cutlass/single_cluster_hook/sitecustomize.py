"""Cap CuTe-DSL persistent scheduling to one active cluster for c450v3."""
import os

if os.environ.get("BW_SINGLE_CLUSTER_MAX_CLUSTERS") == "1":
    try:
        from cutlass.utils.hardware_info import HardwareInfo

        HardwareInfo.get_max_active_clusters = lambda self, cluster_size, stream=None: 1
    except ImportError:
        pass

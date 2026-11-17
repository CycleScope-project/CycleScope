"""Repository tools plus the retained calibration command packages."""

from pathlib import Path


# The refactor keeps active project tools small while preserving the historical
# import names used by the calibration collectors.  Direct source paths are
# resolved by the collectors themselves; this only supplies ``tools.<suite>``
# package imports.
_LEGACY_TOOLS = Path(__file__).resolve().parents[1] / "calibration/legacy_tools"
if _LEGACY_TOOLS.is_dir():
    __path__.append(str(_LEGACY_TOOLS))

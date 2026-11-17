"""Public microbench entry backed by the pinned DeepGEMM suite runner."""

from tools.microbench.deepgemm import deepgemm_sm90_runner as _runner

for _name in dir(_runner):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_runner, _name)

if __name__ == "__main__":
    raise SystemExit(main())

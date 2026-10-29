"""
Module entry point: viam-server starts this (via run.sh) with a socket path; run_from_registry registers every
imported resource model and serves it.
"""
import asyncio
import sys
from pathlib import Path

# Archive / repo root, so `sorter` and `scripts` import the same way in both layouts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viam.module.module import Module  # noqa: E402

from module.sorter_service import GraspByDescription  # noqa: E402,F401  registers the model

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())

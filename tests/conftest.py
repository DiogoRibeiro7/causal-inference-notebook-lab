from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Use a non-interactive backend so plotting tests never open Tk windows or
# fail when many figures accumulate across the suite.
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

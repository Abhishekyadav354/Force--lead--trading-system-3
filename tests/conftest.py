from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "force-lead-trading-system"

for path in [str(ROOT), str(PACKAGE_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

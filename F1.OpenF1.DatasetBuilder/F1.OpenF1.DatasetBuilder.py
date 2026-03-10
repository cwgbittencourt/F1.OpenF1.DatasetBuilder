from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "f1_dataset" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jobs.build_openf1_dataset import main


if __name__ == "__main__":
    main()

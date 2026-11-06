"""Shared dataset location; override with --root or RAPESEED_DATA_ROOT."""

import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path(os.environ.get("RAPESEED_DATA_ROOT", "/home/ADS/mranj004/germany8020"))

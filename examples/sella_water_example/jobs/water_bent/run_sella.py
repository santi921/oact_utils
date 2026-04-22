#!/usr/bin/env python
"""Generated Sella runner. Reads sella_config.json for parameters."""
import json
from pathlib import Path

from oact_utilities.core.orca.sella_runner import run_sella_optimization

config = json.loads(Path("sella_config.json").read_text())
run_sella_optimization(job_dir=".", **config)

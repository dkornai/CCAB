import sys
from pathlib import Path

import torch

# Make the project modules (and the built pf_cpp module) importable from the tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.set_grad_enabled(False)

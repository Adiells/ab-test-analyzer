"""Application-wide constants.

Centralises magic values so they can be changed in one place.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Dataset conventions ──────────────────────────────────────────────

CONTROL_GROUP_LABEL = "control"
TREATMENT_GROUP_LABEL = "treatment"

# ── Statistical defaults ─────────────────────────────────────────────

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SIGNIFICANCE_LEVEL = 0.05

# ── Supported file formats ───────────────────────────────────────────

SUPPORTED_FILE_EXTENSIONS = frozenset({".csv"})

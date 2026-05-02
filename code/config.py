"""
Configuration for the multi-domain support triage agent.
All secrets are read from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
TICKETS_DIR = PROJECT_ROOT / "support_tickets"
INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV = TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"
VECTOR_STORE_DIR = CODE_DIR / ".vectorstore"

# ── API keys (never hardcoded) ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Model configuration ───────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = 0.0          # deterministic output
SEED = 42                  # reproducibility

# ── Retrieval configuration ───────────────────────────────────────────
TOP_K = 10                 # chunks to retrieve per query
CHUNK_MAX_CHARS = 2000     # max characters per chunk
CHUNK_OVERLAP_CHARS = 200  # overlap between chunks

# ── Company → corpus folder mapping ───────────────────────────────────
COMPANY_CORPUS = {
    "hackerrank": "hackerrank",
    "claude":     "claude",
    "visa":       "visa",
}

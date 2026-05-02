## Multi Domain Support Agent

A RAG-based AI agent that triages customer support tickets across **HackerRank**, **Claude (Anthropic)**, and **Visa** ecosystems using only the provided support corpus.

![workflow](/asset/programWorkflow.png)

> This was the problem statement given for Hackerrank's orchestrate contest May26

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **RAG over full-context** | Corpus is ~5.8 MB / 774 files — too large for a single LLM context window. Chunked retrieval surfaces only the most relevant docs. |
| **ChromaDB persistent store** | Index is built once and reused across runs, making subsequent runs fast. |
| **OpenAI text-embedding-3-small** | (Replaced) Now using Local Cross-Encoder (`ms-marco`) + BM25 for Hybrid Search with zero API costs. |
| **Llama 3.3 70B with Tool Use** | Agentic loop with tools guarantees the LLM can search dynamically instead of being force-fed context. |
| **Company-filtered retrieval** | When the ticket specifies a company, we restrict search to that corpus slice, improving precision. |
| **Explicit escalation rules** | Safety-critical decisions (billing, security, account access) are escalated by design rather than relying solely on LLM judgment. |
| **Temperature 0 + seed** | Deterministic outputs for reproducibility. |

### Escalation Logic

The agent escalates (rather than guessing) when:
- Billing, refund, or payment disputes requiring account access
- Security incidents (fraud, identity theft, vulnerabilities)
- Account restoration needing admin privileges
- Score/grade disputes on assessments
- Service-wide outages requiring engineering investigation
- Legal/compliance/infosec requests
- Insufficient information for a safe response


## Setup

### Prerequisites
- Python 3.10+
- A Groq API key

### Installation

```bash
cd code
pip install -r requirements.txt
```

### Configuration

```bash
# Copy and edit the env file
cp ../.env.example ../.env
# Add your key:
# GROQ_API_KEY=gsk-...
```

Or export directly:
```bash
export GROQ_API_KEY="gsk-..."
```

## Running

```bash
# Process support_tickets.csv → output.csv
python main.py

# Force rebuild the vector index
python main.py --rebuild

# Test on the sample tickets first
python main.py --sample
```

### Output

Results are written to `support_tickets/output.csv` with columns:
- `issue`, `subject`, `company` (original input)
- `response` — user-facing answer grounded in the corpus
- `product_area` — relevant support category
- `status` — Replied or Escalated
- `request_type` — product_issue / feature_request / bug / invalid
- `justification` — concise reasoning for the decision

## File Structure

```
code/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── main.py            # Entry point — CLI + CSV I/O
├── agent.py           # TriageAgent — orchestration pipeline
├── retriever.py       # ChromaDB vector store + semantic search
├── corpus_loader.py   # Markdown corpus loading & chunking
├── prompts.py         # LLM prompt templates & formatting
└── config.py          # Configuration (paths, models, hyperparams)
```

## Dependencies

| Package | Purpose |
|---|---|
| `groq` | LLM (Llama 3.3 70B via Groq API) |
| `chromadb` | Persistent vector store + local embeddings (`all-MiniLM-L6-v2`) |
| `rank_bm25` | Keyword search for Hybrid RAG |
| `sentence-transformers` | Cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) |
| `pandas` | CSV reading/writing |
| `pytest` | Unit testing framework |
| `python-dotenv` | Load API keys from `.env` |


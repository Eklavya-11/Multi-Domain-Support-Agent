#!/usr/bin/env python3
"""
main.py — Entry point for the multi-domain support triage agent.

Usage:
    python main.py                  # process support_tickets.csv → output.csv
    python main.py --rebuild        # rebuild the vector index first
    python main.py --sample         # run on sample_support_tickets.csv instead

Reads GROQ_API_KEY from environment / .env file.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd

from config import INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV
from agent import TriageAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-domain support triage agent"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild of the vector store index",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Run on sample_support_tickets.csv instead of the main file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output path (default: support_tickets/output.csv)",
    )
    args = parser.parse_args()

    input_path = SAMPLE_CSV if args.sample else INPUT_CSV
    output_path = Path(args.output) if args.output else OUTPUT_CSV

    print(f"{'='*60}")
    print(f"  Multi-Domain Support Triage Agent")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # ── 1. Initialise agent (builds index on first run) ───────────
    agent = TriageAgent(rebuild_index=args.rebuild)

    # ── 2. Load tickets ──────────────────────────────────────────
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"\n[main] Loaded {len(df)} tickets from {input_path.name}\n")

    # ── 3. Process each ticket ───────────────────────────────────
    results = []
    for idx, row in df.iterrows():
        issue   = str(row.get("Issue", row.get("issue", "")))
        subject = str(row.get("Subject", row.get("subject", "")))
        company = str(row.get("Company", row.get("company", "")))

        print(f"[{idx+1}/{len(df)}] Processing: {subject[:60] or issue[:60]}...")
        t0 = time.time()

        try:
            result = agent.triage(issue, subject, company)
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            from agent import TriageResult
            result = TriageResult(
                status="escalated",
                product_area="unknown",
                response="This ticket requires human review.",
                justification=f"Agent error during processing: {type(e).__name__}",
                request_type="product_issue",
            )

        elapsed = time.time() - t0
        print(f"  → {result.status} | {result.request_type} | "
              f"{result.product_area} ({elapsed:.1f}s)")

        results.append({
            "issue":         issue,
            "subject":       subject,
            "company":       company,
            "response":      result.response,
            "product_area":  result.product_area,
            "status":        result.status.capitalize(),   # Replied / Escalated
            "request_type":  result.request_type,
            "justification": result.justification,
        })

    # ── 4. Write output CSV ──────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n{'='*60}")
    print(f"  ✅ Wrote {len(results)} results to {output_path}")
    print(f"{'='*60}")

    # ── 5. Print summary stats ───────────────────────────────────
    replied   = sum(1 for r in results if r["status"] == "Replied")
    escalated = sum(1 for r in results if r["status"] == "Escalated")
    print(f"\n  Summary: {replied} replied, {escalated} escalated")
    for rt in ("product_issue", "feature_request", "bug", "invalid"):
        n = sum(1 for r in results if r["request_type"] == rt)
        if n:
            print(f"    {rt}: {n}")


if __name__ == "__main__":
    main()

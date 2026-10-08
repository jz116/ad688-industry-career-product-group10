#!/usr/bin/env python3
"""Recover the original source rows for the audited posting IDs.

This exports original records only. It does not infer the missing panel or
structured skill bridge from description text or aggregate counts.
Run from the original project root. No third-party packages are required.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def normalize_id(value):
    value = str(value or "").strip()
    if re.fullmatch(r"[0-9]+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Original 2024 Lightcast CSV")
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/processed/step3/career_market_descriptions.csv"))
    parser.add_argument("--output", type=Path, default=Path("step3_source_rows.csv"))
    args = parser.parse_args()
    csv.field_size_limit(100_000_000)

    if not args.corpus.is_file():
        parser.error(f"Description file not found: {args.corpus}")
    with args.corpus.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "JOB_ID" not in (reader.fieldnames or []):
            parser.error("The description file must contain JOB_ID.")
        target_list = [normalize_id(row["JOB_ID"]) for row in reader]
    if not target_list or any(not value for value in target_list):
        parser.error("The description file has no IDs or contains blank IDs.")
    if len(set(target_list)) != len(target_list):
        parser.error("The description file contains duplicate JOB_ID values.")
    targets = set(target_list)

    raw_path = args.raw
    if raw_path is None:
        candidates = [Path("data/raw/step3/lightcast_job_postings.csv"),
                      Path("lightcast_job_postings.csv")]
        for candidate in candidates:
            if candidate.is_file():
                raw_path = candidate
                break
    if raw_path is None or not raw_path.is_file():
        parser.error("Original 2024 CSV not found. Use --raw /path/to/lightcast_job_postings.csv. "
                     "Run this on the computer or EC2 instance that has the original dataset.")

    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        parser.error("An output already exists. Choose another --output filename; nothing was overwritten.")
    selected = {}
    duplicates = set()
    scanned = 0
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if "ID" not in columns:
            parser.error("The raw CSV must contain ID. Use the original 2024 Lightcast file.")
        for row in reader:
            scanned += 1
            job_id = normalize_id(row.get("ID"))
            if job_id not in targets:
                continue
            if job_id in selected:
                duplicates.add(job_id)
            else:
                selected[job_id] = row
    missing = targets - set(selected)
    if missing or duplicates:
        print(f"Source rows scanned: {scanned:,}")
        print(f"Target IDs found: {len(selected)} / {len(targets)}")
        if missing:
            print("Missing target IDs:", ", ".join(sorted(missing)))
        if duplicates:
            print("Duplicate matching source IDs:", ", ".join(sorted(duplicates)))
        raise SystemExit("Recovery stopped before writing. Confirm the source file and inspect duplicates.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for job_id in target_list:
            writer.writerow(selected[job_id])
    report = {"raw_path": str(raw_path), "corpus_path": str(args.corpus),
              "source_rows_scanned": scanned, "source_columns": len(columns),
              "target_ids": len(targets), "matched_ids": len(selected),
              "missing_ids": [], "duplicate_matching_ids": [],
              "output": str(args.output),
              "purpose": "Exact original source rows; missing analysis inputs have not yet been rebuilt."}
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Recovered {len(selected)} original records from {scanned:,} source rows.")
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"Created {audit_path}")
    print("Upload these two small files for reconstruction and verification.")


if __name__ == "__main__":
    main()

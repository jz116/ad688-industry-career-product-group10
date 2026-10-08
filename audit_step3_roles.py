#!/usr/bin/env python3
"""Audit 2024 Software Publishers postings before building the Step 3 panel.

This script streams the large Lightcast CSV one record at a time, filters to
2022 NAICS 513210 and the documented May--September 2024 window, and applies a
transparent Data Science / AI / ML role classification. It deliberately does
not use a detailed SOC label by itself to include a posting: broad SOC mappings
can otherwise turn titles such as "Enterprise Analyst" into false Data
Scientist matches.

Run from the repository root:
    python audit_step3_roles.py

Outputs:
    outputs/step3/software_publishers_role_audit.csv
    outputs/step3/included_role_candidates.csv
    outputs/step3/soc_mapping_only_review.csv
    outputs/step3/role_segment_summary.csv
    outputs/step3/included_title_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path


DEFAULT_SOURCE = Path("data/raw/step3/lightcast_job_postings.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/step3")
NAICS_CODE = "513210"
WINDOW_START = date(2024, 5, 1)
WINDOW_END = date(2024, 9, 30)

SOURCE_COLUMNS = [
    "ID",
    "LAST_UPDATED_TIMESTAMP",
    "POSTED",
    "TITLE_RAW",
    "TITLE_CLEAN",
    "TITLE_NAME",
    "ONET",
    "ONET_NAME",
    "SOC_2021_5",
    "SOC_2021_5_NAME",
    "SOC_5",
    "SOC_5_NAME",
    "COMPANY_NAME",
    "CITY_NAME",
    "STATE_NAME",
    "NAICS_2022_6",
    "NAICS_2022_6_NAME",
]

OUTPUT_COLUMNS = SOURCE_COLUMNS + [
    "ROLE_SEGMENT",
    "ROLE_MATCH_REASON",
    "SCOPE_TIER",
    "SOC_DATA_SCIENTIST_MAPPING",
    "MAPPING_ONLY_CORE_RISK",
]

DATA_SCIENTIST_PATTERN = re.compile(r"\bdata scientists?\b", re.IGNORECASE)
DATA_SCIENCE_PATTERN = re.compile(r"\bdata science\b", re.IGNORECASE)
DATA_SCIENCE_ROLE_PATTERN = re.compile(
    r"\b(scientist|analyst|manager|director|lead|intern)\b",
    re.IGNORECASE,
)
ML_PATTERN = re.compile(
    r"machine learning|(?<![a-z0-9])ml(?![a-z0-9])",
    re.IGNORECASE,
)
AI_PATTERN = re.compile(
    r"artificial intelligence|(?<![a-z0-9.])ai(?![a-z0-9])",
    re.IGNORECASE,
)
TECHNICAL_ROLE_PATTERN = re.compile(
    r"\b(engineer|scientist|developer|architect|intern|researcher)\b",
    re.IGNORECASE,
)
APPLIED_DECISION_PATTERN = re.compile(
    r"\b(applied|decision) scientists?\b",
    re.IGNORECASE,
)
RESEARCH_SCIENTIST_PATTERN = re.compile(
    r"\bresearch scientists?\b",
    re.IGNORECASE,
)
ANALYTICS_ENGINEER_PATTERN = re.compile(
    r"\banalytics engineers?\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Step 3 Software Publishers role filter."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to lightcast_job_postings.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit CSV files.",
    )
    return parser.parse_args()


def text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def joined_text(row: dict[str, str], columns: list[str]) -> str:
    values = []
    for column in columns:
        value = text(row.get(column))
        if value:
            values.append(value)
    return " ".join(values)


def normalized_naics(value: object) -> str:
    result = text(value)
    if result.endswith(".0"):
        result = result[:-2]
    return result


def parse_posted_date(value: object) -> date | None:
    raw = text(value)
    if not raw:
        return None

    formats = ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y")
    for date_format in formats:
        try:
            return datetime.strptime(raw, date_format).date()
        except ValueError:
            continue
    return None


def classify_role(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    # Raw/clean titles are the strongest evidence that the posting itself is
    # for the selected pathway.
    title_text = joined_text(row, ["TITLE_RAW", "TITLE_CLEAN"])

    # Standardized title and O*NET labels may support inclusion. Detailed SOC
    # labels are audited separately because they can be broader than the title.
    standardized_text = joined_text(row, ["TITLE_NAME", "ONET_NAME"])
    soc_text = joined_text(row, ["SOC_2021_5_NAME", "SOC_5_NAME"])

    explicit_data_scientist = bool(DATA_SCIENTIST_PATTERN.search(title_text))
    data_science_leadership = bool(
        DATA_SCIENCE_PATTERN.search(title_text)
        and DATA_SCIENCE_ROLE_PATTERN.search(title_text)
    )
    standardized_data_scientist = bool(
        DATA_SCIENTIST_PATTERN.search(standardized_text)
    )
    soc_data_scientist = bool(DATA_SCIENTIST_PATTERN.search(soc_text))

    ml_term = bool(ML_PATTERN.search(title_text))
    ai_term = bool(AI_PATTERN.search(title_text))
    technical_role = bool(TECHNICAL_ROLE_PATTERN.search(title_text))
    applied_decision = bool(APPLIED_DECISION_PATTERN.search(title_text))
    research_scientist = bool(
        RESEARCH_SCIENTIST_PATTERN.search(title_text)
        and (
            DATA_SCIENCE_PATTERN.search(title_text)
            or ML_PATTERN.search(title_text)
            or AI_PATTERN.search(title_text)
        )
    )
    analytics_engineer = bool(ANALYTICS_ENGINEER_PATTERN.search(title_text))

    role_segment = ""
    role_reason = ""

    if explicit_data_scientist:
        role_segment = "Data Scientist"
        role_reason = "Explicit Data Scientist title"
    elif data_science_leadership:
        role_segment = "Data Scientist"
        role_reason = "Explicit Data Science role title"
    elif standardized_data_scientist:
        role_segment = "Data Scientist"
        role_reason = "Standardized title or O*NET Data Scientist mapping"
    elif applied_decision:
        role_segment = "Applied or Decision Scientist"
        role_reason = "Applied or Decision Scientist title"
    elif research_scientist:
        role_segment = "Data/AI Research Scientist"
        role_reason = "Data/AI Research Scientist title"
    elif ml_term and technical_role:
        role_segment = "Machine Learning Technical Role"
        role_reason = "Explicit ML technical title"
    elif ai_term and technical_role:
        role_segment = "AI Technical Role"
        role_reason = "Explicit AI technical title"
    elif analytics_engineer:
        role_segment = "Analytics Engineer"
        role_reason = "Explicit Analytics Engineer title"

    if role_segment == "Data Scientist":
        scope_tier = "Core"
    elif role_segment:
        scope_tier = "Adjacent"
    else:
        scope_tier = ""

    strong_core_signal = (
        explicit_data_scientist
        or data_science_leadership
        or standardized_data_scientist
    )
    mapping_only_core_risk = soc_data_scientist and not strong_core_signal

    return (
        role_segment,
        role_reason,
        scope_tier,
        "Yes" if soc_data_scientist else "No",
        "Yes" if mapping_only_core_risk else "No",
    )


def latest_record_is_newer(
    candidate: dict[str, str],
    current: dict[str, str],
) -> bool:
    candidate_timestamp = text(candidate.get("LAST_UPDATED_TIMESTAMP"))
    current_timestamp = text(current.get("LAST_UPDATED_TIMESTAMP"))
    return candidate_timestamp >= current_timestamp


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_counter(
    path: Path,
    first_column: str,
    counts: Counter[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([first_column, "job_count"])
        for value, count in counts.most_common():
            writer.writerow([value, count])


def main() -> None:
    args = parse_args()
    source_path = args.source
    output_dir = args.output_dir

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)

    source_rows = 0
    target_code_rows = 0
    invalid_target_dates = 0
    outside_window_rows = 0
    target_by_id: dict[str, dict[str, str]] = {}

    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        available_columns = set(reader.fieldnames or [])
        missing_columns = [
            column for column in SOURCE_COLUMNS if column not in available_columns
        ]
        if missing_columns:
            raise RuntimeError(
                "Source CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        for row in reader:
            source_rows += 1

            if normalized_naics(row.get("NAICS_2022_6")) != NAICS_CODE:
                continue
            target_code_rows += 1

            posted_date = parse_posted_date(row.get("POSTED"))
            if posted_date is None:
                invalid_target_dates += 1
                continue
            if not WINDOW_START <= posted_date <= WINDOW_END:
                outside_window_rows += 1
                continue

            job_id = text(row.get("ID"))
            if not job_id:
                continue

            selected = {column: text(row.get(column)) for column in SOURCE_COLUMNS}
            current = target_by_id.get(job_id)
            if current is None or latest_record_is_newer(selected, current):
                target_by_id[job_id] = selected

            if source_rows % 10_000 == 0:
                print(
                    f"Processed {source_rows:,} rows; "
                    f"retained {len(target_by_id):,} unique target-industry IDs"
                )

    audited_rows = []
    included_rows = []
    mapping_review_rows = []
    role_counts: Counter[str] = Counter()
    included_title_counts: Counter[str] = Counter()

    for job_id in sorted(target_by_id):
        row = target_by_id[job_id]
        (
            role_segment,
            role_reason,
            scope_tier,
            soc_mapping,
            mapping_only_risk,
        ) = classify_role(row)

        audited = dict(row)
        audited["ROLE_SEGMENT"] = role_segment
        audited["ROLE_MATCH_REASON"] = role_reason
        audited["SCOPE_TIER"] = scope_tier
        audited["SOC_DATA_SCIENTIST_MAPPING"] = soc_mapping
        audited["MAPPING_ONLY_CORE_RISK"] = mapping_only_risk
        audited_rows.append(audited)

        if mapping_only_risk == "Yes":
            mapping_review_rows.append(audited)

        if role_segment:
            included_rows.append(audited)
            role_counts[role_segment] += 1
            title = text(audited.get("TITLE_CLEAN")) or text(
                audited.get("TITLE_RAW")
            )
            included_title_counts[title or "Unspecified title"] += 1

    audit_path = output_dir / "software_publishers_role_audit.csv"
    included_path = output_dir / "included_role_candidates.csv"
    mapping_path = output_dir / "soc_mapping_only_review.csv"
    role_summary_path = output_dir / "role_segment_summary.csv"
    title_summary_path = output_dir / "included_title_summary.csv"

    write_rows(audit_path, audited_rows)
    write_rows(included_path, included_rows)
    write_rows(mapping_path, mapping_review_rows)
    write_counter(role_summary_path, "role_segment", role_counts)
    write_counter(title_summary_path, "title", included_title_counts)

    print("\nSTEP 3 ROLE AUDIT")
    print(f"Source rows: {source_rows:,}")
    print(f"Rows with exact NAICS 513210: {target_code_rows:,}")
    print(f"Target rows with invalid dates: {invalid_target_dates:,}")
    print(f"Target rows outside the stated window: {outside_window_rows:,}")
    print(f"Unique target-industry jobs in window: {len(audited_rows):,}")
    print(f"Included pathway candidates: {len(included_rows):,}")
    print(f"SOC mapping-only core risks: {len(mapping_review_rows):,}")

    print("\nROLE SEGMENTS")
    for role_segment, count in role_counts.most_common():
        print(f"{role_segment}: {count:,}")

    print("\nTOP INCLUDED TITLES")
    for title, count in included_title_counts.most_common(30):
        print(f"{count:4}  {title}")

    print("\nCREATED FILES")
    for output_path in (
        audit_path,
        included_path,
        mapping_path,
        role_summary_path,
        title_summary_path,
    ):
        print(output_path)


if __name__ == "__main__":
    main()

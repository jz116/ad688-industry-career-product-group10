#!/usr/bin/env python3
"""Build the validated Step 3 career-market panel from the 2024 Lightcast CSV.

The script streams the 684 MB source CSV, retaining only IDs approved by
``audit_step3_roles.py``. The retained cohort is then cleaned in memory because
it contains only a small number of records.

Run from the repository root:
    python prepare_step3_market_panel.py

Inputs:
    data/raw/step3/lightcast_job_postings.csv
    outputs/step3/included_role_candidates.csv

Outputs (the Step 2 files are not overwritten):
    data/processed/step3/career_market_panel.csv
    data/processed/step3/career_market_skills.csv
    data/processed/step3/career_market_data_dictionary.csv
    data/processed/step3/career_market_cleaning_audit.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path("data/raw/step3/lightcast_job_postings.csv")
DEFAULT_CANDIDATES = Path("outputs/step3/included_role_candidates.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/step3")

NAICS_CODE = "513210"
NAICS_NAME = "Software Publishers"
WINDOW_START = date(2024, 5, 1)
WINDOW_END = date(2024, 9, 30)
MIN_PLAUSIBLE_ANNUAL_SALARY = 15_000.0
MAX_PLAUSIBLE_ANNUAL_SALARY = 1_000_000.0

STRUCTURED_SKILL_COLUMNS = [
    "SKILLS_NAME",
    "SPECIALIZED_SKILLS_NAME",
    "COMMON_SKILLS_NAME",
    "SOFTWARE_SKILLS_NAME",
]

SKILL_CANONICAL_NAMES = {
    "amazon web services": "AWS",
    "amazon web services (aws)": "AWS",
    "aws": "AWS",
    "google cloud platform": "Google Cloud",
    "google cloud platform (gcp)": "Google Cloud",
    "gcp": "Google Cloud",
    "microsoft azure": "Azure",
    "python (programming language)": "Python",
    "r (programming language)": "R",
    "sql (programming language)": "SQL",
    "java (programming language)": "Java",
    "scala (programming language)": "Scala",
    "go (programming language)": "Go",
    "c++ (programming language)": "C++",
    "c# (programming language)": "C#",
    "javascript (programming language)": "JavaScript",
    "typescript (programming language)": "TypeScript",
    "git (version control system)": "Git",
    "scikit-learn (python package)": "scikit-learn",
}

SKILL_PATTERNS = [
    ("Python", re.compile(r"(?<![a-z0-9])python(?![a-z0-9])", re.IGNORECASE)),
    ("SQL", re.compile(r"(?<![a-z0-9])sql(?![a-z0-9])", re.IGNORECASE)),
    (
        "R",
        re.compile(
            r"(?i:\br programming\b|\br language\b)|(?<![A-Za-z0-9])R(?=[\s,;/)])",
        ),
    ),
    ("Java", re.compile(r"(?<![a-z0-9])java(?![a-z0-9])", re.IGNORECASE)),
    ("C++", re.compile(r"c\+\+", re.IGNORECASE)),
    ("C#", re.compile(r"c#", re.IGNORECASE)),
    ("JavaScript", re.compile(r"javascript", re.IGNORECASE)),
    ("TypeScript", re.compile(r"typescript", re.IGNORECASE)),
    ("Go", re.compile(r"golang|\bgo programming\b", re.IGNORECASE)),
    ("Rust", re.compile(r"(?<![a-z0-9])rust(?![a-z0-9])", re.IGNORECASE)),
    ("Scala", re.compile(r"(?<![a-z0-9])scala(?![a-z0-9])", re.IGNORECASE)),
    ("Machine Learning", re.compile(r"machine learning", re.IGNORECASE)),
    ("Deep Learning", re.compile(r"deep learning", re.IGNORECASE)),
    ("Generative AI", re.compile(r"generative ai|genai|gen ai", re.IGNORECASE)),
    (
        "Large Language Models",
        re.compile(
            r"large language models?|(?<![a-z0-9])llms?(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "Natural Language Processing",
        re.compile(
            r"natural language processing|(?<![a-z0-9])nlp(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    ("Computer Vision", re.compile(r"computer vision", re.IGNORECASE)),
    (
        "Reinforcement Learning",
        re.compile(r"reinforcement learning", re.IGNORECASE),
    ),
    (
        "Statistics",
        re.compile(r"statistics|statistical modeling", re.IGNORECASE),
    ),
    (
        "A/B Testing",
        re.compile(r"a/b test|ab test|experimentation", re.IGNORECASE),
    ),
    ("PyTorch", re.compile(r"pytorch", re.IGNORECASE)),
    ("TensorFlow", re.compile(r"tensorflow", re.IGNORECASE)),
    ("scikit-learn", re.compile(r"scikit-learn|sklearn", re.IGNORECASE)),
    ("Keras", re.compile(r"(?<![a-z0-9])keras(?![a-z0-9])", re.IGNORECASE)),
    ("JAX", re.compile(r"(?<![a-z0-9])jax(?![a-z0-9])", re.IGNORECASE)),
    ("Apache Spark", re.compile(r"apache spark|pyspark", re.IGNORECASE)),
    ("Databricks", re.compile(r"databricks", re.IGNORECASE)),
    (
        "AWS",
        re.compile(
            r"amazon web services|(?<![a-z0-9])aws(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "Azure",
        re.compile(
            r"microsoft azure|(?<![a-z0-9])azure(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "Google Cloud",
        re.compile(
            r"google cloud|(?<![a-z0-9])gcp(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    ("Docker", re.compile(r"(?<![a-z0-9])docker(?![a-z0-9])", re.IGNORECASE)),
    (
        "Kubernetes",
        re.compile(r"kubernetes|(?<![a-z0-9])k8s(?![a-z0-9])", re.IGNORECASE),
    ),
    ("Kafka", re.compile(r"apache kafka|(?<![a-z0-9])kafka(?![a-z0-9])", re.IGNORECASE)),
    ("Snowflake", re.compile(r"(?<![a-z0-9])snowflake(?![a-z0-9])", re.IGNORECASE)),
    ("Git", re.compile(r"github|gitlab|(?<![a-z0-9])git(?![a-z0-9])", re.IGNORECASE)),
    ("Linux", re.compile(r"(?<![a-z0-9])linux(?![a-z0-9])", re.IGNORECASE)),
    ("MLOps", re.compile(r"mlops|machine learning operations", re.IGNORECASE)),
    (
        "Data Engineering",
        re.compile(r"data engineering|data pipelines?|etl pipelines?", re.IGNORECASE),
    ),
    ("Distributed Systems", re.compile(r"distributed systems?", re.IGNORECASE)),
    ("REST APIs", re.compile(r"restful|rest apis?|api development", re.IGNORECASE)),
    ("Terraform", re.compile(r"(?<![a-z0-9])terraform(?![a-z0-9])", re.IGNORECASE)),
    (
        "CI/CD",
        re.compile(
            r"ci/cd|continuous integration|continuous delivery",
            re.IGNORECASE,
        ),
    ),
]

DATA_SCIENTIST_PATTERN = re.compile(r"\bdata scientists?\b", re.IGNORECASE)
DATA_SCIENCE_ADJACENT_PATTERN = re.compile(r"\bdata science\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the validated Step 3 career-market panel."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def text(value: object) -> str:
    if value is None:
        return ""
    result = str(value).strip()
    if result.lower() in {"nan", "none", "null", "[none]"}:
        return ""
    return result


def optional_text(value: object) -> str | None:
    result = text(value)
    return result if result else None


def parse_date(value: object) -> date | None:
    raw = text(value)
    if not raw:
        return None

    for date_format in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, date_format).date()
        except ValueError:
            continue
    return None


def parse_number(value: object) -> float | None:
    raw = text(value)
    if not raw:
        return None
    cleaned = re.sub(r"[$,]", "", raw)
    if not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", cleaned):
        return None
    return float(cleaned)


def parse_array(value: object) -> list[str]:
    raw = text(value)
    if not raw:
        return []

    parsed: object
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []

    if not isinstance(parsed, list):
        return []

    result = []
    for item in parsed:
        cleaned = text(item)
        if cleaned:
            result.append(cleaned)
    return result


def canonical_skill_name(value: object) -> str:
    cleaned = re.sub(r"\s+", " ", text(value))
    if not cleaned:
        return ""
    return SKILL_CANONICAL_NAMES.get(cleaned.casefold(), cleaned)


def normalized_skill_set(values: list[str]) -> set[str]:
    result = set()
    casefold_to_name: dict[str, str] = {}
    for value in values:
        canonical = canonical_skill_name(value)
        if not canonical:
            continue
        key = canonical.casefold()
        if key not in casefold_to_name:
            casefold_to_name[key] = canonical
    result.update(casefold_to_name.values())
    return result


def extract_description_skills(row: dict[str, str]) -> set[str]:
    searchable = " ".join(
        [
            text(row.get("TITLE_RAW")),
            text(row.get("TITLE_CLEAN")),
            text(row.get("BODY")),
        ]
    )
    result = set()
    for skill_name, pattern in SKILL_PATTERNS:
        if pattern.search(searchable):
            result.add(skill_name)
    return result


def clean_role(candidate: dict[str, str]) -> tuple[str, str, str]:
    role_segment = text(candidate.get("ROLE_SEGMENT"))
    role_reason = text(candidate.get("ROLE_MATCH_REASON"))
    title = " ".join(
        [text(candidate.get("TITLE_RAW")), text(candidate.get("TITLE_CLEAN"))]
    )

    # Only an explicit Data Scientist title remains in the core tier. A Data
    # Science Analyst is a valuable adjacent role, but it is not relabeled as a
    # Data Scientist.
    if (
        role_segment == "Data Scientist"
        and DATA_SCIENCE_ADJACENT_PATTERN.search(title)
        and not DATA_SCIENTIST_PATTERN.search(title)
    ):
        role_segment = "Data Science Adjacent Role"
        role_reason = "Explicit Data Science adjacent-role title"

    scope_tier = "Core" if role_segment == "Data Scientist" else "Adjacent"
    return role_segment, role_reason, scope_tier


def annualization_factor(pay_period: object) -> float | None:
    period = text(pay_period).casefold()
    factor_by_period = {
        "annual": 1.0,
        "annually": 1.0,
        "year": 1.0,
        "yearly": 1.0,
        "hour": 2080.0,
        "hourly": 2080.0,
        "week": 52.0,
        "weekly": 52.0,
        "month": 12.0,
        "monthly": 12.0,
        "day": 260.0,
        "daily": 260.0,
    }
    return factor_by_period.get(period)


def clean_salary(row: dict[str, str]) -> dict[str, object]:
    salary_from = parse_number(row.get("SALARY_FROM"))
    salary_to = parse_number(row.get("SALARY_TO"))

    if salary_from is None and salary_to is None:
        return {
            "SALARY_FROM_ANNUAL": None,
            "SALARY_TO_ANNUAL": None,
            "SALARY_MIDPOINT_ANNUAL": None,
            "SALARY_CURRENCY": None,
            "SALARY_STATUS": "Missing",
            "SALARY_RANGE_COMPLETENESS": None,
        }

    if (salary_from or 0) <= 0 and (salary_to or 0) <= 0:
        return {
            "SALARY_FROM_ANNUAL": None,
            "SALARY_TO_ANNUAL": None,
            "SALARY_MIDPOINT_ANNUAL": None,
            "SALARY_CURRENCY": None,
            "SALARY_STATUS": "Zero placeholder treated as missing",
            "SALARY_RANGE_COMPLETENESS": None,
        }

    factor = annualization_factor(row.get("ORIGINAL_PAY_PERIOD"))
    if factor is None:
        return {
            "SALARY_FROM_ANNUAL": None,
            "SALARY_TO_ANNUAL": None,
            "SALARY_MIDPOINT_ANNUAL": None,
            "SALARY_CURRENCY": None,
            "SALARY_STATUS": "Unsupported or missing pay period",
            "SALARY_RANGE_COMPLETENESS": None,
        }

    annual_values = []
    for value in (salary_from, salary_to):
        if value is not None and value > 0:
            annual_value = value * factor
            if MIN_PLAUSIBLE_ANNUAL_SALARY <= annual_value <= MAX_PLAUSIBLE_ANNUAL_SALARY:
                annual_values.append(annual_value)

    if not annual_values:
        return {
            "SALARY_FROM_ANNUAL": None,
            "SALARY_TO_ANNUAL": None,
            "SALARY_MIDPOINT_ANNUAL": None,
            "SALARY_CURRENCY": None,
            "SALARY_STATUS": "Outside plausible annual range",
            "SALARY_RANGE_COMPLETENESS": None,
        }

    annual_from = min(annual_values)
    annual_to = max(annual_values)
    midpoint = (annual_from + annual_to) / 2.0
    completeness = "Complete range" if len(annual_values) == 2 else "Single valid bound"
    salary_raw = text(row.get("SALARY"))
    currency = "USD" if "USD" in salary_raw.upper() else None

    return {
        "SALARY_FROM_ANNUAL": annual_from,
        "SALARY_TO_ANNUAL": annual_to,
        "SALARY_MIDPOINT_ANNUAL": midpoint,
        "SALARY_CURRENCY": currency,
        "SALARY_STATUS": "Valid",
        "SALARY_RANGE_COMPLETENESS": completeness,
    }


def salary_is_outlier(value: object, lower: float, upper: float) -> bool:
    return bool(pd.notna(value) and (float(value) < lower or float(value) > upper))


def internship_flag(row: dict[str, str]) -> bool:
    source_value = text(row.get("IS_INTERNSHIP")).casefold()
    title_value = text(row.get("TITLE_RAW"))
    return source_value in {"true", "1", "yes"} or bool(
        re.search(r"\bintern(ship)?\b", title_value, re.IGNORECASE)
    )


def seniority_proxy(row: dict[str, str], is_internship: bool) -> str:
    title = text(row.get("TITLE_RAW"))
    if is_internship:
        return "Internship"
    if re.search(
        r"\b(staff|principal|lead|architect|distinguished)\b",
        title,
        re.IGNORECASE,
    ):
        return "Staff/Principal/Lead title"
    if re.search(r"\b(senior|sr\.?|level iv|level v)\b", title, re.IGNORECASE):
        return "Senior title"
    if re.search(
        r"\b(entry|entry-level|junior|jr\.?|new grad|graduate|associate)\b",
        title,
        re.IGNORECASE,
    ):
        return "Entry-level title"
    return "No explicit seniority signal"


def clean_education(value: object) -> str:
    lowered = text(value).casefold()
    if "doctor" in lowered:
        return "Doctoral degree"
    if "master" in lowered:
        return "Master's degree"
    if "bachelor" in lowered:
        return "Bachelor's degree"
    if "associate" in lowered:
        return "Associate degree"
    if "high school" in lowered:
        return "High school or equivalent"
    return "Unknown"


def clean_employment(value: object, is_internship: bool) -> str:
    lowered = text(value).casefold()
    if is_internship or "intern" in lowered:
        return "Internship"
    if "full" in lowered:
        return "Full Time"
    if "part" in lowered:
        return "Part Time"
    if "contract" in lowered:
        return "Contract"
    if "temporary" in lowered:
        return "Temporary"
    return "Unknown"


def clean_remote(value: object) -> str:
    lowered = text(value).casefold()
    if "not remote" in lowered or "onsite" in lowered or "on-site" in lowered:
        return "Onsite"
    if "hybrid" in lowered:
        return "Hybrid"
    if "remote" in lowered:
        return "Remote"
    return "Unknown"


def clean_staffing(value: object) -> bool | None:
    lowered = text(value).casefold()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def clean_location(row: dict[str, str], remote_status: str) -> tuple[str, str | None, str | None, str | None]:
    city = optional_text(row.get("CITY_NAME"))
    county = optional_text(row.get("COUNTY_NAME"))
    state = optional_text(row.get("STATE_NAME"))
    raw_location = optional_text(row.get("LOCATION"))

    if city:
        location = city
    elif state:
        location = state
    elif raw_location:
        location = raw_location
    elif remote_status == "Remote":
        location = "Remote / Unspecified"
    else:
        location = "Unspecified"

    return location, city, county, state


def load_candidates(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing role-candidate audit: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as candidate_file:
        reader = csv.DictReader(candidate_file)
        result = {}
        for row in reader:
            job_id = text(row.get("ID"))
            if not job_id:
                continue
            if job_id in result:
                raise RuntimeError(f"Duplicate candidate ID in audit file: {job_id}")
            result[job_id] = row

    if not result:
        raise RuntimeError("The role-candidate audit contains no included jobs.")
    return result


def stream_selected_source_rows(
    source_path: Path,
    candidate_ids: set[str],
) -> tuple[dict[str, dict[str, str]], int, int, int]:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_path}")

    selected: dict[str, dict[str, str]] = {}
    source_rows = 0
    exact_naics_rows = 0
    exact_naics_window_rows = 0
    csv.field_size_limit(sys.maxsize)

    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        for row in reader:
            source_rows += 1
            raw_naics = text(row.get("NAICS_2022_6"))
            if raw_naics.endswith(".0"):
                raw_naics = raw_naics[:-2]
            if raw_naics == NAICS_CODE:
                exact_naics_rows += 1
                posted_date = parse_date(row.get("POSTED"))
                if posted_date is not None and WINDOW_START <= posted_date <= WINDOW_END:
                    exact_naics_window_rows += 1

            job_id = text(row.get("ID"))
            if job_id in candidate_ids:
                current = selected.get(job_id)
                if current is None or text(row.get("LAST_UPDATED_TIMESTAMP")) >= text(
                    current.get("LAST_UPDATED_TIMESTAMP")
                ):
                    selected[job_id] = row

            if source_rows % 10_000 == 0:
                print(
                    f"Processed {source_rows:,} source rows; "
                    f"found {len(selected):,} of {len(candidate_ids):,} candidate IDs"
                )

    return selected, source_rows, exact_naics_rows, exact_naics_window_rows


def build_outputs(
    source_rows: dict[str, dict[str, str]],
    candidates: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_rows = []
    skill_rows = []

    for job_id in sorted(candidates):
        row = source_rows[job_id]
        candidate = candidates[job_id]

        posted_date = parse_date(row.get("POSTED"))
        expired_date = parse_date(row.get("EXPIRED"))
        if posted_date is None or not WINDOW_START <= posted_date <= WINDOW_END:
            raise RuntimeError(f"Candidate {job_id} is outside the validated date window.")

        naics_code = text(row.get("NAICS_2022_6"))
        if naics_code.endswith(".0"):
            naics_code = naics_code[:-2]
        if naics_code != NAICS_CODE:
            raise RuntimeError(f"Candidate {job_id} is outside NAICS {NAICS_CODE}.")

        role_segment, role_reason, scope_tier = clean_role(candidate)
        is_internship = internship_flag(row)
        remote_status = clean_remote(row.get("REMOTE_TYPE_NAME"))
        location, city, county, state = clean_location(row, remote_status)
        salary = clean_salary(row)
        min_experience = parse_number(row.get("MIN_YEARS_EXPERIENCE"))
        max_experience = parse_number(row.get("MAX_YEARS_EXPERIENCE"))

        structured_values = []
        for column in STRUCTURED_SKILL_COLUMNS:
            structured_values.extend(parse_array(row.get(column)))
        structured_skills = normalized_skill_set(structured_values)
        description_skills = extract_description_skills(row)
        all_skills = sorted(
            structured_skills | description_skills,
            key=str.casefold,
        )

        if structured_skills and description_skills:
            skill_source = "Structured fields + description keywords"
        elif structured_skills:
            skill_source = "Structured fields"
        elif description_skills:
            skill_source = "Description keywords"
        else:
            skill_source = "Unavailable"

        for skill in all_skills:
            if skill in structured_skills and skill in description_skills:
                origin = "Both"
            elif skill in structured_skills:
                origin = "Structured field"
            else:
                origin = "Description keyword"

            skill_rows.append(
                {
                    "JOB_ID": job_id,
                    "SCOPE_TIER": scope_tier,
                    "ROLE_SEGMENT": role_segment,
                    "SKILL": skill,
                    "SKILL_ORIGIN": origin,
                }
            )

        experience_source = (
            "Reported years"
            if min_experience is not None or max_experience is not None
            else "Title-based seniority proxy"
        )

        panel_row = {
            "JOB_ID": job_id,
            "POSTED_DATE": posted_date.isoformat(),
            "EXPIRED_DATE": expired_date.isoformat() if expired_date else None,
            "TITLE_RAW": optional_text(row.get("TITLE_RAW")),
            "TITLE_CLEAN": optional_text(row.get("TITLE_CLEAN")),
            "TITLE_STANDARDIZED": optional_text(row.get("TITLE_NAME")),
            "SCOPE_TIER": scope_tier,
            "ROLE_SEGMENT": role_segment,
            "ROLE_MATCH_REASON": role_reason,
            "SENIORITY_PROXY": seniority_proxy(row, is_internship),
            "EXPERIENCE_DATA_SOURCE": experience_source,
            "MIN_YEARS_EXPERIENCE": min_experience,
            "MAX_YEARS_EXPERIENCE": max_experience,
            "EDUCATION_REQUIRED": clean_education(row.get("MIN_EDULEVELS_NAME")),
            "EDUCATION_RAW": optional_text(row.get("MIN_EDULEVELS_NAME")),
            "EMPLOYMENT_TYPE": clean_employment(
                row.get("EMPLOYMENT_TYPE_NAME"), is_internship
            ),
            "IS_INTERNSHIP": is_internship,
            "REMOTE_STATUS": remote_status,
            "REMOTE_RAW": optional_text(row.get("REMOTE_TYPE_NAME")),
            "LOCATION_CLEAN": location,
            "CITY": city,
            "COUNTY": county,
            "STATE": state,
            "COMPANY_NAME": optional_text(row.get("COMPANY_NAME")),
            "COMPANY_RAW": optional_text(row.get("COMPANY_RAW")),
            "COMPANY_IS_STAFFING": clean_staffing(row.get("COMPANY_IS_STAFFING")),
            **salary,
            "SALARY_OUTLIER_FLAG": False,
            "SALARY_RAW": optional_text(row.get("SALARY")),
            "ORIGINAL_PAY_PERIOD": optional_text(row.get("ORIGINAL_PAY_PERIOD")),
            "SKILLS_CLEAN": json.dumps(all_skills, ensure_ascii=False),
            "SKILL_COUNT": len(all_skills),
            "SKILL_SOURCE": skill_source,
            "SKILLS_RAW": optional_text(row.get("SKILLS_NAME")),
            "SPECIALIZED_SKILLS_RAW": optional_text(
                row.get("SPECIALIZED_SKILLS_NAME")
            ),
            "COMMON_SKILLS_RAW": optional_text(row.get("COMMON_SKILLS_NAME")),
            "SOFTWARE_SKILLS_RAW": optional_text(row.get("SOFTWARE_SKILLS_NAME")),
            "CERTIFICATIONS_RAW": optional_text(row.get("CERTIFICATIONS_NAME")),
            "ONET": optional_text(row.get("ONET")),
            "ONET_NAME": optional_text(row.get("ONET_NAME")),
            "SOC_2021_5": optional_text(row.get("SOC_2021_5")),
            "SOC_2021_5_NAME": optional_text(row.get("SOC_2021_5_NAME")),
            "SOC_5": optional_text(row.get("SOC_5")),
            "SOC_5_NAME": optional_text(row.get("SOC_5_NAME")),
            "NAICS_2022_6": NAICS_CODE,
            "NAICS_2022_6_NAME": NAICS_NAME,
            "INDUSTRY_MATCH_METHOD": "Exact NAICS_2022_6 code",
            "NEAR_DUPLICATE_FLAG": False,
            "SOURCE_DUPLICATE_INDICATOR": optional_text(row.get("DUPLICATES")),
        }
        panel_rows.append(panel_row)

    panel = pd.DataFrame(panel_rows)
    skills = pd.DataFrame(skill_rows)

    fingerprints = []
    for panel_row in panel_rows:
        fingerprint = "||".join(
            [
                text(panel_row.get("COMPANY_NAME")).casefold(),
                text(panel_row.get("TITLE_RAW")).casefold(),
                text(panel_row.get("POSTED_DATE")),
                text(panel_row.get("CITY")).casefold(),
                text(panel_row.get("STATE")).casefold(),
            ]
        )
        fingerprints.append(fingerprint)

    fingerprint_counts = Counter(fingerprints)
    panel["NEAR_DUPLICATE_FLAG"] = [
        fingerprint_counts[fingerprint] > 1 for fingerprint in fingerprints
    ]

    valid_salary = panel.loc[
        panel["SALARY_STATUS"].eq("Valid"), "SALARY_MIDPOINT_ANNUAL"
    ].dropna()

    salary_statistics = {
        "salary_q1": None,
        "salary_q3": None,
        "salary_iqr_lower": MIN_PLAUSIBLE_ANNUAL_SALARY,
        "salary_iqr_upper": MAX_PLAUSIBLE_ANNUAL_SALARY,
    }
    if len(valid_salary) >= 4:
        salary_q1 = float(valid_salary.quantile(0.25))
        salary_q3 = float(valid_salary.quantile(0.75))
        salary_iqr = salary_q3 - salary_q1
        lower = max(MIN_PLAUSIBLE_ANNUAL_SALARY, salary_q1 - 1.5 * salary_iqr)
        upper = min(MAX_PLAUSIBLE_ANNUAL_SALARY, salary_q3 + 1.5 * salary_iqr)
        outlier_flags = []
        for value in panel["SALARY_MIDPOINT_ANNUAL"]:
            outlier_flags.append(salary_is_outlier(value, lower, upper))
        panel["SALARY_OUTLIER_FLAG"] = outlier_flags
        salary_statistics = {
            "salary_q1": salary_q1,
            "salary_q3": salary_q3,
            "salary_iqr_lower": lower,
            "salary_iqr_upper": upper,
        }

    panel = panel.sort_values(["POSTED_DATE", "JOB_ID"]).reset_index(drop=True)
    if not skills.empty:
        skills = skills.sort_values(["JOB_ID", "SKILL"]).reset_index(drop=True)

    salary_stats = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in salary_statistics.items()]
    )
    return panel, skills, salary_stats


def data_dictionary() -> pd.DataFrame:
    rows = [
        ("JOB_ID", "string", "Unique posting identifier."),
        ("POSTED_DATE", "date", "Posting date standardized to ISO format."),
        ("TITLE_RAW", "string", "Original posting title."),
        ("TITLE_CLEAN", "string", "Cleaned Lightcast title."),
        ("TITLE_STANDARDIZED", "string", "Standardized Lightcast title label."),
        ("SCOPE_TIER", "category", "Core Data Scientist or adjacent role."),
        ("ROLE_SEGMENT", "category", "Mutually exclusive pathway segment."),
        ("ROLE_MATCH_REASON", "string", "Rule supporting role inclusion."),
        ("SENIORITY_PROXY", "category", "Title-based seniority signal."),
        ("MIN_YEARS_EXPERIENCE", "number", "Reported minimum experience."),
        ("MAX_YEARS_EXPERIENCE", "number", "Reported maximum experience."),
        ("EDUCATION_REQUIRED", "category", "Normalized minimum education."),
        ("EMPLOYMENT_TYPE", "category", "Normalized employment type."),
        ("REMOTE_STATUS", "category", "Remote, hybrid, onsite, or unknown."),
        ("LOCATION_CLEAN", "string", "Best available cleaned location."),
        ("COMPANY_NAME", "string", "Employer name."),
        ("SALARY_FROM_ANNUAL", "number", "Annualized lower salary bound."),
        ("SALARY_TO_ANNUAL", "number", "Annualized upper salary bound."),
        ("SALARY_MIDPOINT_ANNUAL", "number", "Annualized salary midpoint."),
        ("SALARY_STATUS", "category", "Salary cleaning result."),
        ("SALARY_OUTLIER_FLAG", "boolean", "1.5-IQR salary flag."),
        ("SKILLS_CLEAN", "JSON array", "Normalized distinct skill names."),
        ("SKILL_COUNT", "integer", "Number of normalized skills."),
        ("SKILL_SOURCE", "category", "Structured or description source."),
        ("ONET_NAME", "string", "Source O*NET occupation label."),
        ("SOC_2021_5_NAME", "string", "2021 detailed SOC label; audit-only."),
        ("NAICS_2022_6", "string", "Exact six-digit 2022 NAICS code."),
        ("NEAR_DUPLICATE_FLAG", "boolean", "Repeated posting fingerprint flag."),
    ]
    return pd.DataFrame(rows, columns=["variable", "type", "description"])


def audit_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def cleaning_audit(
    source_row_count: int,
    exact_naics_rows: int,
    exact_naics_window_rows: int,
    candidates: dict[str, dict[str, str]],
    panel: pd.DataFrame,
    salary_stats: pd.DataFrame,
) -> pd.DataFrame:
    salary_values = dict(zip(salary_stats["metric"], salary_stats["value"]))
    metrics = [
        ("source_rows", source_row_count, "All rows in the 2024 Lightcast CSV."),
        ("exact_naics_513210_rows", exact_naics_rows, "Exact code match in the source CSV."),
        ("exact_naics_rows_in_window", exact_naics_window_rows, "Exact code match from 2024-05-01 through 2024-09-30."),
        ("approved_candidate_ids", len(candidates), "Title-based pathway audit."),
        ("final_panel_rows", len(panel), "One row per approved candidate ID."),
        ("core_data_scientist_rows", int(panel["SCOPE_TIER"].eq("Core").sum()), "Explicit Data Scientist title."),
        ("adjacent_pathway_rows", int(panel["SCOPE_TIER"].eq("Adjacent").sum()), "Approved adjacent role titles."),
        ("valid_salary_rows", int(panel["SALARY_STATUS"].eq("Valid").sum()), "Annualized usable salary."),
        ("salary_q1", salary_values.get("salary_q1"), "First quartile among valid salaries."),
        ("salary_q3", salary_values.get("salary_q3"), "Third quartile among valid salaries."),
        ("salary_iqr_lower", salary_values.get("salary_iqr_lower"), "Lower outlier threshold."),
        ("salary_iqr_upper", salary_values.get("salary_iqr_upper"), "Upper outlier threshold."),
        ("salary_outlier_rows", int(panel["SALARY_OUTLIER_FLAG"].sum()), "Flagged but retained."),
        ("rows_with_reported_experience", int(panel[["MIN_YEARS_EXPERIENCE", "MAX_YEARS_EXPERIENCE"]].notna().any(axis=1).sum()), "No years are imputed."),
        ("known_remote_rows", int(panel["REMOTE_STATUS"].ne("Unknown").sum()), "Unknown is not treated as onsite."),
        ("rows_with_state", int(panel["STATE"].notna().sum()), "Usable state name."),
        ("rows_with_skills", int(panel["SKILL_COUNT"].gt(0).sum()), "Structured or controlled keyword skill."),
        ("near_duplicate_flagged_rows", int(panel["NEAR_DUPLICATE_FLAG"].sum()), "Employer/title/date/location fingerprint."),
    ]
    rows = []
    for metric, value, notes in metrics:
        rows.append(
            {
                "metric": metric,
                "value": audit_value(value),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidates)
    (
        selected_source_rows,
        source_row_count,
        exact_naics_rows,
        exact_naics_window_rows,
    ) = stream_selected_source_rows(
        args.source,
        set(candidates),
    )

    missing_ids = sorted(set(candidates) - set(selected_source_rows))
    if missing_ids:
        raise RuntimeError(
            "Candidate IDs missing from source CSV: " + ", ".join(missing_ids)
        )

    panel, skills, salary_stats = build_outputs(selected_source_rows, candidates)

    if panel["JOB_ID"].nunique() != len(panel):
        raise RuntimeError("The final panel does not contain one unique row per job ID.")
    if not panel["NAICS_2022_6"].eq(NAICS_CODE).all():
        raise RuntimeError(f"At least one final row is outside NAICS {NAICS_CODE}.")
    if int(panel["SCOPE_TIER"].eq("Core").sum()) != 1:
        raise RuntimeError("Expected exactly one explicit core Data Scientist posting.")

    dictionary = data_dictionary()
    audit = cleaning_audit(
        source_row_count,
        exact_naics_rows,
        exact_naics_window_rows,
        candidates,
        panel,
        salary_stats,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "career_market_panel.csv"
    skills_path = args.output_dir / "career_market_skills.csv"
    dictionary_path = args.output_dir / "career_market_data_dictionary.csv"
    audit_path = args.output_dir / "career_market_cleaning_audit.csv"

    panel.to_csv(panel_path, index=False)
    skills.to_csv(skills_path, index=False)
    dictionary.to_csv(dictionary_path, index=False)
    audit.to_csv(audit_path, index=False)

    print("\nSTEP 3 PANEL VALIDATION")
    print(f"Source rows scanned: {source_row_count:,}")
    print(f"Approved candidate IDs: {len(candidates):,}")
    print(f"Final panel rows: {len(panel):,}")
    print(f"Distinct final job IDs: {panel['JOB_ID'].nunique():,}")
    print(f"Core Data Scientist rows: {int(panel['SCOPE_TIER'].eq('Core').sum()):,}")
    print(f"Adjacent pathway rows: {int(panel['SCOPE_TIER'].eq('Adjacent').sum()):,}")
    print(f"Rows with valid salary: {int(panel['SALARY_STATUS'].eq('Valid').sum()):,}")
    print(f"Rows with reported experience: {int(panel[['MIN_YEARS_EXPERIENCE', 'MAX_YEARS_EXPERIENCE']].notna().any(axis=1).sum()):,}")
    print(f"Rows with known remote status: {int(panel['REMOTE_STATUS'].ne('Unknown').sum()):,}")
    print(f"Rows with at least one skill: {int(panel['SKILL_COUNT'].gt(0).sum()):,}")
    print(f"Near-duplicate rows flagged: {int(panel['NEAR_DUPLICATE_FLAG'].sum()):,}")

    print("\nROLE SEGMENTS")
    print(panel["ROLE_SEGMENT"].value_counts().to_string())

    print("\nTOP SKILLS")
    if skills.empty:
        print("No skills were extracted.")
    else:
        top_skills = skills.groupby("SKILL")["JOB_ID"].nunique().sort_values(
            ascending=False
        ).head(25)
        print(top_skills.to_string())

    print("\nCREATED FILES")
    for path in (panel_path, skills_path, dictionary_path, audit_path):
        print(f"{path}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

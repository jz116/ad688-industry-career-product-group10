"""Audit the raw NAICS values behind the final 66-row market panel.

Run from the repository root after restructure_data.py and
prepare_market_panel.py have completed.
"""

from pathlib import Path
import csv


PANEL_PATH = Path("data/processed/career_market_panel.csv")
NAICS_PATH = Path("_output/naics_details.csv")
AUDIT_PATH = Path("data/processed/target_naics_source_audit.csv")

TARGET_CODE_2022 = "513210"
TARGET_CODE_LEGACY = "511210"
TARGET_NAME = "software publishers"


if not PANEL_PATH.exists():
    raise FileNotFoundError(f"Missing {PANEL_PATH}")
if not NAICS_PATH.exists():
    raise FileNotFoundError(f"Missing {NAICS_PATH}")


with PANEL_PATH.open("r", encoding="utf-8-sig", newline="") as panel_file:
    panel_rows = list(csv.DictReader(panel_file))

panel_by_id = {str(row["JOB_ID"]).strip(): row for row in panel_rows}
target_ids = set(panel_by_id)

naics_by_id = {}

with NAICS_PATH.open("r", encoding="utf-8-sig", newline="") as naics_file:
    reader = csv.DictReader(naics_file)
    available_columns = set(reader.fieldnames or [])

    required_columns = {"ID", "NAICS6", "NAICS6_NAME"}
    missing_columns = required_columns - available_columns

    if missing_columns:
        raise RuntimeError(
            "naics_details.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    for row in reader:
        job_id = str(row.get("ID", "")).strip()
        if job_id in target_ids:
            naics_by_id[job_id] = {
                "NAICS6_RAW": str(row.get("NAICS6", "")).strip(),
                "NAICS6_NAME_RAW": str(row.get("NAICS6_NAME", "")).strip(),
            }


audit_rows = []

for job_id, panel_row in panel_by_id.items():
    raw_values = naics_by_id.get(
        job_id,
        {"NAICS6_RAW": "", "NAICS6_NAME_RAW": ""},
    )

    raw_code = raw_values["NAICS6_RAW"]
    raw_name = raw_values["NAICS6_NAME_RAW"]

    code_2022_match = TARGET_CODE_2022 in raw_code
    legacy_code_match = TARGET_CODE_LEGACY in raw_code
    name_match = TARGET_NAME in raw_name.lower()

    audit_rows.append(
        {
            "JOB_ID": job_id,
            "TITLE_RAW": panel_row.get("TITLE_RAW", ""),
            "COMPANY_NAME": panel_row.get("COMPANY_NAME", ""),
            "NAICS6_RAW": raw_code,
            "NAICS6_NAME_RAW": raw_name,
            "MATCHES_513210": code_2022_match,
            "MATCHES_LEGACY_511210": legacy_code_match,
            "MATCHES_SOFTWARE_PUBLISHERS_NAME": name_match,
        }
    )


AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "JOB_ID",
    "TITLE_RAW",
    "COMPANY_NAME",
    "NAICS6_RAW",
    "NAICS6_NAME_RAW",
    "MATCHES_513210",
    "MATCHES_LEGACY_511210",
    "MATCHES_SOFTWARE_PUBLISHERS_NAME",
]

with AUDIT_PATH.open("w", encoding="utf-8", newline="") as audit_file:
    writer = csv.DictWriter(audit_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(audit_rows)


def count_where(field_name):
    return sum(bool(row[field_name]) for row in audit_rows)


pair_counts = {}
for row in audit_rows:
    pair = (row["NAICS6_RAW"], row["NAICS6_NAME_RAW"])
    pair_counts[pair] = pair_counts.get(pair, 0) + 1


print(f"Panel jobs: {len(panel_rows)}")
print(f"Jobs found in naics_details.csv: {len(naics_by_id)}")
print(f"Raw code contains 513210: {count_where('MATCHES_513210')}")
print(f"Raw code contains legacy 511210: {count_where('MATCHES_LEGACY_511210')}")
print(
    "Raw name contains Software Publishers: "
    f"{count_where('MATCHES_SOFTWARE_PUBLISHERS_NAME')}"
)

print("\nRaw NAICS code/name combinations:")

def pair_sort_key(item):
    return (-item[1], item[0])


for (raw_code, raw_name), count in sorted(pair_counts.items(), key=pair_sort_key):
    print(f"count={count}")
    print(f"  NAICS6_RAW={raw_code!r}")
    print(f"  NAICS6_NAME_RAW={raw_name!r}")

print(f"\nWrote {AUDIT_PATH}")

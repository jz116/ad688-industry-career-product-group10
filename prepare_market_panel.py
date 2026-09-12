"""Build the Step 2 Software Publishers AI/ML career-market dataset.

Run from the repository root:
    python prepare_market_panel.py

Inputs:
    data/MET_CareerCompass_2026/*.parquet

Outputs:
    data/processed/career_market_panel.csv
    data/processed/career_market_skills.csv
    data/processed/career_market_data_dictionary.csv
    data/processed/career_market_cleaning_audit.csv
"""

from pathlib import Path
import shutil

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType


DATA_DIR = Path("data/MET_CareerCompass_2026")
OUTPUT_DIR = Path("data/processed")
SPARK_TMP_DIR = Path(".spark-tmp")

NAICS_CODE = "513210"
NAICS_NAME = "Software Publishers"
WINDOW_START = "2026-01-01"
WINDOW_END = "2026-07-31"

MIN_PLAUSIBLE_ANNUAL_SALARY = 15_000.0
MAX_PLAUSIBLE_ANNUAL_SALARY = 1_000_000.0


def nonblank(column_name):
    """Return a trimmed string or NULL for null/blank input."""
    value = F.trim(F.col(column_name))
    return F.when(F.col(column_name).isNull() | (value == ""), F.lit(None)).otherwise(value)


def safe_number(column_name):
    """Convert a simple numeric string to double without ANSI cast errors."""
    value = F.regexp_replace(
        F.trim(F.coalesce(F.col(column_name), F.lit(""))),
        r"[,$]",
        "",
    )
    return F.when(
        value.rlike(r"^-?[0-9]+(?:\.[0-9]+)?$"),
        value.cast("double"),
    ).otherwise(F.lit(None).cast("double"))


def safe_date(column_name):
    """Convert a date string safely; blank or malformed values become NULL."""
    return F.expr(f"try_cast(trim(`{column_name}`) as date)")


def safe_timestamp(column_name):
    """Convert a timestamp string safely; blank or malformed values become NULL."""
    return F.expr(f"try_cast(trim(`{column_name}`) as timestamp)")


def parse_json_array(column_name):
    """Parse a JSON array string; blanks and malformed values become empty arrays."""
    array_schema = ArrayType(StringType())
    empty_array = F.from_json(F.lit("[]"), array_schema)
    return F.coalesce(F.from_json(F.trim(F.col(column_name)), array_schema), empty_array)


def write_single_csv(frame, target_path):
    """Write a Spark DataFrame as one real CSV file instead of a part-file folder."""
    target_path = Path(target_path)
    temporary_path = target_path.parent / f".{target_path.stem}_spark_output"

    if temporary_path.exists():
        shutil.rmtree(temporary_path)

    frame.coalesce(1).write.mode("overwrite").option("header", True).csv(
        str(temporary_path)
    )

    part_files = list(temporary_path.glob("part-*.csv"))
    if len(part_files) != 1:
        raise RuntimeError(
            f"Expected one Spark part file for {target_path.name}, found {len(part_files)}."
        )

    if target_path.exists():
        target_path.unlink()

    shutil.move(str(part_files[0]), str(target_path))
    shutil.rmtree(temporary_path)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPARK_TMP_DIR.mkdir(exist_ok=True)

parquet_files = sorted(str(path) for path in DATA_DIR.rglob("*.parquet"))
if len(parquet_files) != 17:
    raise RuntimeError(f"Expected 17 Parquet files, found {len(parquet_files)}.")

spark = (
    SparkSession.builder.appName("PrepareCareerMarketPanel")
    .config("spark.local.dir", str(SPARK_TMP_DIR.resolve()))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

source_columns = [
    "ID",
    "LAST_UPDATED_TIMESTAMP",
    "DUPLICATES",
    "POSTED",
    "EXPIRED",
    "TITLE_RAW",
    "TITLE_NAME",
    "TITLE_CLEAN",
    "BODY",
    "COMPANY_NAME",
    "COMPANY_RAW",
    "COMPANY_IS_STAFFING",
    "EDUCATION_LEVELS_NAME",
    "MIN_EDULEVELS_NAME",
    "MAX_EDULEVELS_NAME",
    "EMPLOYMENT_TYPE_NAME",
    "MIN_YEARS_EXPERIENCE",
    "MAX_YEARS_EXPERIENCE",
    "IS_INTERNSHIP",
    "SALARY",
    "SALARY_FROM",
    "SALARY_TO",
    "ORIGINAL_PAY_PERIOD",
    "REMOTE_TYPE_NAME",
    "LOCATION",
    "CITY_NAME",
    "COUNTY_NAME",
    "STATE_NAME",
    "SKILLS_NAME",
    "SPECIALIZED_SKILLS_NAME",
    "COMMON_SKILLS_NAME",
    "SOFTWARE_SKILLS_NAME",
    "CERTIFICATIONS_NAME",
    "ONET_NAME",
    "SOC_5",
    "SOC_5_NAME",
    "NAICS_2022_6",
    "NAICS_2022_6_NAME",
]

raw = spark.read.parquet(*parquet_files).select(*source_columns)
raw_row_count = raw.count()

# Remove rows without a usable job ID, then deterministically retain the latest
# record if a source ID appears more than once.
identified = raw.filter(nonblank("ID").isNotNull())
identified_row_count = identified.count()

id_window = Window.partitionBy("ID").orderBy(
    safe_timestamp("LAST_UPDATED_TIMESTAMP").desc_nulls_last()
)

deduplicated = (
    identified.withColumn("_id_rank", F.row_number().over(id_window))
    .filter(F.col("_id_rank") == 1)
    .drop("_id_rank")
    .withColumn("POSTED_DATE", safe_date("POSTED"))
    .withColumn("EXPIRED_DATE", safe_date("EXPIRED"))
    .persist(StorageLevel.DISK_ONLY)
)
deduplicated_row_count = deduplicated.count()

industry_code_match = F.coalesce(
    F.col("NAICS_2022_6"), F.lit("")
).contains(NAICS_CODE)
industry_name_match = F.lower(
    F.coalesce(F.col("NAICS_2022_6_NAME"), F.lit(""))
).contains(NAICS_NAME.lower())
industry_condition = industry_code_match | industry_name_match

date_condition = F.col("POSTED_DATE").between(WINDOW_START, WINDOW_END)

industry = (
    deduplicated.filter(industry_condition & date_condition)
    .withColumn(
        "INDUSTRY_MATCH_METHOD",
        F.when(industry_code_match, "NAICS code")
        .when(industry_name_match, "Canonical industry name fallback"),
    )
    .persist(StorageLevel.DISK_ONLY)
)
industry_row_count = industry.count()

# Role matching deliberately excludes BODY. A skill or team name in the job
# description is not enough to make a posting part of the target occupation.
role_profile = (
    industry.withColumn(
        "_title_text",
        F.lower(
            F.concat_ws(
                " ",
                F.coalesce(F.col("TITLE_RAW"), F.lit("")),
                F.coalesce(F.col("TITLE_CLEAN"), F.lit("")),
            )
        ),
    )
    .withColumn(
        "_mapping_text",
        F.lower(
            F.concat_ws(
                " ",
                F.coalesce(F.col("TITLE_NAME"), F.lit("")),
                F.coalesce(F.col("ONET_NAME"), F.lit("")),
                F.coalesce(F.col("SOC_5_NAME"), F.lit("")),
            )
        ),
    )
)

title_text = F.col("_title_text")
mapping_text = F.col("_mapping_text")

explicit_data_scientist = title_text.rlike(r"\bdata scientists?\b")
data_science_leadership = title_text.rlike(r"\bdata science\b") & title_text.rlike(
    r"\b(scientist|analyst|manager|director|lead|intern)\b"
)
mapped_data_scientist = mapping_text.rlike(r"\bdata scientists?\b")

ml_term = title_text.rlike(
    r"machine learning|(^|[^a-z0-9])ml([^a-z0-9]|$)"
)

# The dot is excluded immediately before AI so a product/web name ending in
# '.ai' does not by itself qualify a posting.
ai_term = title_text.rlike(
    r"artificial intelligence|(^|[^a-z0-9.])ai([^a-z0-9]|$)"
)

technical_role = title_text.rlike(
    r"\b(engineer|scientist|developer|architect|intern)\b"
)
applied_decision = title_text.rlike(r"\b(applied|decision) scientists?\b")
research_scientist = title_text.rlike(r"\bresearch scientists?\b") & (
    title_text.rlike(r"\bdata science\b") | ml_term | ai_term
)
analytics_engineer = title_text.rlike(r"\banalytics engineers?\b")

role_segment = (
    F.when(
        explicit_data_scientist
        | data_science_leadership
        | mapped_data_scientist,
        "Data Scientist",
    )
    .when(applied_decision, "Applied or Decision Scientist")
    .when(research_scientist, "Data/AI Research Scientist")
    .when(ml_term & technical_role, "Machine Learning Technical Role")
    .when(ai_term & technical_role, "AI Technical Role")
    .when(analytics_engineer, "Analytics Engineer")
)

role_reason = (
    F.when(explicit_data_scientist, "Explicit Data Scientist title")
    .when(data_science_leadership, "Explicit Data Science role title")
    .when(mapped_data_scientist, "SOC/O*NET Data Scientists mapping")
    .when(applied_decision, "Applied or Decision Scientist title")
    .when(research_scientist, "Data/AI Research Scientist title")
    .when(ml_term & technical_role, "Explicit ML technical title")
    .when(ai_term & technical_role, "Explicit AI technical title")
    .when(analytics_engineer, "Explicit Analytics Engineer title")
)

cohort = (
    role_profile.withColumn("ROLE_SEGMENT", role_segment)
    .withColumn("ROLE_MATCH_REASON", role_reason)
    .filter(F.col("ROLE_SEGMENT").isNotNull())
    .withColumn(
        "SCOPE_TIER",
        F.when(F.col("ROLE_SEGMENT") == "Data Scientist", "Core").otherwise(
            "Adjacent"
        ),
    )
    .persist(StorageLevel.DISK_ONLY)
)

cohort_row_count = cohort.count()
if cohort_row_count == 0:
    raise RuntimeError("The documented industry-career filter returned zero rows.")

# Salary cleaning and annualization.
salary_from_numeric = safe_number("SALARY_FROM")
salary_to_numeric = safe_number("SALARY_TO")
pay_period = F.lower(F.trim(F.coalesce(F.col("ORIGINAL_PAY_PERIOD"), F.lit(""))))

annualization_factor = (
    F.when(pay_period.isin("annual", "annually", "year", "yearly"), 1.0)
    .when(pay_period.isin("hour", "hourly"), 2080.0)
    .when(pay_period.isin("week", "weekly"), 52.0)
    .when(pay_period.isin("month", "monthly"), 12.0)
    .when(pay_period.isin("day", "daily"), 260.0)
)

positive_from = F.when(salary_from_numeric > 0, salary_from_numeric)
positive_to = F.when(salary_to_numeric > 0, salary_to_numeric)
annual_from_unordered = positive_from * annualization_factor
annual_to_unordered = positive_to * annualization_factor

ordered_from = F.when(
    annual_from_unordered.isNotNull() & annual_to_unordered.isNotNull(),
    F.least(annual_from_unordered, annual_to_unordered),
).otherwise(F.coalesce(annual_from_unordered, annual_to_unordered))

ordered_to = F.when(
    annual_from_unordered.isNotNull() & annual_to_unordered.isNotNull(),
    F.greatest(annual_from_unordered, annual_to_unordered),
).otherwise(F.coalesce(annual_to_unordered, annual_from_unordered))

valid_from = F.when(
    ordered_from.between(
        MIN_PLAUSIBLE_ANNUAL_SALARY, MAX_PLAUSIBLE_ANNUAL_SALARY
    ),
    ordered_from,
)
valid_to = F.when(
    ordered_to.between(MIN_PLAUSIBLE_ANNUAL_SALARY, MAX_PLAUSIBLE_ANNUAL_SALARY),
    ordered_to,
)

salary_status = (
    F.when(
        salary_from_numeric.isNull() & salary_to_numeric.isNull(),
        "Missing",
    )
    .when(
        (F.coalesce(salary_from_numeric, F.lit(0.0)) == 0)
        & (F.coalesce(salary_to_numeric, F.lit(0.0)) == 0),
        "Zero placeholder treated as missing",
    )
    .when(annualization_factor.isNull(), "Unsupported or missing pay period")
    .when(valid_from.isNull() & valid_to.isNull(), "Outside plausible annual range")
    .when(valid_from.isNull() | valid_to.isNull(), "Partially valid")
    .otherwise("Valid")
)

prepared = (
    cohort.withColumn("_salary_from_numeric", salary_from_numeric)
    .withColumn("_salary_to_numeric", salary_to_numeric)
    .withColumn("SALARY_FROM_ANNUAL", valid_from)
    .withColumn("SALARY_TO_ANNUAL", valid_to)
    .withColumn(
        "SALARY_MIDPOINT_ANNUAL",
        F.when(
            valid_from.isNotNull() & valid_to.isNotNull(),
            (valid_from + valid_to) / 2.0,
        ).otherwise(F.coalesce(valid_from, valid_to)),
    )
    .withColumn("SALARY_STATUS", salary_status)
    .withColumn(
        "SALARY_CURRENCY",
        F.when(F.upper(F.coalesce(F.col("SALARY"), F.lit(""))).contains("USD"), "USD"),
    )
)

# Flag statistical salary outliers without deleting plausible high-paying jobs.
valid_salary_count = prepared.filter(
    F.col("SALARY_MIDPOINT_ANNUAL").isNotNull()
).count()

if valid_salary_count >= 4:
    salary_q1, salary_q3 = prepared.approxQuantile(
        "SALARY_MIDPOINT_ANNUAL", [0.25, 0.75], 0.0
    )
    salary_iqr = salary_q3 - salary_q1
    salary_iqr_lower = max(
        MIN_PLAUSIBLE_ANNUAL_SALARY, salary_q1 - 1.5 * salary_iqr
    )
    salary_iqr_upper = min(
        MAX_PLAUSIBLE_ANNUAL_SALARY, salary_q3 + 1.5 * salary_iqr
    )
else:
    salary_q1 = None
    salary_q3 = None
    salary_iqr_lower = MIN_PLAUSIBLE_ANNUAL_SALARY
    salary_iqr_upper = MAX_PLAUSIBLE_ANNUAL_SALARY

prepared = prepared.withColumn(
    "SALARY_OUTLIER_FLAG",
    F.when(F.col("SALARY_MIDPOINT_ANNUAL").isNull(), False).otherwise(
        (F.col("SALARY_MIDPOINT_ANNUAL") < salary_iqr_lower)
        | (F.col("SALARY_MIDPOINT_ANNUAL") > salary_iqr_upper)
    ),
)

# Reported experience remains numeric when supplied. Because all target records
# currently lack it, the separate proxy uses title language and never invents years.
min_experience = safe_number("MIN_YEARS_EXPERIENCE")
max_experience = safe_number("MAX_YEARS_EXPERIENCE")
title_lower = F.lower(F.coalesce(F.col("TITLE_RAW"), F.lit("")))
internship_flag = (
    F.lower(F.trim(F.coalesce(F.col("IS_INTERNSHIP"), F.lit("")))).isin(
        "true", "1", "yes"
    )
    | title_lower.rlike(r"\bintern(ship)?\b")
)

seniority_proxy = (
    F.when(internship_flag, "Internship")
    .when(
        title_lower.rlike(r"\b(staff|principal|lead|architect|distinguished)\b"),
        "Staff/Principal/Lead title",
    )
    .when(title_lower.rlike(r"\b(senior|sr\.?|level iv|level v)\b"), "Senior title")
    .when(
        title_lower.rlike(
            r"\b(entry|entry-level|junior|jr\.?|new grad|graduate|associate)\b"
        ),
        "Entry-level title",
    )
    .otherwise("No explicit seniority signal")
)

education_lower = F.lower(
    F.coalesce(F.col("MIN_EDULEVELS_NAME"), F.lit(""))
)
education_clean = (
    F.when(education_lower.contains("doctor"), "Doctoral degree")
    .when(education_lower.contains("master"), "Master's degree")
    .when(education_lower.contains("bachelor"), "Bachelor's degree")
    .when(education_lower.contains("associate"), "Associate degree")
    .when(education_lower.contains("high school"), "High school or equivalent")
    .otherwise("Unknown")
)

employment_lower = F.lower(
    F.coalesce(F.col("EMPLOYMENT_TYPE_NAME"), F.lit(""))
)
employment_clean = (
    F.when(internship_flag | employment_lower.contains("intern"), "Internship")
    .when(employment_lower.contains("full"), "Full Time")
    .when(employment_lower.contains("part"), "Part Time")
    .when(employment_lower.contains("contract"), "Contract")
    .when(employment_lower.contains("temporary"), "Temporary")
    .otherwise("Unknown")
)

remote_lower = F.lower(F.coalesce(F.col("REMOTE_TYPE_NAME"), F.lit("")))
remote_clean = (
    F.when(
        remote_lower.contains("not remote")
        | remote_lower.contains("onsite")
        | remote_lower.contains("on-site"),
        "Onsite",
    )
    .when(remote_lower.contains("hybrid"), "Hybrid")
    .when(remote_lower.contains("remote"), "Remote")
    .otherwise("Unknown")
)

city_clean = nonblank("CITY_NAME")
county_clean = nonblank("COUNTY_NAME")
state_clean = nonblank("STATE_NAME")
location_parts = F.concat_ws(", ", city_clean, state_clean)
location_clean = (
    F.when((city_clean.isNotNull()) | (state_clean.isNotNull()), location_parts)
    .when(nonblank("LOCATION").isNotNull(), nonblank("LOCATION"))
    .when(remote_clean == "Remote", "Remote / Unspecified")
    .otherwise("Unspecified")
)

# Structured arrays are retained when available. Controlled keyword extraction
# from BODY supplements them because structured skills are mostly missing.
structured_skill_columns = [
    "SKILLS_NAME",
    "SPECIALIZED_SKILLS_NAME",
    "COMMON_SKILLS_NAME",
    "SOFTWARE_SKILLS_NAME",
]

empty_string_array = F.from_json(F.lit("[]"), ArrayType(StringType()))
structured_skills = empty_string_array
for structured_column in structured_skill_columns:
    structured_skills = F.array_union(
        structured_skills, parse_json_array(structured_column)
    )

skill_text = F.lower(
    F.concat_ws(
        " ",
        F.coalesce(F.col("TITLE_RAW"), F.lit("")),
        F.coalesce(F.col("BODY"), F.lit("")),
        F.coalesce(F.col("SKILLS_NAME"), F.lit("")),
        F.coalesce(F.col("SPECIALIZED_SKILLS_NAME"), F.lit("")),
        F.coalesce(F.col("COMMON_SKILLS_NAME"), F.lit("")),
        F.coalesce(F.col("SOFTWARE_SKILLS_NAME"), F.lit("")),
    )
)

skill_patterns = [
    ("Python", r"(^|[^a-z0-9])python([^a-z0-9]|$)"),
    ("SQL", r"(^|[^a-z0-9])sql([^a-z0-9]|$)"),
    ("R", r"\br programming\b|\br language\b|(^|[\s,(])r([\s,;/)]|$)"),
    ("Java", r"(^|[^a-z0-9])java([^a-z0-9]|$)"),
    ("C++", r"c\+\+"),
    ("C#", r"c#"),
    ("JavaScript", r"javascript"),
    ("TypeScript", r"typescript"),
    ("Go", r"golang|\bgo programming\b"),
    ("Rust", r"(^|[^a-z0-9])rust([^a-z0-9]|$)"),
    ("Scala", r"(^|[^a-z0-9])scala([^a-z0-9]|$)"),
    ("Machine Learning", r"machine learning"),
    ("Deep Learning", r"deep learning"),
    ("Generative AI", r"generative ai|genai|gen ai"),
    ("Large Language Models", r"large language models?|(^|[^a-z0-9])llms?([^a-z0-9]|$)"),
    ("Natural Language Processing", r"natural language processing|(^|[^a-z0-9])nlp([^a-z0-9]|$)"),
    ("Computer Vision", r"computer vision"),
    ("Reinforcement Learning", r"reinforcement learning"),
    ("Statistics", r"statistics|statistical modeling"),
    ("A/B Testing", r"a/b test|ab test|experimentation"),
    ("PyTorch", r"pytorch"),
    ("TensorFlow", r"tensorflow"),
    ("scikit-learn", r"scikit-learn|sklearn"),
    ("Keras", r"(^|[^a-z0-9])keras([^a-z0-9]|$)"),
    ("JAX", r"(^|[^a-z0-9])jax([^a-z0-9]|$)"),
    ("Apache Spark", r"apache spark|pyspark"),
    ("Databricks", r"databricks"),
    ("AWS", r"amazon web services|(^|[^a-z0-9])aws([^a-z0-9]|$)"),
    ("Azure", r"microsoft azure|(^|[^a-z0-9])azure([^a-z0-9]|$)"),
    ("Google Cloud", r"google cloud|(^|[^a-z0-9])gcp([^a-z0-9]|$)"),
    ("Docker", r"(^|[^a-z0-9])docker([^a-z0-9]|$)"),
    ("Kubernetes", r"kubernetes|(^|[^a-z0-9])k8s([^a-z0-9]|$)"),
    ("Kafka", r"apache kafka|(^|[^a-z0-9])kafka([^a-z0-9]|$)"),
    ("Snowflake", r"(^|[^a-z0-9])snowflake([^a-z0-9]|$)"),
    ("Git", r"github|gitlab|(^|[^a-z0-9])git([^a-z0-9]|$)"),
    ("Linux", r"(^|[^a-z0-9])linux([^a-z0-9]|$)"),
    ("MLOps", r"mlops|machine learning operations"),
    ("Data Engineering", r"data engineering|data pipelines?|etl pipelines?"),
    ("Distributed Systems", r"distributed systems?"),
    ("REST APIs", r"restful|rest apis?|api development"),
    ("Terraform", r"(^|[^a-z0-9])terraform([^a-z0-9]|$)"),
    ("CI/CD", r"ci/cd|continuous integration|continuous delivery"),
]

extracted_skill_values = []
for skill_name, skill_pattern in skill_patterns:
    extracted_skill_values.append(
        F.when(skill_text.rlike(skill_pattern), F.lit(skill_name))
    )

extracted_skills = F.array_compact(F.array(*extracted_skill_values))

prepared = (
    prepared.withColumn("MIN_YEARS_EXPERIENCE_CLEAN", min_experience)
    .withColumn("MAX_YEARS_EXPERIENCE_CLEAN", max_experience)
    .withColumn("SENIORITY_PROXY", seniority_proxy)
    .withColumn(
        "EXPERIENCE_DATA_SOURCE",
        F.when(
            min_experience.isNotNull() | max_experience.isNotNull(),
            "Reported years",
        ).otherwise("Title-based seniority proxy"),
    )
    .withColumn("EDUCATION_REQUIRED", education_clean)
    .withColumn("EMPLOYMENT_TYPE_CLEAN", employment_clean)
    .withColumn("IS_INTERNSHIP_CLEAN", internship_flag)
    .withColumn("REMOTE_STATUS", remote_clean)
    .withColumn("CITY_CLEAN", city_clean)
    .withColumn("COUNTY_CLEAN", county_clean)
    .withColumn("STATE_CLEAN", state_clean)
    .withColumn("LOCATION_CLEAN", location_clean)
    .withColumn("_structured_skills", F.array_sort(F.array_distinct(structured_skills)))
    .withColumn("_extracted_skills", F.array_sort(F.array_distinct(extracted_skills)))
    .withColumn(
        "_all_skills",
        F.array_sort(
            F.array_distinct(
                F.concat(F.col("_structured_skills"), F.col("_extracted_skills"))
            )
        ),
    )
    .withColumn("SKILLS_CLEAN", F.to_json(F.col("_all_skills")))
    .withColumn("SKILL_COUNT", F.size(F.col("_all_skills")))
    .withColumn(
        "SKILL_SOURCE",
        F.when(
            (F.size(F.col("_structured_skills")) > 0)
            & (F.size(F.col("_extracted_skills")) > 0),
            "Structured fields + description keywords",
        )
        .when(
            F.size(F.col("_structured_skills")) > 0,
            "Structured fields",
        )
        .when(
            F.size(F.col("_extracted_skills")) > 0,
            "Description keywords",
        )
        .otherwise("Unavailable"),
    )
)

# Near-duplicate fingerprints are flagged, not automatically deleted, because
# employers may post the same role separately for distinct locations.
fingerprint = F.sha2(
    F.lower(
        F.concat_ws(
            "||",
            F.coalesce(F.col("COMPANY_NAME"), F.lit("")),
            F.coalesce(F.col("TITLE_RAW"), F.lit("")),
            F.coalesce(F.col("POSTED"), F.lit("")),
            F.coalesce(F.col("CITY_NAME"), F.lit("")),
            F.coalesce(F.col("STATE_NAME"), F.lit("")),
        )
    ),
    256,
)
fingerprint_window = Window.partitionBy(fingerprint)

prepared = prepared.withColumn(
    "NEAR_DUPLICATE_FLAG", F.count(F.lit(1)).over(fingerprint_window) > 1
)

staffing_lower = F.lower(
    F.trim(F.coalesce(F.col("COMPANY_IS_STAFFING"), F.lit("")))
)
staffing_clean = (
    F.when(staffing_lower.isin("true", "1", "yes"), True)
    .when(staffing_lower.isin("false", "0", "no"), False)
    .otherwise(F.lit(None).cast("boolean"))
)

panel = prepared.select(
    nonblank("ID").alias("JOB_ID"),
    F.col("POSTED_DATE"),
    F.col("EXPIRED_DATE"),
    nonblank("TITLE_RAW").alias("TITLE_RAW"),
    nonblank("TITLE_CLEAN").alias("TITLE_CLEAN"),
    nonblank("TITLE_NAME").alias("TITLE_STANDARDIZED"),
    F.col("SCOPE_TIER"),
    F.col("ROLE_SEGMENT"),
    F.col("ROLE_MATCH_REASON"),
    F.col("SENIORITY_PROXY"),
    F.col("EXPERIENCE_DATA_SOURCE"),
    F.col("MIN_YEARS_EXPERIENCE_CLEAN").alias("MIN_YEARS_EXPERIENCE"),
    F.col("MAX_YEARS_EXPERIENCE_CLEAN").alias("MAX_YEARS_EXPERIENCE"),
    F.col("EDUCATION_REQUIRED"),
    nonblank("MIN_EDULEVELS_NAME").alias("EDUCATION_RAW"),
    F.col("EMPLOYMENT_TYPE_CLEAN").alias("EMPLOYMENT_TYPE"),
    F.col("IS_INTERNSHIP_CLEAN").alias("IS_INTERNSHIP"),
    F.col("REMOTE_STATUS"),
    nonblank("REMOTE_TYPE_NAME").alias("REMOTE_RAW"),
    F.col("LOCATION_CLEAN"),
    F.col("CITY_CLEAN").alias("CITY"),
    F.col("COUNTY_CLEAN").alias("COUNTY"),
    F.col("STATE_CLEAN").alias("STATE"),
    nonblank("COMPANY_NAME").alias("COMPANY_NAME"),
    nonblank("COMPANY_RAW").alias("COMPANY_RAW"),
    staffing_clean.alias("COMPANY_IS_STAFFING"),
    F.col("SALARY_FROM_ANNUAL"),
    F.col("SALARY_TO_ANNUAL"),
    F.col("SALARY_MIDPOINT_ANNUAL"),
    F.col("SALARY_CURRENCY"),
    F.col("SALARY_STATUS"),
    F.col("SALARY_OUTLIER_FLAG"),
    nonblank("SALARY").alias("SALARY_RAW"),
    nonblank("ORIGINAL_PAY_PERIOD").alias("ORIGINAL_PAY_PERIOD"),
    F.col("SKILLS_CLEAN"),
    F.col("SKILL_COUNT"),
    F.col("SKILL_SOURCE"),
    nonblank("SKILLS_NAME").alias("SKILLS_RAW"),
    nonblank("SPECIALIZED_SKILLS_NAME").alias("SPECIALIZED_SKILLS_RAW"),
    nonblank("SOFTWARE_SKILLS_NAME").alias("SOFTWARE_SKILLS_RAW"),
    nonblank("CERTIFICATIONS_NAME").alias("CERTIFICATIONS_RAW"),
    nonblank("ONET_NAME").alias("ONET_NAME"),
    nonblank("SOC_5").alias("SOC_5"),
    nonblank("SOC_5_NAME").alias("SOC_5_NAME"),
    F.lit(NAICS_CODE).alias("NAICS_2022_6"),
    F.lit(NAICS_NAME).alias("NAICS_2022_6_NAME"),
    F.col("INDUSTRY_MATCH_METHOD"),
    F.col("NEAR_DUPLICATE_FLAG"),
    nonblank("DUPLICATES").alias("SOURCE_DUPLICATE_INDICATOR"),
)

panel = panel.persist(StorageLevel.DISK_ONLY)
panel_row_count = panel.count()
distinct_job_count = panel.select("JOB_ID").distinct().count()

if panel_row_count != cohort_row_count:
    raise RuntimeError("Panel row count changed unexpectedly during cleaning.")
if panel_row_count != distinct_job_count:
    raise RuntimeError("Final panel does not contain exactly one row per JOB_ID.")
if panel.filter(F.col("NAICS_2022_6") == NAICS_CODE).count() != panel_row_count:
    raise RuntimeError("At least one final row is outside NAICS 513210.")

skills_long = (
    prepared.select(
        nonblank("ID").alias("JOB_ID"),
        F.col("SCOPE_TIER"),
        F.col("ROLE_SEGMENT"),
        F.explode(F.col("_all_skills")).alias("SKILL"),
    )
    .dropDuplicates(["JOB_ID", "SKILL"])
    .orderBy("JOB_ID", "SKILL")
)

data_dictionary_rows = [
    ("JOB_ID", "string", "Unique posting identifier.", "Blank IDs removed; duplicate IDs retain latest update."),
    ("POSTED_DATE", "date", "Posting date.", "Converted from POSTED."),
    ("EXPIRED_DATE", "date", "Expiration date.", "Converted from EXPIRED; missing in this cohort."),
    ("TITLE_RAW", "string", "Original posting title.", "Trimmed; preserved for auditing."),
    ("TITLE_CLEAN", "string", "Cleaned posting title.", "Trimmed Lightcast field."),
    ("TITLE_STANDARDIZED", "string", "Standardized title or occupation label.", "Trimmed Lightcast field."),
    ("SCOPE_TIER", "category", "Core Data Scientist or adjacent technical role.", "Created by documented role rules."),
    ("ROLE_SEGMENT", "category", "Data Scientist, AI, ML, or another included role segment.", "Mutually exclusive title/mapping classification."),
    ("ROLE_MATCH_REASON", "string", "Rule that caused the posting to be included.", "Created for filter transparency."),
    ("SENIORITY_PROXY", "category", "Title-based seniority indicator.", "Used because reported years are missing; does not impute years."),
    ("EXPERIENCE_DATA_SOURCE", "category", "Source of experience information.", "Reported years when present; otherwise title proxy."),
    ("MIN_YEARS_EXPERIENCE", "double", "Reported minimum experience years.", "Safely converted; missing values remain null."),
    ("MAX_YEARS_EXPERIENCE", "double", "Reported maximum experience years.", "Safely converted; missing values remain null."),
    ("EDUCATION_REQUIRED", "category", "Normalized minimum education requirement.", "Standardized from MIN_EDULEVELS_NAME."),
    ("EDUCATION_RAW", "string", "Original minimum education label.", "Trimmed and preserved."),
    ("EMPLOYMENT_TYPE", "category", "Normalized employment type.", "Full Time, Internship, other known type, or Unknown."),
    ("IS_INTERNSHIP", "boolean", "Whether the posting is an internship.", "Source flag supplemented with title pattern."),
    ("REMOTE_STATUS", "category", "Remote, Hybrid, Onsite, or Unknown.", "Unknown is never treated as onsite."),
    ("REMOTE_RAW", "string", "Original remote-work label.", "Trimmed and preserved."),
    ("LOCATION_CLEAN", "string", "Combined cleaned location.", "City/state preferred; raw location fallback."),
    ("CITY", "string", "Cleaned city name.", "Blank converted to null."),
    ("COUNTY", "string", "Cleaned county name.", "Blank converted to null."),
    ("STATE", "string", "Cleaned state name.", "Blank converted to null."),
    ("COMPANY_NAME", "string", "Cleaned employer name.", "Trimmed Lightcast field."),
    ("COMPANY_RAW", "string", "Original employer name.", "Trimmed and preserved."),
    ("COMPANY_IS_STAFFING", "boolean", "Whether employer is identified as staffing firm.", "Normalized from string boolean."),
    ("SALARY_FROM_ANNUAL", "double", "Cleaned annual lower salary bound in USD when identified.", "Zero/nonpositive and implausible values become null; supported periods annualized."),
    ("SALARY_TO_ANNUAL", "double", "Cleaned annual upper salary bound in USD when identified.", "Bounds ordered; zero/nonpositive and implausible values become null."),
    ("SALARY_MIDPOINT_ANNUAL", "double", "Midpoint of valid annual salary bounds.", "Uses one valid bound when only one exists."),
    ("SALARY_CURRENCY", "string", "Currency identified from raw salary text.", "USD only when explicitly identified."),
    ("SALARY_STATUS", "category", "Salary cleaning result.", "Distinguishes valid, missing, zero placeholder, and invalid values."),
    ("SALARY_OUTLIER_FLAG", "boolean", "Whether midpoint is outside 1.5-IQR limits.", "Flagged but not deleted."),
    ("SALARY_RAW", "string", "Original salary description.", "Preserved for audit."),
    ("ORIGINAL_PAY_PERIOD", "string", "Original salary pay period.", "Retained; recognized periods annualized."),
    ("SKILLS_CLEAN", "JSON array", "Deduplicated normalized skills.", "Union of structured fields and controlled description keywords."),
    ("SKILL_COUNT", "integer", "Number of cleaned skills identified.", "Count of SKILLS_CLEAN elements."),
    ("SKILL_SOURCE", "category", "Origin of cleaned skills.", "Structured fields, description keywords, both, or unavailable."),
    ("SKILLS_RAW", "string", "Original general skills array.", "Preserved when supplied."),
    ("SPECIALIZED_SKILLS_RAW", "string", "Original specialized skills array.", "Preserved when supplied."),
    ("SOFTWARE_SKILLS_RAW", "string", "Original software skills array.", "Preserved when supplied."),
    ("CERTIFICATIONS_RAW", "string", "Original certifications array.", "Preserved when supplied."),
    ("ONET_NAME", "string", "O*NET occupation name.", "Trimmed Lightcast field."),
    ("SOC_5", "string", "Detailed SOC code.", "Trimmed Lightcast field."),
    ("SOC_5_NAME", "string", "Detailed SOC occupation name.", "Trimmed Lightcast field."),
    ("NAICS_2022_6", "string", "Six-digit 2022 NAICS code.", "Standardized to 513210 after a code or canonical-name match."),
    ("NAICS_2022_6_NAME", "string", "Six-digit 2022 NAICS industry name.", "Standardized to Software Publishers."),
    ("INDUSTRY_MATCH_METHOD", "category", "Whether the source matched by code or canonical industry name.", "Preserves transparency when the raw code is missing."),
    ("NEAR_DUPLICATE_FLAG", "boolean", "Same employer/title/date/city/state fingerprint appears more than once.", "Flagged for review rather than automatically removed."),
    ("SOURCE_DUPLICATE_INDICATOR", "string", "Original source duplicate indicator.", "Preserved for auditing."),
]

dictionary = spark.createDataFrame(
    data_dictionary_rows,
    ["variable", "type", "description", "cleaning_notes"],
)

core_count = panel.filter(F.col("SCOPE_TIER") == "Core").count()
adjacent_count = panel.filter(F.col("SCOPE_TIER") == "Adjacent").count()
salary_zero_count = prepared.filter(
    (F.coalesce(F.col("_salary_from_numeric"), F.lit(0.0)) == 0)
    & (F.coalesce(F.col("_salary_to_numeric"), F.lit(0.0)) == 0)
    & (
        F.col("_salary_from_numeric").isNotNull()
        | F.col("_salary_to_numeric").isNotNull()
    )
).count()
salary_outlier_count = panel.filter(F.col("SALARY_OUTLIER_FLAG")).count()
missing_experience_count = panel.filter(
    F.col("MIN_YEARS_EXPERIENCE").isNull()
    & F.col("MAX_YEARS_EXPERIENCE").isNull()
).count()
unknown_remote_count = panel.filter(F.col("REMOTE_STATUS") == "Unknown").count()
missing_location_count = panel.filter(F.col("STATE").isNull()).count()
near_duplicate_count = panel.filter(F.col("NEAR_DUPLICATE_FLAG")).count()
structured_skill_count = prepared.filter(
    F.size(F.col("_structured_skills")) > 0
).count()
description_skill_count = prepared.filter(
    F.size(F.col("_extracted_skills")) > 0
).count()
industry_name_fallback_count = panel.filter(
    F.col("INDUSTRY_MATCH_METHOD") == "Canonical industry name fallback"
).count()

audit_rows = [
    ("source_rows", str(raw_row_count), "All rows across the 17 Parquet partitions."),
    ("rows_with_nonblank_id", str(identified_row_count), "Rows eligible for ID deduplication."),
    ("duplicate_id_rows_removed", str(identified_row_count - deduplicated_row_count), "Latest updated record retained per ID."),
    ("industry_rows_in_date_window", str(industry_row_count), f"NAICS {NAICS_CODE}, {WINDOW_START} through {WINDOW_END}."),
    ("industry_name_fallback_rows", str(industry_name_fallback_count), "Canonical Software Publishers name used when the raw code did not contain 513210."),
    ("final_panel_rows", str(panel_row_count), "One row per included posting."),
    ("core_data_scientist_rows", str(core_count), "Core tier."),
    ("adjacent_ai_ml_rows", str(adjacent_count), "Adjacent tier."),
    ("valid_salary_rows", str(valid_salary_count), "Nonmissing midpoint after zero and plausibility cleaning."),
    ("zero_salary_placeholders", str(salary_zero_count), "Converted to missing, not interpreted as pay."),
    ("salary_iqr_q1", str(salary_q1), "Q1 among cleaned midpoint salaries."),
    ("salary_iqr_q3", str(salary_q3), "Q3 among cleaned midpoint salaries."),
    ("salary_outlier_lower_limit", str(salary_iqr_lower), "Lower 1.5-IQR flagging limit, bounded by plausibility threshold."),
    ("salary_outlier_upper_limit", str(salary_iqr_upper), "Upper 1.5-IQR flagging limit, bounded by plausibility threshold."),
    ("salary_outlier_rows", str(salary_outlier_count), "Flagged but retained."),
    ("rows_missing_reported_experience", str(missing_experience_count), "Title proxy supplied separately; no years imputed."),
    ("unknown_remote_rows", str(unknown_remote_count), "Not recoded as onsite."),
    ("rows_missing_state", str(missing_location_count), "Location incompleteness retained."),
    ("rows_with_structured_skills", str(structured_skill_count), "At least one parsed structured skill."),
    ("rows_with_description_skills", str(description_skill_count), "At least one controlled keyword match."),
    ("near_duplicate_flagged_rows", str(near_duplicate_count), "Flagged using employer/title/date/city/state fingerprint."),
]

audit = spark.createDataFrame(audit_rows, ["metric", "value", "notes"])

write_single_csv(panel.orderBy("POSTED_DATE", "JOB_ID"), OUTPUT_DIR / "career_market_panel.csv")
write_single_csv(skills_long, OUTPUT_DIR / "career_market_skills.csv")
write_single_csv(dictionary, OUTPUT_DIR / "career_market_data_dictionary.csv")
write_single_csv(audit, OUTPUT_DIR / "career_market_cleaning_audit.csv")

print("\nValidation summary")
print(f"Source rows: {raw_row_count:,}")
print(f"Industry postings in stated window: {industry_row_count:,}")
print(f"Industry rows matched by canonical-name fallback: {industry_name_fallback_count:,}")
print(f"Final panel rows: {panel_row_count:,}")
print(f"Distinct final job IDs: {distinct_job_count:,}")
print(f"Core Data Scientist rows: {core_count:,}")
print(f"Adjacent AI/ML rows: {adjacent_count:,}")
print(f"Valid midpoint salaries: {valid_salary_count:,}")
print(f"Zero salary placeholders set to missing: {salary_zero_count:,}")
print(f"Salary outliers flagged and retained: {salary_outlier_count:,}")
print(f"Rows missing reported experience years: {missing_experience_count:,}")
print(f"Rows with description-derived skills: {description_skill_count:,}")
print(f"Near-duplicate rows flagged: {near_duplicate_count:,}")

print("\nTop extracted/structured skills by distinct job count")
skills_long.groupBy("SKILL").agg(
    F.countDistinct("JOB_ID").alias("job_count")
).orderBy(F.desc("job_count"), "SKILL").show(25, truncate=False)

print("\nCreated files")
for output_name in [
    "career_market_panel.csv",
    "career_market_skills.csv",
    "career_market_data_dictionary.csv",
    "career_market_cleaning_audit.csv",
]:
    output_path = OUTPUT_DIR / output_name
    print(f"{output_path}: {output_path.stat().st_size:,} bytes")

panel.unpersist()
cohort.unpersist()
industry.unpersist()
deduplicated.unpersist()
spark.stop()

print("\nCareer-market dataset preparation completed successfully.")

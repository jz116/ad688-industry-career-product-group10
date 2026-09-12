from __future__ import annotations

import os
import shutil
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data" / "MET_CareerCompass_2026"
OUTPUT_DIR = REPO_DIR / "_output"
SPARK_TEMP_DIR = REPO_DIR / ".spark-tmp"

OUTPUT_DIR.mkdir(exist_ok=True)
SPARK_TEMP_DIR.mkdir(exist_ok=True)

# Keep Spark temporary files away from the small /tmp filesystem.
os.environ["SPARK_LOCAL_DIRS"] = str(SPARK_TEMP_DIR)


def write_single_csv(frame: DataFrame, filename: str) -> None:
    """Write a Spark DataFrame as one CSV file with a header."""
    destination = OUTPUT_DIR / filename
    temporary_directory = OUTPUT_DIR / f".{filename}.spark-parts"

    if temporary_directory.exists():
        shutil.rmtree(temporary_directory)

    (
        frame.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .option("quoteAll", "true")
        .csv(str(temporary_directory))
    )

    part_files = list(temporary_directory.glob("part-*.csv"))

    if len(part_files) != 1:
        raise RuntimeError(
            f"Expected one CSV part for {filename}, found {len(part_files)}"
        )

    if destination.exists():
        destination.unlink()

    shutil.move(str(part_files[0]), str(destination))
    shutil.rmtree(temporary_directory)

    print(f"Wrote {destination}")


def duplicate_key_count(frame: DataFrame, key: str) -> int:
    return (
        frame.groupBy(key)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )


def main() -> None:
    parquet_files = sorted(DATA_DIR.rglob("*.parquet"))

    if len(parquet_files) != 17:
        raise RuntimeError(
            f"Expected 17 Parquet files, found {len(parquet_files)} "
            f"inside {DATA_DIR}"
        )

    spark = (
        SparkSession.builder
        .appName("JobPostingsRestructuring")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        raw = spark.read.parquet(
            *[str(path) for path in parquet_files]
        )

        required_columns = {
            "ID",
            "LAST_UPDATED_DATE",
            "LAST_UPDATED_TIMESTAMP",
            "TITLE_RAW",
            "TITLE_CLEAN",
            "POSTED",
            "EXPIRED",
            "SALARY_FROM",
            "SALARY_TO",
            "MIN_YEARS_EXPERIENCE",
            "MAX_YEARS_EXPERIENCE",
            "SKILLS",
            "SPECIALIZED_SKILLS",
            "SOFTWARE_SKILLS",
            "EMPLOYMENT_TYPE",
            "COMPANY",
            "COMPANY_NAME",
            "COMPANY_RAW",
            "COMPANY_IS_STAFFING",
            "CITY",
            "STATE",
            "COUNTY",
            "LOCATION",
            "SOC_2",
            "SOC_2_NAME",
            "SOC_3",
            "SOC_3_NAME",
            "SOC_4",
            "SOC_4_NAME",
            "SOC_5",
            "SOC_5_NAME",
            "ONET_NAME",
            "NAICS_2022_2",
            "NAICS_2022_2_NAME",
            "NAICS_2022_3",
            "NAICS_2022_3_NAME",
            "NAICS_2022_4",
            "NAICS_2022_4_NAME",
            "NAICS_2022_5",
            "NAICS_2022_5_NAME",
            "NAICS_2022_6",
            "NAICS_2022_6_NAME",
        }

        missing_columns = sorted(required_columns - set(raw.columns))

        if missing_columns:
            raise RuntimeError(
                "Required source columns are missing: "
                + ", ".join(missing_columns)
            )

        raw_count = raw.count()

        # Keep the most recently updated record for each job ID.
        job_window = Window.partitionBy("ID").orderBy(
            F.col("LAST_UPDATED_TIMESTAMP").desc_nulls_last(),
            F.col("LAST_UPDATED_DATE").desc_nulls_last(),
        )

        base = (
            raw
            .filter(
                F.col("ID").isNotNull()
                & (F.length(F.trim(F.col("ID"))) > 0)
            )
            .withColumn(
                "_job_row",
                F.row_number().over(job_window),
            )
            .filter(F.col("_job_row") == 1)
            .drop("_job_row")
            .persist(StorageLevel.DISK_ONLY)
        )

        base_count = base.count()

        job_postings = base.select(
            "ID",
            "TITLE_RAW",
            "TITLE_CLEAN",
            "POSTED",
            "EXPIRED",
            "SALARY_FROM",
            "SALARY_TO",
            "MIN_YEARS_EXPERIENCE",
            "MAX_YEARS_EXPERIENCE",
            "SKILLS",
            "SPECIALIZED_SKILLS",
            "SOFTWARE_SKILLS",
            "EMPLOYMENT_TYPE",
            F.col("COMPANY").alias("COMPANY_ID"),
        )

        company_window = Window.partitionBy("COMPANY").orderBy(
            F.col("LAST_UPDATED_TIMESTAMP").desc_nulls_last(),
            F.col("LAST_UPDATED_DATE").desc_nulls_last(),
        )

        company = (
            base
            .filter(
                F.col("COMPANY").isNotNull()
                & (F.length(F.trim(F.col("COMPANY"))) > 0)
            )
            .withColumn(
                "_company_row",
                F.row_number().over(company_window),
            )
            .filter(F.col("_company_row") == 1)
            .select(
                F.col("COMPANY").alias("COMPANY_ID"),
                "COMPANY_NAME",
                "COMPANY_RAW",
                "COMPANY_IS_STAFFING",
            )
        )

        job_location = base.select(
            "ID",
            "CITY",
            "STATE",
            "COUNTY",
            "LOCATION",
        )

        soc_details = base.select(
            "ID",
            "SOC_2",
            "SOC_2_NAME",
            "SOC_3",
            "SOC_3_NAME",
            "SOC_4",
            "SOC_4_NAME",
            "SOC_5",
            "SOC_5_NAME",
        )

        lot_source_columns = {
            "LOT_CAREER_AREA",
            "LOT_CAREER_AREA_NAME",
            "LOT_OCCUPATION",
            "LOT_OCCUPATION_NAME",
            "LOT_SPECIALIZED_OCCUPATION",
        }

        if lot_source_columns.issubset(set(base.columns)):
            lot_details = base.select(
                "ID",
                "LOT_CAREER_AREA",
                "LOT_CAREER_AREA_NAME",
                "LOT_OCCUPATION",
                "LOT_OCCUPATION_NAME",
                "LOT_SPECIALIZED_OCCUPATION",
                "ONET_NAME",
            )
        else:
            print(
                "WARNING: LOT source fields are absent. "
                "Creating the required columns as blank values."
            )

            lot_details = base.select(
                "ID",
                F.lit(None).cast("string").alias("LOT_CAREER_AREA"),
                F.lit(None).cast("string").alias(
                    "LOT_CAREER_AREA_NAME"
                ),
                F.lit(None).cast("string").alias("LOT_OCCUPATION"),
                F.lit(None).cast("string").alias(
                    "LOT_OCCUPATION_NAME"
                ),
                F.lit(None).cast("string").alias(
                    "LOT_SPECIALIZED_OCCUPATION"
                ),
                "ONET_NAME",
            )

        naics_details = base.select(
            "ID",
            F.col("NAICS_2022_2").alias("NAICS2"),
            F.col("NAICS_2022_2_NAME").alias("NAICS2_NAME"),
            F.col("NAICS_2022_3").alias("NAICS3"),
            F.col("NAICS_2022_3_NAME").alias("NAICS3_NAME"),
            F.col("NAICS_2022_4").alias("NAICS4"),
            F.col("NAICS_2022_4_NAME").alias("NAICS4_NAME"),
            F.col("NAICS_2022_5").alias("NAICS5"),
            F.col("NAICS_2022_5_NAME").alias("NAICS5_NAME"),
            F.col("NAICS_2022_6").alias("NAICS6"),
            F.col("NAICS_2022_6_NAME").alias("NAICS6_NAME"),
        )

        tables = {
            "job_postings.csv": (job_postings, "ID"),
            "company.csv": (company, "COMPANY_ID"),
            "job_location.csv": (job_location, "ID"),
            "soc_details.csv": (soc_details, "ID"),
            "lot_details.csv": (lot_details, "ID"),
            "naics_details.csv": (naics_details, "ID"),
        }

        print(f"Raw rows: {raw_count:,}")
        print(f"Unique nonmissing job IDs: {base_count:,}")
        print(f"Rows removed: {raw_count - base_count:,}")

        for filename, (table, key) in tables.items():
            row_count = table.count()
            duplicate_count = duplicate_key_count(table, key)

            print(
                f"{filename}: rows={row_count:,}, "
                f"duplicate {key} values={duplicate_count:,}"
            )

            if duplicate_count != 0:
                raise RuntimeError(
                    f"{filename} contains duplicate {key} values"
                )

        company_orphans = (
            job_postings
            .filter(F.col("COMPANY_ID").isNotNull())
            .join(
                company.select("COMPANY_ID"),
                "COMPANY_ID",
                "left_anti",
            )
            .count()
        )

        print(f"Unmatched company IDs: {company_orphans:,}")

        if company_orphans != 0:
            raise RuntimeError("Unmatched company IDs were found")

        job_ids = job_postings.select("ID")

        for filename, table in [
            ("job_location.csv", job_location),
            ("soc_details.csv", soc_details),
            ("lot_details.csv", lot_details),
            ("naics_details.csv", naics_details),
        ]:
            orphan_count = (
                table
                .join(job_ids, "ID", "left_anti")
                .count()
            )

            print(f"{filename} unmatched job IDs: {orphan_count:,}")

            if orphan_count != 0:
                raise RuntimeError(
                    f"{filename} contains unmatched job IDs"
                )

        for filename, (table, _) in tables.items():
            write_single_csv(table, filename)

        print("All six relational CSV files were created successfully.")

        base.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

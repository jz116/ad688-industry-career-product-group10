from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/step3/lightcast_job_postings.csv")
PANEL_PATH = Path("data/processed/step3/career_market_panel.csv")
OUTPUT_PATH = Path(
    "data/processed/step3/career_market_descriptions.csv"
)


def main():
    panel = pd.read_csv(
        PANEL_PATH,
        dtype={"JOB_ID": "string"}
    )

    target_ids = set(panel["JOB_ID"].dropna())
    extracted_chunks = []

    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=["ID", "BODY"],
        dtype={"ID": "string", "BODY": "string"},
        encoding="utf-8-sig",
        chunksize=5000,
        low_memory=False
    ):
        selected = chunk[chunk["ID"].isin(target_ids)].copy()

        if not selected.empty:
            extracted_chunks.append(selected)

    if not extracted_chunks:
        raise ValueError("No target descriptions were found.")

    descriptions = pd.concat(
        extracted_chunks,
        ignore_index=True
    )

    descriptions = descriptions.rename(
        columns={"ID": "JOB_ID"}
    )

    descriptions = descriptions.drop_duplicates(
        subset=["JOB_ID"]
    )

    missing_ids = target_ids - set(descriptions["JOB_ID"])

    if missing_ids:
        raise ValueError(
            f"Missing descriptions for {len(missing_ids)} jobs: "
            f"{sorted(missing_ids)}"
        )

    corpus = panel[[
        "JOB_ID",
        "TITLE_RAW",
        "TITLE_STANDARDIZED",
        "ROLE_SEGMENT",
        "SCOPE_TIER",
        "COMPANY_NAME"
    ]].merge(
        descriptions[["JOB_ID", "BODY"]],
        on="JOB_ID",
        how="left",
        validate="one_to_one"
    )

    corpus["BODY_AVAILABLE"] = (
        corpus["BODY"]
        .fillna("")
        .str.strip()
        .ne("")
    )

    corpus["WORD_COUNT"] = (
        corpus["BODY"]
        .fillna("")
        .str.split()
        .str.len()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    corpus.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("NLP CORPUS VALIDATION")
    print(f"Target job IDs: {len(target_ids):,}")
    print(f"Descriptions extracted: {len(corpus):,}")
    print(
        "Nonempty descriptions: "
        f"{int(corpus['BODY_AVAILABLE'].sum()):,}"
    )
    print(
        "Median description words: "
        f"{corpus['WORD_COUNT'].median():,.0f}"
    )
    print()
    print("CREATED FILE")
    print(
        f"{OUTPUT_PATH}: "
        f"{OUTPUT_PATH.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()

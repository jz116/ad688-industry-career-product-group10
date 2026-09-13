#!/usr/bin/env python3
"""Build reproducible Step 2 summary tables and static website figures.

The script reads the cleaned career-market panel without changing it. Employer
labels are standardized only for presentation in the employer chart and table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


NAVY = "#173B57"
TEAL = "#168A8A"
BLUE = "#4C78A8"
AMBER = "#E29D34"
RED = "#C84C4C"
LIGHT_GRAY = "#E9EEF2"
MID_GRAY = "#667781"


EMPLOYER_DISPLAY_NAMES = {
    "Jpmc.fa.oraclecloud.com": "JPMorgan Chase",
    "Careers.wexinc.com": "WEX",
    "Careers.humana.com": "Humana",
    "Coreweave": "CoreWeave",
    "Nice": "NICE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Step 2 market-baseline charts and summary CSV files."
    )
    parser.add_argument(
        "--panel",
        default="data/processed/career_market_panel.csv",
        help="Path to the cleaned career-market panel.",
    )
    parser.add_argument(
        "--audit",
        default="data/processed/career_market_cleaning_audit.csv",
        help="Path to the cleaning-audit CSV.",
    )
    parser.add_argument(
        "--figures",
        default="figures/step2",
        help="Directory for PNG figures.",
    )
    parser.add_argument(
        "--outputs",
        default="outputs/step2",
        help="Directory for summary CSV files.",
    )
    return parser.parse_args()


def currency_tick(value: float, _position: int) -> str:
    return f"${value / 1000:,.0f}k"


def percent_tick(value: float, _position: int) -> str:
    return f"{value:.0f}%"


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8D0D6")
    ax.spines["bottom"].set_color("#C8D0D6")
    ax.tick_params(colors="#33434D")
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_horizontal_count_chart(
    summary: pd.DataFrame,
    label_column: str,
    count_column: str,
    total: int,
    title: str,
    subtitle: str,
    x_label: str,
    output_path: Path,
    note: str | None = None,
    color: str = TEAL,
) -> None:
    plot_data = summary.sort_values(count_column, ascending=True).reset_index(drop=True)
    height = max(4.2, 0.48 * len(plot_data) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, height))
    bars = ax.barh(
        plot_data[label_column],
        plot_data[count_column],
        color=color,
        edgecolor="none",
        height=0.68,
    )
    largest = max(float(plot_data[count_column].max()), 1.0)
    ax.set_xlim(0, largest * 1.22)
    ax.xaxis.grid(True, color=LIGHT_GRAY, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color=NAVY, pad=24)
    ax.text(
        0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MID_GRAY,
    )
    for bar, value in zip(bars, plot_data[count_column]):
        percentage = 100 * float(value) / total
        ax.text(
            bar.get_width() + largest * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)} ({percentage:.1f}%)",
            va="center",
            fontsize=9.5,
            color="#263640",
        )
    if note:
        fig.text(0.125, 0.015, note, fontsize=9, color=MID_GRAY, ha="left")
        fig.subplots_adjust(bottom=0.13)
    clean_axes(ax)
    save_figure(fig, output_path)


def count_summary(
    series: pd.Series,
    label_name: str,
    total: int,
    preferred_order: list[str] | None = None,
) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    if preferred_order is not None:
        counts = counts.reindex(preferred_order, fill_value=0)
    result = counts.rename_axis(label_name).reset_index(name="job_count")
    result["share_percent"] = (100 * result["job_count"] / total).round(1)
    return result


def nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel)
    audit_path = Path(args.audit)
    figures_dir = Path(args.figures)
    outputs_dir = Path(args.outputs)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Spark wrote JSON-like strings with backslash escapes, so escapechar is required.
    panel = pd.read_csv(panel_path, escapechar="\\")
    audit = pd.read_csv(audit_path)
    total_jobs = len(panel)

    if total_jobs == 0:
        raise RuntimeError("The career-market panel is empty.")
    if panel["JOB_ID"].nunique(dropna=True) != total_jobs:
        raise RuntimeError("The panel must contain one unique row per JOB_ID.")
    standardized_naics = panel["NAICS_2022_6"].astype(str).str.replace(".0", "", regex=False)
    if not standardized_naics.eq("513210").all():
        raise RuntimeError("At least one panel row is outside standardized NAICS 513210.")

    panel["EMPLOYER_DISPLAY"] = panel["COMPANY_NAME"].replace(EMPLOYER_DISPLAY_NAMES)
    panel["STATE_DISPLAY"] = panel["STATE"].fillna("Unspecified U.S. location")

    role_order = [
        "Data Scientist",
        "Machine Learning Technical Role",
        "AI Technical Role",
    ]
    role_summary = count_summary(panel["ROLE_SEGMENT"], "role_segment", total_jobs, role_order)
    write_summary(role_summary, outputs_dir / "role_segment_summary.csv")
    make_horizontal_count_chart(
        role_summary,
        "role_segment",
        "job_count",
        total_jobs,
        "Job volume by pathway segment",
        "Software Publishers (standardized NAICS 513210), 66 postings",
        "Distinct job postings",
        figures_dir / "job_volume_by_segment.png",
        note="Core Data Scientist roles are shown separately from adjacent AI/ML technical roles.",
        color=TEAL,
    )

    seniority_order = [
        "Internship",
        "No explicit seniority signal",
        "Senior title",
        "Staff/Principal/Lead title",
    ]
    seniority_summary = count_summary(
        panel["SENIORITY_PROXY"], "seniority_proxy", total_jobs, seniority_order
    )
    write_summary(seniority_summary, outputs_dir / "seniority_summary.csv")
    make_horizontal_count_chart(
        seniority_summary,
        "seniority_proxy",
        "job_count",
        total_jobs,
        "Experience signal in job titles",
        "Reported experience years were missing for every posting",
        "Distinct job postings",
        figures_dir / "seniority_distribution.png",
        note="Seniority is a title-based proxy; it is not an imputed years-of-experience requirement.",
        color=BLUE,
    )

    remote_order = ["Remote", "Unknown"]
    remote_summary = count_summary(
        panel["REMOTE_STATUS"], "remote_status", total_jobs, remote_order
    )
    write_summary(remote_summary, outputs_dir / "remote_summary.csv")
    make_horizontal_count_chart(
        remote_summary,
        "remote_status",
        "job_count",
        total_jobs,
        "Reported remote-work status",
        "Only explicit source indicators are classified as remote",
        "Distinct job postings",
        figures_dir / "remote_status.png",
        note="Unknown is retained as missing information and must not be interpreted as onsite.",
        color=AMBER,
    )

    education_order = ["Master's degree", "Bachelor's degree"]
    education_summary = count_summary(
        panel["EDUCATION_REQUIRED"], "education_requirement", total_jobs, education_order
    )
    write_summary(education_summary, outputs_dir / "education_summary.csv")
    make_horizontal_count_chart(
        education_summary,
        "education_requirement",
        "job_count",
        total_jobs,
        "Minimum education requirement",
        "Cleaned from the posting-level education fields",
        "Distinct job postings",
        figures_dir / "education_distribution.png",
        color=NAVY,
    )

    state_summary = count_summary(panel["STATE_DISPLAY"], "state", total_jobs)
    state_summary = state_summary.sort_values(
        ["job_count", "state"], ascending=[False, True]
    ).reset_index(drop=True)
    write_summary(state_summary, outputs_dir / "top_states.csv")
    make_horizontal_count_chart(
        state_summary,
        "state",
        "job_count",
        total_jobs,
        "Hiring locations by state",
        "All observed states plus postings without a state value",
        "Distinct job postings",
        figures_dir / "top_states.png",
        note="Four national or U.S.-level postings have no state and remain explicitly unspecified.",
        color=TEAL,
    )

    employer_counts = panel["EMPLOYER_DISPLAY"].value_counts().head(10)
    employer_summary = employer_counts.rename_axis("employer_display").reset_index(
        name="job_count"
    )
    employer_summary["share_percent"] = (
        100 * employer_summary["job_count"] / total_jobs
    ).round(1)
    write_summary(employer_summary, outputs_dir / "top_employers.csv")
    make_horizontal_count_chart(
        employer_summary,
        "employer_display",
        "job_count",
        total_jobs,
        "Top employers in the filtered market",
        "Ten employers with the most distinct postings",
        "Distinct job postings",
        figures_dir / "top_employers.png",
        note="Domain-style source labels were standardized for display; raw labels remain in the panel.",
        color=BLUE,
    )

    valid_salary = panel.loc[
        panel["SALARY_STATUS"].eq("Valid"), "SALARY_MIDPOINT_ANNUAL"
    ].dropna()
    if valid_salary.empty:
        raise RuntimeError("No valid midpoint salaries are available for the salary chart.")
    salary_summary = pd.DataFrame(
        {
            "metric": [
                "jobs_with_valid_salary",
                "coverage_percent",
                "mean_annual_midpoint",
                "minimum_annual_midpoint",
                "first_quartile_annual_midpoint",
                "median_annual_midpoint",
                "third_quartile_annual_midpoint",
                "maximum_annual_midpoint",
                "iqr_outliers_flagged_and_retained",
            ],
            "value": [
                len(valid_salary),
                round(100 * len(valid_salary) / total_jobs, 1),
                round(valid_salary.mean(), 2),
                round(valid_salary.min(), 2),
                round(valid_salary.quantile(0.25), 2),
                round(valid_salary.median(), 2),
                round(valid_salary.quantile(0.75), 2),
                round(valid_salary.max(), 2),
                int(panel["SALARY_OUTLIER_FLAG"].fillna(False).astype(bool).sum()),
            ],
        }
    )
    write_summary(salary_summary, outputs_dir / "salary_summary.csv")

    median_salary = float(valid_salary.median())
    flagged_salaries = panel.loc[
        panel["SALARY_STATUS"].eq("Valid")
        & panel["SALARY_OUTLIER_FLAG"].fillna(False).astype(bool),
        "SALARY_MIDPOINT_ANNUAL",
    ].dropna()
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bins = np.linspace(valid_salary.min(), valid_salary.max(), 12)
    ax.hist(valid_salary, bins=bins, color=TEAL, edgecolor="white", linewidth=1.2)
    ax.axvline(
        median_salary,
        color=NAVY,
        linewidth=2.2,
        linestyle="--",
        label=f"Median: ${median_salary:,.0f}",
    )
    if not flagged_salaries.empty:
        ax.scatter(
            flagged_salaries,
            np.full(len(flagged_salaries), 0.18),
            color=RED,
            marker="|",
            s=180,
            linewidths=2.2,
            label=f"IQR flags retained: {len(flagged_salaries)}",
            zorder=4,
        )
    ax.xaxis.set_major_formatter(FuncFormatter(currency_tick))
    ax.yaxis.grid(True, color=LIGHT_GRAY, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_xlabel("Annualized salary midpoint")
    ax.set_ylabel("Distinct job postings")
    ax.set_title(
        "Annualized salary distribution",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        pad=24,
    )
    ax.text(
        0,
        1.01,
        f"{len(valid_salary)} of {total_jobs} postings reported usable salary ranges",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MID_GRAY,
    )
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)
    fig.text(
        0.125,
        0.015,
        "Five IQR outliers were flagged but retained because the values are plausible for senior AI roles.",
        fontsize=9,
        color=MID_GRAY,
        ha="left",
    )
    fig.subplots_adjust(bottom=0.14)
    save_figure(fig, figures_dir / "salary_distribution.png")

    experience_present = panel[["MIN_YEARS_EXPERIENCE", "MAX_YEARS_EXPERIENCE"]].notna().any(axis=1)
    completeness_rows = [
        ("Posting date", nonblank(panel["POSTED_DATE"]).sum()),
        ("Employer", nonblank(panel["COMPANY_NAME"]).sum()),
        ("Education", nonblank(panel["EDUCATION_REQUIRED"]).sum()),
        ("State", nonblank(panel["STATE"]).sum()),
        ("Employment type", panel["EMPLOYMENT_TYPE"].ne("Unknown").sum()),
        ("Valid salary", panel["SALARY_STATUS"].eq("Valid").sum()),
        ("At least one skill", panel["SKILL_COUNT"].fillna(0).gt(0).sum()),
        ("Known remote status", panel["REMOTE_STATUS"].ne("Unknown").sum()),
        ("Reported experience years", experience_present.sum()),
    ]
    completeness = pd.DataFrame(completeness_rows, columns=["field", "job_count"])
    completeness["coverage_percent"] = (
        100 * completeness["job_count"] / total_jobs
    ).round(1)
    write_summary(completeness, outputs_dir / "data_completeness.csv")

    completeness_plot = completeness.sort_values("coverage_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    bars = ax.barh(
        completeness_plot["field"],
        completeness_plot["coverage_percent"],
        color=NAVY,
        edgecolor="none",
        height=0.68,
    )
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(FuncFormatter(percent_tick))
    ax.xaxis.grid(True, color=LIGHT_GRAY, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_xlabel("Share of 66 postings with usable information")
    ax.set_ylabel("")
    ax.set_title(
        "Coverage of core analytical fields",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        pad=24,
    )
    ax.text(
        0,
        1.01,
        "Availability varies sharply across the source fields",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MID_GRAY,
    )
    for bar, percentage, count in zip(
        bars, completeness_plot["coverage_percent"], completeness_plot["job_count"]
    ):
        ax.text(
            min(bar.get_width() + 1.4, 102),
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.1f}% ({int(count)})",
            va="center",
            fontsize=9.5,
            color="#263640",
        )
    clean_axes(ax)
    fig.text(
        0.125,
        0.015,
        "Unknown remote status and missing experience are preserved; neither is inferred from absence.",
        fontsize=9,
        color=MID_GRAY,
        ha="left",
    )
    fig.subplots_adjust(bottom=0.13)
    save_figure(fig, figures_dir / "data_completeness.png")

    advanced_titles = panel["SENIORITY_PROXY"].isin(
        ["Senior title", "Staff/Principal/Lead title"]
    ).sum()
    market_kpis = pd.DataFrame(
        {
            "metric": [
                "final_panel_jobs",
                "core_data_scientist_jobs",
                "adjacent_ai_ml_jobs",
                "advanced_title_jobs",
                "advanced_title_share_percent",
                "valid_salary_jobs",
                "median_annual_salary_midpoint",
                "explicit_remote_jobs",
                "unknown_remote_jobs",
                "earliest_posting_date",
                "latest_posting_date",
            ],
            "value": [
                total_jobs,
                int(panel["SCOPE_TIER"].eq("Core").sum()),
                int(panel["SCOPE_TIER"].eq("Adjacent").sum()),
                int(advanced_titles),
                round(100 * advanced_titles / total_jobs, 1),
                len(valid_salary),
                round(median_salary, 2),
                int(panel["REMOTE_STATUS"].eq("Remote").sum()),
                int(panel["REMOTE_STATUS"].eq("Unknown").sum()),
                panel["POSTED_DATE"].min(),
                panel["POSTED_DATE"].max(),
            ],
        }
    )
    write_summary(market_kpis, outputs_dir / "market_kpis.csv")

    audit_copy = audit.copy()
    write_summary(audit_copy, outputs_dir / "cleaning_audit_copy.csv")

    print(f"Validated panel rows: {total_jobs:,}")
    print(f"Figures written: {len(list(figures_dir.glob('*.png')))}")
    print(f"Summary tables written: {len(list(outputs_dir.glob('*.csv')))}")
    print(f"Figure directory: {figures_dir.resolve()}")
    print(f"Summary directory: {outputs_dir.resolve()}")


if __name__ == "__main__":
    main()

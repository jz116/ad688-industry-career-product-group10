#!/usr/bin/env python3
"""Create the Step 3 Plotly EDA figures and supporting summary tables.

Run this script from the repository root after ``prepare_step3_market_panel.py``.
The PNG files are intended for the Quarto website. The HTML files are optional
interactive previews and are not required for GitHub Pages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


SKILL_ALIASES = {
    "Docker (Software)": "Docker",
    "Generative Artificial Intelligence": "Generative AI",
    "Github": "GitHub",
    "Kafka": "Apache Kafka",
    "MLOps (Machine Learning Operations)": "MLOps",
    "Tableau (Business Intelligence Software)": "Tableau",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create consistently themed Plotly charts for Step 3."
    )
    parser.add_argument(
        "--panel",
        default="data/processed/step3/career_market_panel.csv",
        help="Clean Step 3 posting-level panel.",
    )
    parser.add_argument(
        "--skills",
        default="data/processed/step3/career_market_skills.csv",
        help="Clean Step 3 job-skill bridge table.",
    )
    parser.add_argument(
        "--figures",
        default="figures/step3",
        help="Directory for Plotly PNG and HTML figures.",
    )
    parser.add_argument(
        "--outputs",
        default="outputs/step3/chart_data",
        help="Directory for chart-ready summary CSV files.",
    )
    parser.add_argument(
        "--top-skills",
        type=int,
        default=15,
        help="Number of skills shown in the skill-demand figure.",
    )
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Skip static PNG export if Kaleido or Chrome is unavailable.",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip interactive HTML previews.",
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Write summary CSV files without importing Plotly.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"{label} is missing required columns: {joined}")


def nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def count_summary(series: pd.Series, label_column: str, total: int) -> pd.DataFrame:
    result = (
        series.fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis(label_column)
        .reset_index(name="job_count")
    )
    result["share_percent"] = (100 * result["job_count"] / total).round(1)
    return result


def build_summaries(
    panel: pd.DataFrame,
    skills: pd.DataFrame,
    top_skill_count: int,
) -> dict[str, pd.DataFrame]:
    total = len(panel)

    panel = panel.copy()
    panel["POSTED_DATE"] = pd.to_datetime(panel["POSTED_DATE"], errors="coerce")
    panel["SALARY_MIDPOINT_ANNUAL"] = pd.to_numeric(
        panel["SALARY_MIDPOINT_ANNUAL"], errors="coerce"
    )
    panel["MIN_YEARS_EXPERIENCE"] = pd.to_numeric(
        panel["MIN_YEARS_EXPERIENCE"], errors="coerce"
    )
    panel["SKILL_COUNT"] = pd.to_numeric(panel["SKILL_COUNT"], errors="coerce")

    role_summary = count_summary(panel["ROLE_SEGMENT"], "role_segment", total)
    role_summary = role_summary.sort_values(
        ["job_count", "role_segment"], ascending=[False, True]
    ).reset_index(drop=True)

    monthly_summary = (
        panel.dropna(subset=["POSTED_DATE"])
        .assign(posting_month=panel["POSTED_DATE"].dt.to_period("M").dt.to_timestamp())
        .groupby("posting_month", as_index=False)
        .agg(job_count=("JOB_ID", "nunique"))
        .sort_values("posting_month")
        .reset_index(drop=True)
    )
    monthly_summary["month_label"] = monthly_summary["posting_month"].dt.strftime(
        "%b %Y"
    )
    monthly_summary["share_percent"] = (
        100 * monthly_summary["job_count"] / total
    ).round(1)

    salary_summary = panel.loc[
        panel["SALARY_MIDPOINT_ANNUAL"].notna(),
        [
            "JOB_ID",
            "POSTED_DATE",
            "TITLE_STANDARDIZED",
            "ROLE_SEGMENT",
            "SCOPE_TIER",
            "COMPANY_NAME",
            "STATE",
            "SALARY_FROM_ANNUAL",
            "SALARY_TO_ANNUAL",
            "SALARY_MIDPOINT_ANNUAL",
        ],
    ].copy()
    salary_summary = salary_summary.sort_values(
        ["SALARY_MIDPOINT_ANNUAL", "COMPANY_NAME", "POSTED_DATE"]
    ).reset_index(drop=True)
    salary_summary["posting_label"] = (
        salary_summary["COMPANY_NAME"].fillna("Unknown employer")
        + " — "
        + salary_summary["TITLE_STANDARDIZED"].fillna("Unknown title")
        + " — "
        + salary_summary["POSTED_DATE"].dt.strftime("%b %d")
    )

    state_summary = count_summary(panel["STATE"], "state", total)
    state_summary = state_summary.sort_values(
        ["job_count", "state"], ascending=[False, True]
    ).reset_index(drop=True)

    employer_summary = count_summary(panel["COMPANY_NAME"], "employer", total)
    employer_summary = employer_summary.sort_values(
        ["job_count", "employer"], ascending=[False, True]
    ).reset_index(drop=True)

    remote_summary = count_summary(panel["REMOTE_STATUS"], "remote_status", total)
    remote_order = pd.Categorical(
        remote_summary["remote_status"], categories=["Remote", "Onsite", "Hybrid", "Unknown"]
    )
    remote_summary = (
        remote_summary.assign(remote_order=remote_order)
        .sort_values("remote_order")
        .drop(columns="remote_order")
        .reset_index(drop=True)
    )

    education_summary = count_summary(
        panel["EDUCATION_REQUIRED"], "education_requirement", total
    )
    education_summary = education_summary.sort_values(
        ["job_count", "education_requirement"], ascending=[False, True]
    ).reset_index(drop=True)

    clean_skills = skills.copy()
    clean_skills["skill_display"] = clean_skills["SKILL"].replace(SKILL_ALIASES)
    clean_skills = clean_skills.drop_duplicates(subset=["JOB_ID", "skill_display"])
    skill_demand = (
        clean_skills.groupby("skill_display", as_index=False)
        .agg(job_count=("JOB_ID", "nunique"))
        .sort_values(["job_count", "skill_display"], ascending=[False, True])
        .reset_index(drop=True)
    )
    skill_demand["share_percent"] = (100 * skill_demand["job_count"] / total).round(1)
    top_skills = skill_demand.head(top_skill_count).copy()

    completeness_checks = [
        ("Posting date", panel["POSTED_DATE"].notna()),
        ("Employer", nonblank(panel["COMPANY_NAME"])),
        ("State", nonblank(panel["STATE"])),
        ("Reported minimum experience", panel["MIN_YEARS_EXPERIENCE"].notna()),
        ("At least one skill", panel["SKILL_COUNT"].fillna(0).gt(0)),
        (
            "Known education requirement",
            nonblank(panel["EDUCATION_REQUIRED"])
            & panel["EDUCATION_REQUIRED"].ne("Unknown"),
        ),
        ("Usable annualized salary", panel["SALARY_MIDPOINT_ANNUAL"].notna()),
        (
            "Known remote status",
            nonblank(panel["REMOTE_STATUS"]) & panel["REMOTE_STATUS"].ne("Unknown"),
        ),
    ]
    completeness_rows: list[dict[str, Any]] = []
    for field_name, usable_mask in completeness_checks:
        usable_count = int(usable_mask.sum())
        completeness_rows.append(
            {
                "field": field_name,
                "usable_count": usable_count,
                "total_jobs": total,
                "coverage_percent": round(100 * usable_count / total, 1),
            }
        )
    completeness_summary = pd.DataFrame(completeness_rows).sort_values(
        ["coverage_percent", "field"], ascending=[False, True]
    )

    salary_values = salary_summary["SALARY_MIDPOINT_ANNUAL"]
    overview_rows = [
        {"metric": "Distinct audited postings", "value": total},
        {
            "metric": "Core Data Scientist postings",
            "value": int(panel["SCOPE_TIER"].eq("Core").sum()),
        },
        {
            "metric": "Adjacent pathway postings",
            "value": int(panel["SCOPE_TIER"].eq("Adjacent").sum()),
        },
        {"metric": "Postings with usable salary", "value": int(salary_values.count())},
        {
            "metric": "Median annualized salary midpoint",
            "value": float(salary_values.median()) if salary_values.notna().any() else None,
        },
        {
            "metric": "Explicitly remote postings",
            "value": int(panel["REMOTE_STATUS"].eq("Remote").sum()),
        },
        {
            "metric": "Postings with at least one skill",
            "value": int(panel["SKILL_COUNT"].fillna(0).gt(0).sum()),
        },
    ]

    return {
        "eda_overview": pd.DataFrame(overview_rows),
        "monthly_posting_summary": monthly_summary,
        "role_segment_summary": role_summary,
        "salary_posting_summary": salary_summary,
        "state_summary": state_summary,
        "employer_summary": employer_summary,
        "remote_summary": remote_summary,
        "education_summary": education_summary,
        "market_skill_demand": skill_demand,
        "top_skill_summary": top_skills,
        "data_completeness_summary": completeness_summary,
    }


def percentage_labels(frame: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for row in frame.itertuples(index=False):
        labels.append(f"{int(row.job_count)} ({float(row.share_percent):.0f}%)")
    return labels


def count_bar_figure(
    go: Any,
    frame: pd.DataFrame,
    *,
    category_column: str,
    color: str,
    title: str,
    subtitle: str,
    note: str,
    height: int,
) -> Any:
    plot_data = frame.sort_values(
        ["job_count", category_column], ascending=[True, False]
    ).reset_index(drop=True)
    fig = go.Figure(
        go.Bar(
            x=plot_data["job_count"],
            y=plot_data[category_column],
            orientation="h",
            marker={"color": color},
            text=percentage_labels(plot_data),
            textposition="outside",
            cliponaxis=False,
            customdata=plot_data[["share_percent"]],
            hovertemplate=(
                "%{y}<br>%{x} distinct postings"
                "<br>%{customdata[0]:.1f}% of sample<extra></extra>"
            ),
        )
    )
    return fig, title, subtitle, note, height


def build_figures(
    panel: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> list[tuple[str, Any, int, int]]:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plotly is not installed. Run: python -m pip install --upgrade plotly kaleido"
        ) from exc

    from plotly_theme import AMBER, BLUE, NAVY, RED, TEAL, apply_job_market_theme

    total = len(panel)
    dates = pd.to_datetime(panel["POSTED_DATE"], errors="coerce")
    date_min = dates.min()
    date_max = dates.max()
    period_text = f"{date_min:%b %d, %Y} to {date_max:%b %d, %Y}"
    sample_text = f"{total} audited pathway postings in Software Publishers"
    figures: list[tuple[str, Any, int, int]] = []

    monthly = summaries["monthly_posting_summary"]
    monthly_fig = go.Figure(
        go.Bar(
            x=monthly["month_label"],
            y=monthly["job_count"],
            marker={"color": TEAL},
            text=monthly["job_count"].astype(int),
            textposition="outside",
            cliponaxis=False,
            customdata=monthly[["share_percent"]],
            hovertemplate=(
                "%{x}<br>%{y} distinct postings"
                "<br>%{customdata[0]:.1f}% of sample<extra></extra>"
            ),
        )
    )
    apply_job_market_theme(
        monthly_fig,
        title="Monthly volume in the audited pathway sample",
        subtitle=f"{sample_text}, {period_text}",
        x_title="Posting month",
        y_title="Distinct job postings",
        note=(
            "The sample is too small for a market-wide time trend; monthly counts describe only "
            "the audited pathway sample."
        ),
        height=500,
        grid_axis="y",
    )
    monthly_fig.update_yaxes(dtick=1, rangemode="tozero")
    figures.append(("monthly_posting_volume", monthly_fig, 1200, 620))

    roles = summaries["role_segment_summary"].sort_values(
        ["job_count", "role_segment"], ascending=[True, False]
    )
    role_colors: list[str] = []
    for segment in roles["role_segment"]:
        role_colors.append(NAVY if segment == "Data Scientist" else TEAL)
    role_fig = go.Figure(
        go.Bar(
            x=roles["job_count"],
            y=roles["role_segment"],
            orientation="h",
            marker={"color": role_colors},
            text=percentage_labels(roles),
            textposition="outside",
            cliponaxis=False,
            customdata=roles[["share_percent"]],
            hovertemplate=(
                "%{y}<br>%{x} distinct postings"
                "<br>%{customdata[0]:.1f}% of sample<extra></extra>"
            ),
        )
    )
    apply_job_market_theme(
        role_fig,
        title="Role mix within the selected career pathway",
        subtitle=sample_text,
        x_title="Distinct job postings",
        note=(
            "The navy bar is the explicit core Data Scientist role. Teal bars are adjacent "
            "AI, ML, analytics, or applied-science roles."
        ),
        height=590,
    )
    role_fig.update_xaxes(dtick=1, rangemode="tozero")
    figures.append(("role_segment_distribution", role_fig, 1300, 720))

    salary = summaries["salary_posting_summary"]
    median_salary = salary["SALARY_MIDPOINT_ANNUAL"].median()
    salary_fig = go.Figure(
        go.Scatter(
            x=salary["SALARY_MIDPOINT_ANNUAL"],
            y=salary["posting_label"],
            mode="markers+text",
            marker={"color": TEAL, "size": 14, "line": {"color": "white", "width": 1}},
            text=salary["SALARY_MIDPOINT_ANNUAL"].map("${:,.0f}".format),
            textposition="middle right",
            customdata=salary[
                [
                    "ROLE_SEGMENT",
                    "STATE",
                    "SALARY_FROM_ANNUAL",
                    "SALARY_TO_ANNUAL",
                ]
            ],
            hovertemplate=(
                "%{y}<br>Midpoint: $%{x:,.0f}"
                "<br>Role segment: %{customdata[0]}"
                "<br>State: %{customdata[1]}"
                "<br>Range: $%{customdata[2]:,.0f}–$%{customdata[3]:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    salary_fig.add_vline(
        x=median_salary,
        line={"color": NAVY, "width": 3, "dash": "dash"},
        annotation_text=f"Median ${median_salary:,.0f}",
        annotation_position="top left",
        annotation_font={"color": NAVY, "size": 14},
    )
    apply_job_market_theme(
        salary_fig,
        title="Reported annualized salary midpoints",
        subtitle=f"{len(salary)} of {total} postings reported usable salary ranges",
        x_title="Annualized salary midpoint",
        note=(
            "All seven salary observations are adjacent roles; the sole core Data Scientist "
            "posting has no usable salary. Values are descriptive and are not imputed."
        ),
        height=650,
    )
    salary_fig.update_xaxes(tickprefix="$", tickformat=",.0f", rangemode="tozero")
    figures.append(("salary_midpoints", salary_fig, 1450, 780))

    state_fig, title, subtitle, note, height = count_bar_figure(
        go,
        summaries["state_summary"],
        category_column="state",
        color=TEAL,
        title="Hiring locations by state",
        subtitle=sample_text,
        note="Counts describe posting locations; remote status is reported separately.",
        height=520,
    )
    apply_job_market_theme(
        state_fig,
        title=title,
        subtitle=subtitle,
        x_title="Distinct job postings",
        note=note,
        height=height,
    )
    state_fig.update_xaxes(dtick=1, rangemode="tozero")
    figures.append(("hiring_by_state", state_fig, 1200, 650))

    employer_fig, title, subtitle, note, height = count_bar_figure(
        go,
        summaries["employer_summary"],
        category_column="employer",
        color=BLUE,
        title="Employers represented in the pathway sample",
        subtitle=sample_text,
        note="Each count represents a distinct job ID; repeated titles are retained when the job IDs differ.",
        height=570,
    )
    apply_job_market_theme(
        employer_fig,
        title=title,
        subtitle=subtitle,
        x_title="Distinct job postings",
        note=note,
        height=height,
    )
    employer_fig.update_xaxes(dtick=1, rangemode="tozero")
    figures.append(("top_employers", employer_fig, 1200, 700))

    remote_fig, title, subtitle, note, height = count_bar_figure(
        go,
        summaries["remote_summary"],
        category_column="remote_status",
        color=AMBER,
        title="Reported remote-work status",
        subtitle=f"Only explicit source indicators are classified as remote, n={total}",
        note="Unknown is retained as missing information and must not be interpreted as onsite.",
        height=430,
    )
    apply_job_market_theme(
        remote_fig,
        title=title,
        subtitle=subtitle,
        x_title="Distinct job postings",
        note=note,
        height=height,
    )
    remote_fig.update_xaxes(dtick=1, rangemode="tozero")
    figures.append(("remote_status", remote_fig, 1150, 560))

    skill_fig, title, subtitle, note, height = count_bar_figure(
        go,
        summaries["top_skill_summary"],
        category_column="skill_display",
        color=TEAL,
        title="Most frequently requested skills",
        subtitle=f"Distinct-posting prevalence in the {total}-posting audited sample",
        note=(
            "Aliases are consolidated before counting. One posting can request many skills, "
            "so percentages do not sum to 100%."
        ),
        height=720,
    )
    apply_job_market_theme(
        skill_fig,
        title=title,
        subtitle=subtitle,
        x_title="Distinct postings mentioning the skill",
        note=note,
        height=height,
    )
    skill_fig.update_xaxes(dtick=1, rangemode="tozero")
    figures.append(("top_skills", skill_fig, 1300, 880))

    completeness = summaries["data_completeness_summary"].sort_values(
        ["coverage_percent", "field"], ascending=[True, False]
    )
    completeness_colors: list[str] = []
    for coverage in completeness["coverage_percent"]:
        completeness_colors.append(NAVY if coverage >= 80 else RED)
    completeness_text: list[str] = []
    for row in completeness.itertuples(index=False):
        completeness_text.append(
            f"{float(row.coverage_percent):.0f}% ({int(row.usable_count)}/{int(row.total_jobs)})"
        )
    completeness_fig = go.Figure(
        go.Bar(
            x=completeness["coverage_percent"],
            y=completeness["field"],
            orientation="h",
            marker={"color": completeness_colors},
            text=completeness_text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{text}<extra></extra>",
        )
    )
    apply_job_market_theme(
        completeness_fig,
        title="Coverage of core analytical fields",
        subtitle=f"Usable information among {total} audited pathway postings",
        x_title="Share of postings with usable information",
        note="Unknown values remain missing; no salary, education, or remote values are inferred.",
        height=620,
    )
    completeness_fig.update_xaxes(range=[0, 112], ticksuffix="%", dtick=20)
    figures.append(("data_completeness", completeness_fig, 1300, 760))

    return figures


def write_summaries(summaries: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in summaries.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def export_figures(
    figures: list[tuple[str, Any, int, int]],
    figure_dir: Path,
    *,
    skip_png: bool,
    skip_html: bool,
) -> tuple[list[Path], str | None]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    png_error: str | None = None
    png_enabled = not skip_png

    for stem, fig, width, height in figures:
        if not skip_html:
            html_path = figure_dir / f"{stem}.html"
            fig.write_html(
                html_path,
                include_plotlyjs="cdn",
                full_html=True,
                config={"displaylogo": False, "responsive": True},
            )
            created.append(html_path)

        if png_enabled:
            png_path = figure_dir / f"{stem}.png"
            try:
                fig.write_image(png_path, width=width, height=height, scale=2)
                created.append(png_path)
            except Exception as exc:
                png_error = str(exc)
                png_enabled = False

    return created, png_error


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel)
    skills_path = Path(args.skills)
    figure_dir = Path(args.figures)
    output_dir = Path(args.outputs)

    panel = pd.read_csv(panel_path)
    skills = pd.read_csv(skills_path)

    require_columns(
        panel,
        [
            "JOB_ID",
            "POSTED_DATE",
            "TITLE_STANDARDIZED",
            "SCOPE_TIER",
            "ROLE_SEGMENT",
            "COMPANY_NAME",
            "STATE",
            "REMOTE_STATUS",
            "EDUCATION_REQUIRED",
            "MIN_YEARS_EXPERIENCE",
            "SALARY_MIDPOINT_ANNUAL",
            "SKILL_COUNT",
            "NAICS_2022_6",
        ],
        "Posting panel",
    )
    require_columns(skills, ["JOB_ID", "SKILL"], "Job-skill table")

    if panel.empty:
        raise RuntimeError("The Step 3 posting panel is empty.")
    if panel["JOB_ID"].nunique(dropna=True) != len(panel):
        raise RuntimeError("The panel must contain exactly one row per JOB_ID.")

    naics = (
        panel["NAICS_2022_6"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    if not naics.eq("513210").all():
        raise RuntimeError("At least one panel row is outside exact NAICS 513210.")

    unknown_skill_jobs = set(skills["JOB_ID"].dropna()) - set(panel["JOB_ID"].dropna())
    if unknown_skill_jobs:
        raise RuntimeError("The skill table contains JOB_ID values that are absent from the panel.")

    summaries = build_summaries(panel, skills, args.top_skills)
    write_summaries(summaries, output_dir)

    print("\nSTEP 3 CHART DATA")
    print(f"Distinct postings: {len(panel):,}")
    print(
        "Core / adjacent: "
        f"{int(panel['SCOPE_TIER'].eq('Core').sum())} / "
        f"{int(panel['SCOPE_TIER'].eq('Adjacent').sum())}"
    )
    print(
        "Usable salary rows: "
        f"{int(pd.to_numeric(panel['SALARY_MIDPOINT_ANNUAL'], errors='coerce').notna().sum())}"
    )
    print(f"Chart-data tables: {len(summaries)}")

    if args.summaries_only:
        print("Summaries-only mode: Plotly figures were not generated.")
        return

    figures = build_figures(panel, summaries)
    created, png_error = export_figures(
        figures,
        figure_dir,
        skip_png=args.skip_png,
        skip_html=args.skip_html,
    )

    print(f"Plotly figures prepared: {len(figures)}")
    print(f"Figure files created: {len(created)}")
    for path in created:
        print(path)

    if png_error:
        print("\nPNG EXPORT WARNING")
        print("Interactive HTML files were created, but PNG export stopped.")
        print("Install or update Kaleido and ensure Chrome or Chromium is available:")
        print("  python -m pip install --upgrade plotly kaleido")
        print("  plotly_get_chrome")
        print(f"Original export error: {png_error}")


if __name__ == "__main__":
    main()

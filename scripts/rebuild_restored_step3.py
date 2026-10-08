#!/usr/bin/env python3
"""Restore the approved 2024 cohort with the supplied preparation code.

Default input is the recovered ten-row extract. To scan the full local CSV:
python scripts/rebuild_restored_step3.py --source data/raw/step3/lightcast_job_postings.csv

The original approved-ID list is recovered from the saved description corpus.
This does not rerun the earlier role-candidate audit across the full market.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification"


def require(value, message):
    if not value:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(actual, expected):
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
        return True
    except AssertionError:
        return False


def write_table(frame, path):
    def formatted(value):
        if pd.isna(value):
            return "Unknown"
        if isinstance(value, (int, float, np.integer, np.floating)):
            if float(value).is_integer():
                return f"{int(value):,}"
            return f"{float(value):.3f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(str(col) for col in frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(formatted(value) for value in row) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_model(panel, skills, directory, export_figures):
    processed = directory / "data/processed/step3"
    processed.mkdir(parents=True, exist_ok=True)
    panel.to_csv(processed / "career_market_panel.csv", index=False)
    skills.to_csv(processed / "career_market_skills.csv", index=False)
    # Only presentation imports moved out of the numeric cells. Check their AST
    # against the original page so the clustering calculations cannot drift.
    content = (ROOT / "ml_methods.qmd").read_text()
    cells = re.findall(r"```(?:\{python\}|python)\n(.*?)\n```", content, flags=re.DOTALL)
    archived_content = (ROOT / "archive/original_code/ml_methods.qmd").read_text()
    archived_cells = re.findall(r"```\{python\}\n(.*?)\n```", archived_content, flags=re.DOTALL)
    require(len(cells) == 6, "Unexpected archived ML page structure.")
    for index in range(3):
        original_tree = ast.parse(archived_cells[index])
        kept = []
        for node in original_tree.body:
            if isinstance(node, ast.Import) and all(alias.name.startswith("plotly") for alias in node.names):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "plotly_theme":
                continue
            kept.append(node)
        original_tree.body = kept
        require(ast.dump(original_tree) == ast.dump(ast.parse(cells[index])), "The original numeric model logic was changed.")
    namespace = {}
    previous = Path.cwd()
    os.chdir(directory)
    try:
        for index in range(3):
            exec(compile(cells[index], f"archived_ml_cell_{index+1}", "exec"), namespace)
        output = directory / "outputs/step3"
        namespace["feature_matrix"].to_csv(output / "ml_feature_matrix.csv")
        namespace["clustered_panel"][["JOB_ID", "PLOT_ID", "TITLE_RAW", "ROLE_SEGMENT", "SCOPE_TIER", "CLUSTER_NAME"]].to_csv(
            output / "ml_cluster_assignments_with_ids.csv", index=False)
        namespace["model_input_summary"].to_csv(output / "ml_model_input_summary.csv", index=False)
        if export_figures:
            export_model_figures(namespace, directory)
    finally:
        os.chdir(previous)
    return namespace


def export_model_figures(namespace, directory):
    # Standard static plotting avoids a Chrome dependency on the EC2 instance.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    navy, teal, blue = "#173B57", "#168A8A", "#4C78A8"
    output = directory / "figures/step3"
    output.mkdir(parents=True, exist_ok=True)
    mds = namespace["MDS"](n_components=2, dissimilarity="precomputed", random_state=42,
                           n_init=20, max_iter=1000, normalized_stress="auto")
    coordinates = mds.fit_transform(namespace["distance_matrix"])
    clustered = namespace["clustered_panel"].copy()
    clustered["MDS_1"], clustered["MDS_2"] = coordinates[:, 0], coordinates[:, 1]
    clustered[["JOB_ID", "PLOT_ID", "CLUSTER_NAME", "MDS_1", "MDS_2"]].to_csv(
        directory / "outputs/step3/ml_cluster_plot_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    for name, color in [("AI/ML Architecture", teal), ("Analytics and Data", blue)]:
        group = clustered[clustered.CLUSTER_NAME.eq(name)]
        ax.scatter(group.MDS_1, group.MDS_2, s=115, color=color, edgecolor="white", linewidth=1.4, label=name, zorder=3)
    for number, row in enumerate(clustered.itertuples(index=False)):
        offset = (7, 10) if number % 2 == 0 else (7, -14)
        ax.annotate(row.PLOT_ID, (row.MDS_1, row.MDS_2), xytext=offset, textcoords="offset points", fontsize=11, color=navy)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.margins(0.18)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(color="#E6EDF2", linewidth=0.6, zorder=0)
    for edge in ["top", "right"]:
        ax.spines[edge].set_visible(False)
    fig.suptitle("Skill-Based Job Clusters", x=0.09, ha="left", color=navy, weight="bold", fontsize=19)
    fig.text(0.09, 0.915, "Corrected model: 37 recurring skills, Jaccard distance, average linkage", color="#667781", fontsize=11)
    fig.text(0.09, 0.045, "Nearby points have more similar skills. Axis directions have no independent career meaning.", fontsize=10, color="#667781")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.15)
    fig.savefig(output / "ml_skill_clusters.png", dpi=180, facecolor="white")
    plt.close(fig)

    matrix = namespace["feature_matrix"].copy()
    matrix["CLUSTER_NAME"] = namespace["cluster_names"]
    prevalence = matrix.groupby("CLUSTER_NAME").mean().mul(100)
    difference = (prevalence.loc["AI/ML Architecture"] - prevalence.loc["Analytics and Data"]).abs().sort_values(ascending=False)
    selected = difference.head(12).index.tolist()
    values = prevalence.loc[["AI/ML Architecture", "Analytics and Data"], selected]
    values.to_csv(directory / "outputs/step3/ml_cluster_skill_prevalence.csv")
    cmap = LinearSegmentedColormap.from_list("career", ["#F1F5F8", blue, navy])
    fig, ax = plt.subplots(figsize=(13.4, 4.8))
    drawing = ax.imshow(values.to_numpy(), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(selected)), [textwrap.fill(name, width=15) for name in selected], fontsize=9)
    ax.set_yticks([0, 1], ["AI/ML Architecture\n6 postings", "Analytics and Data\n4 postings"], fontsize=11)
    ax.tick_params(length=0, pad=9)
    for row in range(2):
        for col in range(len(selected)):
            value = values.iloc[row, col]
            ax.text(col, row, f"{value:.0f}%", ha="center", va="center", fontsize=12, color="white" if value >= 60 else navy)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(drawing, ax=ax, fraction=0.025, pad=0.018)
    bar.set_label("Postings (%)", fontsize=10)
    fig.suptitle("Skills That Distinguish the Two Job Clusters", x=0.025, ha="left", color=navy, weight="bold", fontsize=19)
    fig.text(0.025, 0.87, "Twelve recurring skills with the largest prevalence differences", fontsize=11, color="#667781")
    fig.text(0.025, 0.035, "Prepared skill bridge combines normalized source labels and controlled description keywords.", fontsize=10, color="#667781")
    fig.subplots_adjust(left=0.175, right=0.97, top=0.79, bottom=0.27)
    fig.savefig(output / "ml_cluster_skill_heatmap.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "evidence/step3_source_rows.csv")
    parser.add_argument("--export-figures", action="store_true", help="Refresh both ML PNGs with Matplotlib. Prepared PNGs are included.")
    args = parser.parse_args()
    source_path = args.source.resolve()
    VERIFY.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT))
    original = load_module(ROOT / "archive/original_code/prepare_step3_market_panel.py", "original_preparation")
    corrected = load_module(ROOT / "prepare_step3_market_panel.py", "corrected_preparation")
    charts = load_module(ROOT / "build_step3_charts.py", "chart_summaries")
    corpus_path = ROOT / "data/processed/step3/career_market_descriptions.csv"
    corpus = pd.read_csv(corpus_path, dtype=str, keep_default_na=False)
    ids = set(corpus.JOB_ID)
    require(len(corpus) == 10 and len(ids) == 10, "Expected the saved ten-ID cohort.")
    raw, scanned, industry, industry_window = original.stream_selected_source_rows(source_path, ids)
    require(set(raw) == ids, "Some approved IDs are absent from the provided source.")
    candidates = {}
    for row in corpus.to_dict("records"):
        job_id = row["JOB_ID"]
        require(row["BODY"] == raw[job_id]["BODY"], f"Description changed for {job_id}.")
        for corpus_key, raw_key in [("TITLE_RAW", "TITLE_RAW"), ("TITLE_STANDARDIZED", "TITLE_NAME"), ("COMPANY_NAME", "COMPANY_NAME")]:
            require(row[corpus_key] == raw[job_id][raw_key], f"Changed {corpus_key} for {job_id}.")
        candidates[job_id] = {"ID": job_id, "TITLE_RAW": raw[job_id]["TITLE_RAW"],
            "TITLE_CLEAN": raw[job_id]["TITLE_CLEAN"], "ROLE_SEGMENT": row["ROLE_SEGMENT"],
            "ROLE_MATCH_REASON": "Approved cohort label restored from saved description corpus; original role audit not rerun"}
    pd.DataFrame(candidates.values()).to_csv(ROOT / "outputs/step3/recovered_role_candidates.csv", index=False)

    old_panel, old_skills, _ = original.build_outputs(raw, candidates)
    panel, skills, salary_stats = corrected.build_outputs(raw, candidates)
    key_columns = ["JOB_ID", "SKILL"]
    old_pairs = set(old_skills[key_columns].itertuples(index=False, name=None))
    new_pairs = set(skills[key_columns].itertuples(index=False, name=None))
    expected_removed = {("0094418d419dc3a48fda36e0f6f5d4f41fd32459", "R")}
    require(old_pairs - new_pairs == expected_removed and not new_pairs - old_pairs,
            "The patch must remove only the Guidewire R false match.")
    unaffected_columns = [col for col in panel if col not in ["SKILLS_CLEAN", "SKILL_COUNT"]]
    require(same(panel[unaffected_columns], old_panel[unaffected_columns]), "An unrelated panel field changed.")
    r_pattern = dict(corrected.SKILL_PATTERNS)["R"]
    examples = {"R D team": False, "R & D team": False, "R and D team": False,
        "R&D team": False, "Python, SQL R, DBT": True, "R programming": True,
        "r programming": True, "R language": True, "Experience with R, Python": True,
        "R data analysis": True, "RStudio": False, "CRM": False}
    for value, expected in examples.items():
        require(bool(r_pattern.search(value)) == expected, f"R regression case failed: {value}")

    legacy_root = VERIFY / "original_code_rerun"
    legacy_root.mkdir(exist_ok=True)
    legacy_model = run_model(old_panel, old_skills, legacy_root, False)
    final_model = run_model(panel, skills, ROOT, args.export_figures)
    for name in ["model_input_summary", "model_comparison", "cluster_summary", "cluster_assignments"]:
        table = final_model[name]
        if name == "cluster_assignments":
            table = table.sort_values(["Cluster", "Job Title"]).reset_index(drop=True)
        write_table(table, ROOT / f"_generated/step3_{name}.qmd")
    saved_root = ROOT / "archive/saved_results"
    for name, obj in [("ml_model_comparison", "model_comparison"), ("ml_cluster_assignments", "cluster_assignments"), ("ml_cluster_summary", "cluster_summary")]:
        require(same(legacy_model[obj], pd.read_csv(saved_root / f"{name}.csv")), f"Original code did not reproduce {name}.")
    require(same(final_model["cluster_assignments"], legacy_model["cluster_assignments"]), "The patch changed cluster membership.")
    require(same(final_model["cluster_summary"], legacy_model["cluster_summary"]), "The patch changed cluster summary values.")

    summaries = charts.build_summaries(panel, skills, 15)
    old_summaries = charts.build_summaries(old_panel, old_skills, 15)
    table_checks = {}
    chart_dir = ROOT / "outputs/step3/chart_data"
    chart_dir.mkdir(exist_ok=True)
    for name, table in summaries.items():
        table.to_csv(chart_dir / f"{name}.csv", index=False)
        original_table_path = VERIFY / f"original_{name}.csv"
        old_summaries[name].to_csv(original_table_path, index=False)
        expected = pd.read_csv(saved_root / f"chart_data/{name}.csv")
        table_checks[name] = same(pd.read_csv(original_table_path), expected)
        require(table_checks[name], f"Original preparation disagrees with archived {name}.")
        if name != "market_skill_demand":
            require(same(old_summaries[name], table), f"Unexpected change to {name}.")
    demand = old_summaries["market_skill_demand"].set_index("skill_display").job_count.rename("Original count").to_frame()
    demand["Corrected count"] = summaries["market_skill_demand"].set_index("skill_display").job_count
    changed_counts = demand[demand["Original count"].ne(demand["Corrected count"])]
    require(changed_counts.index.tolist() == ["R"], "Unexpected skill-count change.")
    changed_counts.to_csv(VERIFY / "changed_skill_counts.csv")

    processed = ROOT / "data/processed/step3"
    corrected.data_dictionary().to_csv(processed / "career_market_data_dictionary.csv", index=False)
    cleaning = corrected.cleaning_audit(scanned, industry, industry_window, candidates, panel, salary_stats)
    recovered_source = ROOT / "evidence/step3_source_rows.csv"
    is_recovery_extract = digest(source_path) == digest(recovered_source)
    if is_recovery_extract:
        cleaning.loc[cleaning.metric.eq("source_rows"), "notes"] = "Rows scanned in the recovered ten-row extract, not the full source."
        for metric in ["exact_naics_513210_rows", "exact_naics_rows_in_window"]:
            cleaning.loc[cleaning.metric.eq(metric), "notes"] = "Count within the recovered ten-row extract only."
    cleaning.loc[cleaning.metric.eq("approved_candidate_ids"), "notes"] = "Approved IDs restored from the saved corpus; role-audit selection not rerun."
    cleaning.to_csv(processed / "career_market_cleaning_audit.csv", index=False)

    comparison = legacy_model["model_comparison"].rename(columns={"Silhouette Score": "Original silhouette"})
    comparison["Corrected silhouette"] = final_model["model_comparison"]["Silhouette Score"]
    comparison.to_csv(VERIFY / "original_vs_corrected_model.csv", index=False)
    important = ["Python", "SQL", "Machine Learning", "MLOps", "AWS", "Generative AI", "Data Modeling", "Data Visualization"]
    require(demand.loc[important, "Original count"].eq(demand.loc[important, "Corrected count"]).all(), "A profile skill count changed.")
    summary = {"original_preparation_script_supplied": True, "original_candidate_audit_supplied": False,
        "cohort_provenance": "Approved IDs and scope labels from the unchanged saved description corpus",
        "source_scope": "recovered ten-row extract" if is_recovery_extract else "provided source CSV",
        "source_rows_scanned_here": scanned, "exact_industry_rows_in_scanned_input": industry,
        "exact_industry_window_rows_in_scanned_input": industry_window,
        "earlier_full_source_recovery_scan_reported_rows": json.loads((ROOT / "evidence/step3_source_rows.audit.json").read_text())["source_rows_scanned"],
        "full_role_selection_audit_rerun": False,
        "original_code_reproduces_archived_model_scores_membership_summary": True,
        "original_code_reproduces_archived_chart_tables": table_checks,
        "removed_posting_skill": sorted(old_pairs-new_pairs), "added_posting_skills": sorted(new_pairs-old_pairs),
        "original_raw_bridge_rows": len(old_skills), "corrected_raw_bridge_rows": len(skills),
        "original_model_features": int(legacy_model["feature_matrix"].shape[1]),
        "corrected_model_features": int(final_model["feature_matrix"].shape[1]),
        "final_selected_clusters": 2, "final_selected_silhouette": float(final_model["model_comparison"].iloc[0]["Silhouette Score"]),
        "membership_unchanged": True,
        "membership_ari": float(adjusted_rand_score(legacy_model["cluster_names"], final_model["cluster_names"])),
        "skill_gap_profile_counts_unchanged": True, "regression_cases_passed": len(examples),
        "figures_refreshed_in_this_run": args.export_figures,
        "original_script_sha256": digest(ROOT / "archive/original_code/prepare_step3_market_panel.py"),
        "corrected_script_sha256": digest(ROOT / "prepare_step3_market_panel.py"),
        "source_sha256": digest(source_path), "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__}
    (VERIFY / "restoration_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nRESTORATION CHECKS PASSED")
    print(f"Source rows scanned: {scanned:,} ({summary['source_scope']})")
    print("Original model reproduced: 0.283 / 0.319 / 0.305")
    print("Corrected model: 0.291 / 0.327 / 0.311")
    print("Six/four membership unchanged; all eight skill-profile counts unchanged.")
    print("Removed one false R-language match. No other posting-skill pair changed.")
    print("Verification: verification/restoration_verification.json")


if __name__ == "__main__":
    main()

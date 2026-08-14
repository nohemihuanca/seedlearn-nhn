"""Aggregate benchmark grading results into CSVs and summary statistics."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from seedlearn.benchmarking.id_grader import IDGradeRecord
from seedlearn.benchmarking.trait_grader import TraitGradeRecord, TraitVerdict

logger = logging.getLogger(__name__)


def compute_binary_metrics(
    tp: int,
    fp: int,
    support: int,
    n_graded_specimens: int,
) -> dict[str, float | None]:
    """Compute precision, recall, F1, and prevalence for a binary trait column.

    Args:
        tp: True positives (pred=1, gt=1).
        fp: False positives (pred=1, gt=0).
        support: Number of graded specimens with gt=1 for this column.
        n_graded_specimens: Total number of graded specimens.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``, ``prevalence``.
        Values are None when the metric is undefined (e.g., zero denominator).
    """
    fn = support - tp

    precision: float | None = None
    if tp + fp > 0:
        precision = tp / (tp + fp)

    recall: float | None = None
    if support > 0:
        recall = tp / support

    f1: float | None = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    prevalence: float | None = None
    if n_graded_specimens > 0:
        prevalence = support / n_graded_specimens

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prevalence": prevalence,
    }


def build_trait_per_specimen_df(
    records: list[TraitGradeRecord],
) -> pd.DataFrame:
    """Convert trait grade records to a per-specimen DataFrame.

    Args:
        records: List of TraitGradeRecord from grading all specimens.

    Returns:
        DataFrame with columns: specimen_id, scientific_name, stri_column,
        category, ground_truth, predicted, verdict.
    """
    rows = [
        {
            "specimen_id": r.specimen_id,
            "scientific_name": r.scientific_name,
            "stri_column": r.stri_column,
            "category": r.category,
            "ground_truth": r.ground_truth,
            "predicted": r.predicted,
            "verdict": r.verdict.value,
            "vlm_raw_value": r.vlm_raw_value,
            "match_rule": r.match_rule,
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def build_trait_accuracy_per_trait(
    records: list[TraitGradeRecord],
    stri_support: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute per-trait accuracy from grading records.

    Only counts specimens with a definitive verdict (CORRECT or INCORRECT).
    Skipped specimens are tracked but not included in accuracy.

    Args:
        records: List of TraitGradeRecord.
        stri_support: Optional dict mapping STRI column name to count of
            graded specimens with gt=1 for that column. When provided,
            precision/recall/F1/prevalence columns are appended.

    Returns:
        DataFrame with columns: stri_column, category, n_correct, n_incorrect,
        n_skipped_uncoded, n_skipped_no_pred, n_graded, accuracy.
        When *stri_support* is provided, also includes: support, precision,
        recall, f1, prevalence.
    """
    df = build_trait_per_specimen_df(records)
    if df.empty:
        return pd.DataFrame()

    # Count total graded specimens (unique specimen_ids with a graded verdict)
    graded_mask = df["verdict"].isin(
        {TraitVerdict.CORRECT.value, TraitVerdict.INCORRECT.value}
    )
    n_graded_specimens = df.loc[graded_mask, "specimen_id"].nunique()

    summary_rows = []
    for (col, cat), group in df.groupby(["stri_column", "category"]):
        n_correct = (group["verdict"] == TraitVerdict.CORRECT.value).sum()
        n_incorrect = (group["verdict"] == TraitVerdict.INCORRECT.value).sum()
        n_uncoded = (group["verdict"] == TraitVerdict.SKIPPED_UNCODED.value).sum()
        n_no_pred = (group["verdict"] == TraitVerdict.SKIPPED_NO_PREDICTION.value).sum()
        n_not_obs = (group["verdict"] == TraitVerdict.SKIPPED_NOT_OBSERVED.value).sum()
        n_graded = n_correct + n_incorrect

        accuracy = n_correct / n_graded if n_graded > 0 else None

        row_dict: dict[str, Any] = {
            "stri_column": col,
            "category": cat,
            "n_correct": int(n_correct),
            "n_incorrect": int(n_incorrect),
            "n_skipped_uncoded": int(n_uncoded),
            "n_skipped_no_pred": int(n_no_pred),
            "n_not_observed": int(n_not_obs),
            "n_graded": int(n_graded),
            "accuracy": accuracy,
        }

        if stri_support is not None:
            support = stri_support.get(col, 0)
            metrics = compute_binary_metrics(
                tp=int(n_correct),
                fp=int(n_incorrect),
                support=support,
                n_graded_specimens=n_graded_specimens,
            )
            row_dict["support"] = support
            row_dict.update(metrics)

        summary_rows.append(row_dict)

    return pd.DataFrame(summary_rows).sort_values("stri_column")


def _vote_specimen_category(verdicts: set[str]) -> str:
    """Determine the single verdict for a specimen within a category.

    When a specimen has multiple trait rows in one category (e.g.,
    leaf_arrangement__alternate, leaf_arrangement__opposite), this picks the
    most informative verdict.

    Priority: CORRECT > INCORRECT > SKIPPED_NOT_OBSERVED > SKIPPED_NO_PREDICTION
    > SKIPPED_UNCODED.

    Args:
        verdicts: Set of TraitVerdict string values for one specimen+category.

    Returns:
        The winning TraitVerdict string value.
    """
    for v in (
        TraitVerdict.CORRECT,
        TraitVerdict.INCORRECT,
        TraitVerdict.SKIPPED_NOT_OBSERVED,
        TraitVerdict.SKIPPED_NO_PREDICTION,
        TraitVerdict.SKIPPED_UNCODED,
    ):
        if v.value in verdicts:
            return v.value
    return TraitVerdict.SKIPPED_UNCODED.value


def _count_category_verdicts(
    group: pd.DataFrame,
) -> dict[str, int | float | None]:
    """Count per-specimen category verdicts and compute accuracy.

    Args:
        group: DataFrame subset with columns ``specimen_id`` and ``verdict``
            for a single category (or category+species).

    Returns:
        Dict with n_correct, n_incorrect, n_skipped_uncoded, n_skipped_no_pred,
        n_not_observed, n_graded, accuracy.
    """
    n_correct = 0
    n_incorrect = 0
    n_uncoded = 0
    n_no_pred = 0
    n_not_obs = 0

    for _, spec_group in group.groupby("specimen_id"):
        vote = _vote_specimen_category(set(spec_group["verdict"].values))
        if vote == TraitVerdict.CORRECT.value:
            n_correct += 1
        elif vote == TraitVerdict.INCORRECT.value:
            n_incorrect += 1
        elif vote == TraitVerdict.SKIPPED_NOT_OBSERVED.value:
            n_not_obs += 1
        elif vote == TraitVerdict.SKIPPED_NO_PREDICTION.value:
            n_no_pred += 1
        else:
            n_uncoded += 1

    n_graded = n_correct + n_incorrect
    accuracy = n_correct / n_graded if n_graded > 0 else None

    return {
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_skipped_uncoded": n_uncoded,
        "n_skipped_no_pred": n_no_pred,
        "n_not_observed": n_not_obs,
        "n_graded": n_graded,
        "accuracy": accuracy,
    }


def build_trait_accuracy_per_category(
    records: list[TraitGradeRecord],
    stri_support: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute per-category accuracy from grading records.

    Rolls up per-trait accuracy to the category level (e.g., leaf_arrangement,
    leaf_type). A specimen contributes one vote per category: CORRECT if the
    predicted option matched any ground truth, INCORRECT otherwise.

    Args:
        records: List of TraitGradeRecord.
        stri_support: Optional dict mapping STRI column name to count of
            graded specimens with gt=1. When provided, ``macro_f1`` is added
            (mean of per-option F1 scores within the category).

    Returns:
        DataFrame with columns: category, n_correct, n_incorrect,
        n_skipped_uncoded, n_skipped_no_pred, n_graded, accuracy.
        When *stri_support* is provided, also includes ``macro_f1``.
    """
    df = build_trait_per_specimen_df(records)
    if df.empty:
        return pd.DataFrame()

    # Pre-compute per-trait F1 if stri_support provided
    per_trait_f1: dict[str, float | None] = {}
    if stri_support is not None:
        trait_df = build_trait_accuracy_per_trait(records, stri_support=stri_support)
        if not trait_df.empty and "f1" in trait_df.columns:
            for _, row in trait_df.iterrows():
                per_trait_f1[row["stri_column"]] = row["f1"]

    summary_rows = []
    for cat, cat_group in df.groupby("category"):
        counts = _count_category_verdicts(cat_group)
        counts["category"] = cat

        if stri_support is not None:
            # Compute macro F1: mean of per-option F1 scores in this category
            cat_f1s = [
                per_trait_f1[col]
                for col in per_trait_f1
                if col.startswith(f"{cat}__") and per_trait_f1[col] is not None
            ]
            counts["macro_f1"] = (
                sum(cat_f1s) / len(cat_f1s) if cat_f1s else None
            )

        summary_rows.append(counts)

    return pd.DataFrame(summary_rows).sort_values("category")


def build_trait_accuracy_per_species(
    records: list[TraitGradeRecord],
) -> pd.DataFrame:
    """Compute per-species, per-category trait accuracy.

    Groups by ``(scientific_name, category)``, votes each specimen within
    the group, then counts verdicts.

    Args:
        records: List of TraitGradeRecord.

    Returns:
        DataFrame with columns: scientific_name, category, n_correct,
        n_incorrect, n_skipped_uncoded, n_skipped_no_pred, n_not_observed,
        n_graded, accuracy.
    """
    df = build_trait_per_specimen_df(records)
    if df.empty:
        return pd.DataFrame()

    summary_rows = []
    for (species, cat), group in df.groupby(["scientific_name", "category"]):
        counts = _count_category_verdicts(group)
        counts["scientific_name"] = species
        counts["category"] = cat
        summary_rows.append(counts)

    return pd.DataFrame(summary_rows).sort_values(["scientific_name", "category"])


def build_within_species_agreement(
    records: list[TraitGradeRecord],
) -> pd.DataFrame:
    """Compute within-species VLM prediction agreement per category.

    For each ``(scientific_name, category)``, find each specimen's effective
    prediction (the ``stri_column`` where ``predicted == 1``), then compute
    the modal agreement rate.  This measures VLM consistency independent of
    ground truth.

    Args:
        records: List of TraitGradeRecord.

    Returns:
        DataFrame with columns: scientific_name, category, n_specimens,
        n_with_predictions, mode_prediction, n_agreeing, agreement.
    """
    df = build_trait_per_specimen_df(records)
    if df.empty:
        return pd.DataFrame()

    summary_rows = []
    for (species, cat), group in df.groupby(["scientific_name", "category"]):
        specimen_predictions: dict[str, str | None] = {}
        for specimen_id, spec_group in group.groupby("specimen_id"):
            # Find the stri_column where predicted == 1
            predicted_mask = spec_group["predicted"] == 1
            if predicted_mask.any():
                specimen_predictions[specimen_id] = spec_group.loc[
                    predicted_mask, "stri_column"
                ].iloc[0]
            else:
                specimen_predictions[specimen_id] = None

        n_specimens = len(specimen_predictions)
        preds_with_values = [v for v in specimen_predictions.values() if v is not None]
        n_with_predictions = len(preds_with_values)

        if n_with_predictions == 0:
            summary_rows.append(
                {
                    "scientific_name": species,
                    "category": cat,
                    "n_specimens": n_specimens,
                    "n_with_predictions": 0,
                    "mode_prediction": None,
                    "n_agreeing": 0,
                    "agreement": None,
                }
            )
            continue

        # Find mode prediction
        pred_counts = Counter(preds_with_values)
        mode_pred, mode_count = pred_counts.most_common(1)[0]

        summary_rows.append(
            {
                "scientific_name": species,
                "category": cat,
                "n_specimens": n_specimens,
                "n_with_predictions": n_with_predictions,
                "mode_prediction": mode_pred,
                "n_agreeing": mode_count,
                "agreement": mode_count / n_with_predictions,
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["scientific_name", "category"])


def _normalize_label(name: str | None) -> str:
    """Lowercase and strip a taxonomic name for metric computation."""
    if name is None:
        return ""
    return name.strip().lower()


def compute_multiclass_id_metrics(
    records: list[IDGradeRecord],
) -> dict[str, float | int]:
    """Compute accuracy, F1, precision, and recall for species ID at each rank.

    Args:
        records: List of IDGradeRecord (may include stage5_error records).

    Returns:
        Dict with keys: ``{rank}_accuracy``, ``{rank}_weighted_f1``,
        ``{rank}_macro_f1``, ``{rank}_weighted_precision``,
        ``{rank}_weighted_recall``, ``{rank}_macro_precision``,
        ``{rank}_macro_recall`` for each rank (family, genus, species),
        plus ``n_graded`` and ``n_stage5_error``.
    """
    graded = [r for r in records if not r.stage5_error]
    n_graded = len(graded)
    n_error = len(records) - n_graded

    result: dict[str, float | int] = {
        "n_graded": n_graded,
        "n_stage5_error": n_error,
    }

    if n_graded == 0:
        for rank in ("family", "genus", "species"):
            result[f"{rank}_accuracy"] = 0.0
            result[f"{rank}_weighted_f1"] = 0.0
            result[f"{rank}_macro_f1"] = 0.0
            result[f"{rank}_weighted_precision"] = 0.0
            result[f"{rank}_weighted_recall"] = 0.0
            result[f"{rank}_macro_precision"] = 0.0
            result[f"{rank}_macro_recall"] = 0.0
        return result

    for rank in ("family", "genus", "species"):
        true_labels = [_normalize_label(getattr(r, f"true_{rank}")) for r in graded]
        pred_labels = [_normalize_label(getattr(r, f"pred_{rank}")) for r in graded]

        correct_attr = f"{rank}_correct"
        accuracy = sum(getattr(r, correct_attr) for r in graded) / n_graded
        result[f"{rank}_accuracy"] = accuracy

        # Weighted averages
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted", zero_division=0,
        )
        result[f"{rank}_weighted_f1"] = float(f1_w)
        result[f"{rank}_weighted_precision"] = float(p_w)
        result[f"{rank}_weighted_recall"] = float(r_w)

        # Macro averages
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0,
        )
        result[f"{rank}_macro_f1"] = float(f1_m)
        result[f"{rank}_macro_precision"] = float(p_m)
        result[f"{rank}_macro_recall"] = float(r_m)

    return result


def build_id_summary_by_partition(
    records: list[IDGradeRecord],
) -> dict[str, dict[str, float | int]]:
    """Compute per-partition and overall multiclass ID metrics.

    Args:
        records: List of IDGradeRecord with optional partition field.

    Returns:
        Dict keyed by partition name (plus "all") mapping to metric dicts.
        Partitions with zero records are omitted.
    """
    result: dict[str, dict[str, float | int]] = {}

    # Overall
    result["all"] = compute_multiclass_id_metrics(records)

    # Group by partition
    groups: dict[str, list[IDGradeRecord]] = defaultdict(list)
    for r in records:
        if r.partition is not None:
            groups[r.partition].append(r)

    for partition_name in sorted(groups):
        result[partition_name] = compute_multiclass_id_metrics(groups[partition_name])

    return result


def build_id_per_specimen_df(records: list[IDGradeRecord]) -> pd.DataFrame:
    """Convert ID grade records to a per-specimen DataFrame.

    Args:
        records: List of IDGradeRecord from grading all specimens.

    Returns:
        DataFrame with all IDGradeRecord fields as columns.
    """
    rows = [
        {
            "specimen_id": r.specimen_id,
            "true_family": r.true_family,
            "true_genus": r.true_genus,
            "true_species": r.true_species,
            "pred_family": r.pred_family,
            "pred_genus": r.pred_genus,
            "pred_species": r.pred_species,
            "family_correct": r.family_correct,
            "genus_correct": r.genus_correct,
            "species_correct": r.species_correct,
            "confidence": r.confidence,
            "stage5_error": r.stage5_error,
            "partition": r.partition,
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def build_id_summary(records: list[IDGradeRecord]) -> dict[str, float | int]:
    """Compute overall species ID accuracy and F1 metrics from grading records.

    Args:
        records: List of IDGradeRecord.

    Returns:
        Dict with family_accuracy, genus_accuracy, species_accuracy,
        F1/precision/recall metrics per rank, n_graded, n_stage5_error.
    """
    return compute_multiclass_id_metrics(records)


def save_benchmark_results(
    output_dir: Path,
    trait_records: list[TraitGradeRecord],
    id_records: list[IDGradeRecord],
    source_label: str = "",
    stri_support: dict[str, int] | None = None,
) -> None:
    """Write all grading results to CSV and JSON files.

    Args:
        output_dir: Directory to write output files.
        trait_records: All trait grading records.
        id_records: All species ID grading records.
        source_label: Label for the ground truth source (e.g., "cl185", "merged").
            Used as prefix for output filenames.
        stri_support: Optional dict mapping STRI column to count of graded
            specimens with gt=1. Enables F1/prevalence columns in outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{source_label}_" if source_label else ""

    # Trait grading outputs
    if trait_records:
        trait_per_specimen = build_trait_per_specimen_df(trait_records)
        trait_per_specimen.to_csv(
            output_dir / f"{prefix}trait_grades_per_specimen.csv", index=False
        )

        trait_per_trait = build_trait_accuracy_per_trait(
            trait_records, stri_support=stri_support
        )
        trait_per_trait.to_csv(
            output_dir / f"{prefix}trait_accuracy_per_trait.csv", index=False
        )

        trait_per_cat = build_trait_accuracy_per_category(
            trait_records, stri_support=stri_support
        )
        trait_per_cat.to_csv(
            output_dir / f"{prefix}trait_accuracy_per_category.csv", index=False
        )

        trait_per_species = build_trait_accuracy_per_species(trait_records)
        trait_per_species.to_csv(
            output_dir / f"{prefix}trait_accuracy_per_species.csv", index=False
        )

        agreement = build_within_species_agreement(trait_records)
        agreement.to_csv(
            output_dir / f"{prefix}within_species_agreement.csv", index=False
        )

        logger.info(
            "Wrote trait grading files to %s (%d records)", output_dir, len(trait_records)
        )

    # ID grading outputs
    if id_records:
        id_per_specimen = build_id_per_specimen_df(id_records)
        id_per_specimen.to_csv(
            output_dir / f"{prefix}id_grades_per_specimen.csv", index=False
        )

        id_summary = build_id_summary(id_records)
        with open(output_dir / f"{prefix}id_summary.json", "w") as f:
            json.dump(id_summary, f, indent=2)

        # Save per-partition summary when any record has a partition
        if any(r.partition is not None for r in id_records):
            partition_summary = build_id_summary_by_partition(id_records)
            with open(
                output_dir / f"{prefix}id_summary_by_partition.json", "w"
            ) as f:
                json.dump(partition_summary, f, indent=2)

        logger.info(
            "Wrote ID grading files to %s (%d records)", output_dir, len(id_records)
        )


def _print_pivoted_table(
    df: pd.DataFrame,
    value_col: str,
    title: str,
) -> None:
    """Print a species-by-category pivoted table to stdout.

    Args:
        df: DataFrame with ``scientific_name``, ``category``, and *value_col*.
        value_col: Column containing float values (0-1) or None.
        title: Section title to print.
    """
    categories = sorted(df["category"].unique())
    species_list = sorted(df["scientific_name"].unique())

    # Build a lookup dict for fast access
    lookup: dict[tuple[str, str], float | None] = {}
    for _, row in df.iterrows():
        lookup[(row["scientific_name"], row["category"])] = row[value_col]

    # Abbreviate category names for column headers
    abbrev = {c: c.replace("leaf_", "").replace("arrangement", "leaf_arr").replace("margin", "leaf_mar").replace("type", "leaf_typ") for c in categories}

    name_width = max(len(s) for s in species_list) if species_list else 20
    name_width = max(name_width, 10)
    col_width = 10

    print(f"\n  {title}:")
    header = f"  {'Species':<{name_width}}"
    for cat in categories:
        header += f"  {abbrev[cat]:>{col_width}}"
    print(header)
    print(f"  {'-' * name_width}" + f"  {'-' * col_width}" * len(categories))

    for sp in species_list:
        line = f"  {sp:<{name_width}}"
        for cat in categories:
            val = lookup.get((sp, cat))
            if val is not None:
                line += f"  {val:>{col_width}.1%}"
            else:
                line += f"  {'N/A':>{col_width}}"
        print(line)


def print_summary(
    trait_records: list[TraitGradeRecord],
    id_records: list[IDGradeRecord],
    source_label: str = "",
    stri_support: dict[str, int] | None = None,
) -> None:
    """Print a human-readable summary to stdout.

    Args:
        trait_records: All trait grading records.
        id_records: All species ID grading records.
        source_label: Label for ground truth source.
        stri_support: Optional dict mapping STRI column to count of graded
            specimens with gt=1. Enables F1/prevalence in the summary.
    """
    header = f"Benchmark Results ({source_label})" if source_label else "Benchmark Results"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")

    # Trait accuracy per category
    if trait_records:
        cat_df = build_trait_accuracy_per_category(
            trait_records, stri_support=stri_support
        )
        has_f1 = "macro_f1" in cat_df.columns
        if has_f1:
            # Pre-compute per-option GT distribution from stri_support
            from seedlearn.benchmarking.trait_mapping import TRAIT_RULES as _TR

            # Count total graded specimens for percentage calculation
            graded_mask = build_trait_per_specimen_df(trait_records)["verdict"].isin(
                {TraitVerdict.CORRECT.value, TraitVerdict.INCORRECT.value}
            )
            _n_graded_total = build_trait_per_specimen_df(trait_records).loc[
                graded_mask, "specimen_id"
            ].nunique()

            print("\n  Trait Accuracy by Category:")
            print(
                f"  {'Category':<20} {'Acc':>6} {'F1':>6}"
                f"  {'GT Distribution':<45} {'Graded':>7} {'Skip':>6} {'NotObs':>6}"
            )
            print(f"  {'-' * 100}")
            for _, row in cat_df.iterrows():
                acc_str = f"{row['accuracy']:.1%}" if row["accuracy"] is not None else "N/A"
                f1_str = f"{row['macro_f1']:.3f}" if row.get("macro_f1") is not None else "N/A"
                # Per-option GT distribution
                cat_name = row["category"]
                gt_parts: list[str] = []
                for rule in _TR:
                    if rule.category == cat_name and stri_support is not None:
                        option_label = rule.stri_column.replace(f"{cat_name}__", "")
                        sup = stri_support.get(rule.stri_column, 0)
                        if _n_graded_total > 0:
                            pct = sup / _n_graded_total
                            gt_parts.append(f"{option_label}: {pct:.0%}")
                        else:
                            gt_parts.append(f"{option_label}: N/A")
                gt_dist = ", ".join(gt_parts) if gt_parts else "N/A"
                n_skip = row["n_skipped_uncoded"] + row["n_skipped_no_pred"]
                n_not_obs = row.get("n_not_observed", 0)
                print(
                    f"  {cat_name:<20} {acc_str:>6} {f1_str:>6}"
                    f"  {gt_dist:<45} {row['n_graded']:>7} {n_skip:>6} {n_not_obs:>6}"
                )
        else:
            print("\n  Trait Accuracy by Category:")
            print(f"  {'Category':<25} {'Accuracy':>10} {'N Graded':>10} {'N Skip':>8} {'N NotObs':>10}")
            print(f"  {'-' * 65}")
            for _, row in cat_df.iterrows():
                acc_str = f"{row['accuracy']:.1%}" if row["accuracy"] is not None else "N/A"
                n_skip = row["n_skipped_uncoded"] + row["n_skipped_no_pred"]
                n_not_obs = row.get("n_not_observed", 0)
                print(
                    f"  {row['category']:<25} {acc_str:>10} {row['n_graded']:>10} {n_skip:>8} {n_not_obs:>10}"
                )

        # Overall trait accuracy
        n_correct = sum(1 for r in trait_records if r.verdict == TraitVerdict.CORRECT)
        n_graded = sum(
            1 for r in trait_records
            if r.verdict in {TraitVerdict.CORRECT, TraitVerdict.INCORRECT}
        )
        if n_graded > 0:
            print(f"\n  Overall trait accuracy: {n_correct / n_graded:.1%} ({n_correct}/{n_graded})")

        # Per-species trait accuracy (pivoted)
        species_df = build_trait_accuracy_per_species(trait_records)
        if not species_df.empty:
            _print_pivoted_table(
                species_df,
                value_col="accuracy",
                title="Trait Accuracy by Species",
            )

        # Within-species agreement (pivoted)
        agree_df = build_within_species_agreement(trait_records)
        if not agree_df.empty:
            _print_pivoted_table(
                agree_df,
                value_col="agreement",
                title="Within-Species Agreement (VLM consistency)",
            )

    # ID accuracy
    if id_records:
        has_partitions = any(r.partition is not None for r in id_records)

        if has_partitions:
            partition_data = build_id_summary_by_partition(id_records)
            print("\n  Species ID Accuracy:")
            header = (
                f"    {'':>12} {'Family':>8} {'Genus':>8} {'Species':>8}"
                f" {'wF1(F)':>8} {'wF1(G)':>8} {'wF1(S)':>8}"
                f" {'mF1(S)':>8} {'N':>6}"
            )
            print(header)
            print(f"    {'-' * (len(header) - 4)}")
            # Print "all" first, then sorted partitions
            order = ["all"] + [k for k in partition_data if k != "all"]
            for key in order:
                m = partition_data[key]
                print(
                    f"    {key:>12}"
                    f" {m['family_accuracy']:>7.1%}"
                    f" {m['genus_accuracy']:>7.1%}"
                    f" {m['species_accuracy']:>7.1%}"
                    f" {m['family_weighted_f1']:>8.2f}"
                    f" {m['genus_weighted_f1']:>8.2f}"
                    f" {m['species_weighted_f1']:>8.2f}"
                    f" {m['species_macro_f1']:>8.2f}"
                    f" {m['n_graded']:>6}"
                )
            n_errors = partition_data["all"]["n_stage5_error"]
            if n_errors:
                print(f"    errors: {n_errors}")
        else:
            id_summary = build_id_summary(id_records)
            print("\n  Species ID Accuracy:")
            print(f"    Family:  {id_summary['family_accuracy']:.1%}")
            print(f"    Genus:   {id_summary['genus_accuracy']:.1%}")
            print(f"    Species: {id_summary['species_accuracy']:.1%}")
            print(
                f"    wF1: F={id_summary['family_weighted_f1']:.2f}"
                f"  G={id_summary['genus_weighted_f1']:.2f}"
                f"  S={id_summary['species_weighted_f1']:.2f}"
                f"  | mF1(S)={id_summary['species_macro_f1']:.2f}"
            )
            print(f"    N graded: {id_summary['n_graded']}, N errors: {id_summary['n_stage5_error']}")

    print(f"\n{'=' * 60}\n")

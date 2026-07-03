## This file contains the evaluation metrics

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def compute_method_f1(true_methods, pred_methods, average='macro'):
    """
    Computes the F1 score for method selection

    Args:
        true_methods: (List[str]) List of reference methods
        pred_methods: (List[str]) List of predicted methods
        average: (str) Averaging method for F1 score

    Returns:
        F1 score for method selection
    """

    # Only include entries where a prediction was made
    filtered_true = []
    filtered_pred = []

    for true_method, pred_method in zip(true_methods, pred_methods):
        if pd.notna(pred_method):
            filtered_true.append(true_method)
            filtered_pred.append(pred_method)

    f1 = f1_score(filtered_true, filtered_pred, average=average) * 100

    return round(f1, 2)


def compute_method_accuracy(true_methods, pred_methods):
    """
    Compute the method selection accuracy

    Args:
        true_methods: (List[str]) list of the reference methods
        pred_methods: (List[str]) list of the predicted methods

    Returns:
        (float) method selection accuracy
    """

    correct = 0
    total = 0
    for true_method, pred_method in zip(true_methods, pred_methods):
        # Only count entries where a prediction was made
        if pd.notna(pred_method):
            total += 1
            if true_method == pred_method:
                correct += 1
    acc = correct / total * 100

    return round(acc, 2)


def compute_relative_error(true_effects, pred_effects, metric="mean"):
    """
    Computes the relative error between true and predicted effects

    Args:
        true_effects: (List[float]) List of true causal effects
        pred_effects: (List[float]) List of predicted effects
        metric: (str) metric for computing the relative error

    Returns:
        (float) Relative error
    """

    relative_errors = []
    for true_val, pred_val in zip(true_effects, pred_effects):
        if pd.isna(true_val) or pd.isna(pred_val):
            relative_errors.append(np.nan)

        else:
            rel_error = abs(pred_val - true_val) / abs(true_val) * 100
            rel_error = min(rel_error, 100)  # Cap at 100%
            relative_errors.append(rel_error)

    if metric == "median":
        return round(np.nanmedian(relative_errors), 2)
    else:
        return round(np.nanmean(relative_errors), 2)


def compute_effect_accuracy(true_effects, pred_effects, thresh=5.0):
    """
    Computes the effect accuracy. The answer is deemed correct if predicted value
    is within (thresh) percentage of the true value

    Args:
        true_effects: (List[float]) List of true causal effects
        pred_effects: (List[float]) List of predicted effects
        thresh: (float) Threshold for considering the prediction accurate

    Returns:
        (float) Effect accuracy
    """

    effect_accuracy = []
    for true_eff, pred_eff in zip(true_effects, pred_effects):
        if pd.isna(true_eff) or pd.isna(pred_eff):
            effect_accuracy.append(np.nan)
        else:
            rel_error = abs(true_eff - pred_eff) / abs(true_eff) * 100
            if rel_error <= thresh:
                effect_accuracy.append(1)
            else:
                effect_accuracy.append(0)

    return round(np.nanmean(effect_accuracy) * 100, 2)


def is_nan(value):
    """
    Checks if a value is NaN or None

    Args:
        value: value to check

    Returns:
        (bool) True if value is NaN or None
    """

    return value is None or (isinstance(value, float) and np.isnan(value))


def evaluate_treatment(ref_treat, ref_multirct, pred_treat, methods, exclude_methods=None):
    """
    Computes treatment variable selection accuracy

    Args:
        ref_treat: (List[str]) list of reference treatment variable names
        ref_multirct: (List[str]) list of multi-RCT treatment variable names (or NaN)
        pred_treat: (List[str]) list of predicted treatment variable names
        methods: (List[str]) list of causal methods used
        exclude_methods: (List[str]) methods to exclude from evaluation

    Returns:
        (float) treatment variable selection accuracy
    """

    if exclude_methods is None:
        exclude_methods = []

    treatment_matches = []
    for ref, mult, pred, method in zip(ref_treat, ref_multirct, pred_treat, methods):
        if method in exclude_methods:
            continue

        # Normalize NaN values
        ref = None if is_nan(ref) else ref
        mult = None if is_nan(mult) else mult
        pred = None if is_nan(pred) else pred

        if ref is None or pred is None:
            treatment_matches.append(np.nan)
            continue

        ## the LLM 
        # Multi-RCT case: check against constructed treatment variable
        if isinstance(mult, str):
            match = mult == pred or mult in pred or pred in mult
        # Binary case: check against original treatment variable
        else:
            match = ref == pred or ref in pred or pred in ref

        treatment_matches.append(1 if match else 0)

    return round(np.nanmean(treatment_matches) * 100, 2) if treatment_matches else np.nan


def evaluate_outcome(ref_outcome, pred_outcome):
    """
    Computes outcome variable selection accuracy

    Args:
        ref_outcome: (List[str]) list of reference outcome variable names
        pred_outcome: (List[str]) list of predicted outcome variable names

    Returns:
        (float) outcome variable selection accuracy
    """

    outcome_matches = []
    for ref, pred in zip(ref_outcome, pred_outcome):
        # Normalize NaN values
        ref = None if is_nan(ref) else ref
        pred = None if is_nan(pred) else pred

        if ref is None or pred is None:
            outcome_matches.append(np.nan)
            continue
        try:
            match = ref == pred or ref in pred or pred in ref
            outcome_matches.append(1 if match else 0)
        except Exception:
            outcome_matches.append(np.nan)

    return round(np.nanmean(outcome_matches) * 100, 2) if outcome_matches else np.nan


def completion_rate(pred_effect, num_tries):
    """
    Computes the completion rate i.e. how often the pipeline produces a valid causal effect,
    and the mean number of attempts among successful runs

    Args:
        pred_effect: (pd.Series or List[float]) predicted causal effects
        num_tries: (pd.Series or List[float]) number of retries per query

    Returns:
        (tuple) completion rate as a percentage, mean number of attempts
    """

    total_success = 0
    attempt_counts = []
    total = len(pred_effect)

    for e, n in zip(pred_effect, num_tries):
        try:
            if not np.isnan(e) and e is not None:
                total_success += 1
                attempt_counts.append(n)
        except Exception:
            continue

    rate = round(total_success / total * 100, 2) if total > 0 else np.nan
    mean_attempts = round(np.mean(attempt_counts), 2) if attempt_counts else np.nan

    return rate, mean_attempts


def evaluate_covariates(ref_cov, pred_cov, causal_effect):
    """
    Computes control variable overlap using Jaccard-style similarity.
    Only includes rows where the causal effect was successfully estimated.

    Args:
        ref_cov: (List[str]) list of reference covariate strings (comma-separated)
        pred_cov: (List[str | List]) list of predicted covariate strings or lists
        causal_effect: (List[float]) list of predicted causal effects (used to filter failed runs)

    Returns:
        (float) mean overlap score as a percentage
    """

    def parse_covariate_set(value):
        """Parse covariate string into a set of lowercase variable names"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return set()
        if isinstance(value, float):
            return set()
        value_str = str(value).strip()
        if value_str == "":
            return set()
        return set(v.strip().lower() for v in value_str.split(",") if v.strip())

    overlap_scores = []

    for ref, pred, effect in zip(ref_cov, pred_cov, causal_effect):
        # Skip rows where causal effect estimation failed
        if effect is None or (isinstance(effect, float) and np.isnan(effect)):
            continue

        ref_set = parse_covariate_set(ref)

        # Handle predicted covariates (may be list or string)
        if pred is None or (isinstance(pred, float) and np.isnan(pred)):
            pred_set = set()
        else:
            try:
                pred_str = ", ".join(pred) if isinstance(pred, (list, tuple)) else str(pred)
                pred_set = parse_covariate_set(pred_str)
            except TypeError:
                continue

        ref_len = len(ref_set)
        pred_len = len(pred_set)

        # Both empty is a perfect match
        if ref_len == 0 and pred_len == 0:
            overlap_scores.append(1)
            continue

        # Overlap: intersection over reference size
        overlap = len(ref_set.intersection(pred_set)) / ref_len if ref_len > 0 else np.nan
        overlap_scores.append(overlap)

    return round(np.nanmean(overlap_scores) * 100, 2) if overlap_scores else np.nan

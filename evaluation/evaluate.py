## This file contains functions to evaluate model outputs across all metrics

import pandas as pd
from extract import main_output_extraction, extract_metadata
from metrics import (compute_method_accuracy, compute_method_f1,
                     compute_effect_accuracy, evaluate_treatment,
                     evaluate_outcome, evaluate_covariates, completion_rate)


def evaluate_single(pred_df, meta_df, source):
    """
    Computes all evaluation metrics for a single (prompt, model) combination.

    Args:
        pred_df: (pd.DataFrame) parsed model predictions (output of extract_results)
        meta_df: (pd.DataFrame) reference metadata (output of extract_metadata)
        source: (str) one of 'real', 'qrdata', 'synthetic'

    Returns:
        (dict) metric name -> value
    """

    # 1. Method selection accuracy and F1
    method_accuracy = compute_method_accuracy(meta_df['method'].values, pred_df['pred_method'].values)
    method_f1 = compute_method_f1(meta_df['method'].values, pred_df['pred_method'].values)

    # 2. Causal effect accuracy
    effect_accuracy = compute_effect_accuracy(meta_df['answer'].values, pred_df['pred_effect'].values)

    # 3. Treatment variable selection accuracy
    # Exclude DiD and RDD since the treatment variable may not be present, and one has to construct it. 
    treatment_accuracy = evaluate_treatment(meta_df['treatment'].values, meta_df['multirct_treatment'].values,
                                            pred_df['pred_treatment'].values, meta_df['method'].values,
                                            exclude_methods=['did', 'rdd', 'diff-in-diff'])

    # 4. Outcome variable selection accuracy
    outcome_accuracy = evaluate_outcome(meta_df['outcome'].values, pred_df['pred_outcome'].values)

    # 5. Control variable overlap
    control_overlap = evaluate_covariates( meta_df['covariates'].values, pred_df['pred_covariates'].values,
                                          pred_df['pred_effect'].values)

    # 6. Completion rate and mean attempts
    rate, mean_attempts = completion_rate(pred_df['pred_effect'].values, pred_df['num_tries'].values)

    return {'method_accuracy': method_accuracy, 'method_f1': method_f1, 'effect_accuracy': effect_accuracy,
            'treatment_accuracy': treatment_accuracy, 'outcome_accuracy': outcome_accuracy,
            'control_overlap': control_overlap, 'completion_rate': rate, 'mean_attempts': mean_attempts}


def evaluate_all(output_folder, meta_path, source):
    """
    Runs the full evaluation pipeline across all models and prompting strategies.

    Args:
        output_folder: (str) path to folder containing JSON output files
        meta_path: (str) path to the metadata CSV file
        source: (str) one of 'real', 'qrdata', 'synthetic'

    Returns:
        (pd.DataFrame) one row per (prompt, model) with all metric values
    """

    all_results = main_output_extraction(output_folder)
    meta_df = extract_metadata(meta_path, source)

    # main_output_extraction returns {source: {prompt: {model: df}}}
    rows = []
    for _, prompt_dict in all_results.items():
        for prompt, model_dict in sorted(prompt_dict.items()):
            for model, pred_df in sorted(model_dict.items()):
                print(f"Evaluating: prompt={prompt}, model={model}")
                metrics = evaluate_single(pred_df, meta_df, source)
                rows.append({'prompt': prompt, 'model': model, **metrics})

    return pd.DataFrame(rows)

## This file contains functions to load and parse model outputs and metadata

import os
import json
import numpy as np
import pandas as pd
from utils import standardize_method_name


def compute_significance(effect, se, z_val=1.96):
    """
    Compute the statistical significance of the results

    Args:
        effect: (list) the causal effects
        se: (list) the standard errors
        z_val: (float) z-score threshold for significance

    Returns:
        (list) binary significance values
    """

    effect = np.array(effect)
    se = np.array(se)
    se = np.where(se == 0, np.nan, se)
    significance = (np.abs(effect / se) > z_val).astype(int)

    return significance.tolist()


def extract_results(results_json, query_key="query", name_key="name",
                    method_key="method", effect_key="effect",
                    path_key="dataset_path", pred_results_key="result",
                    pred_results_summary_key="final_result"):
    """
    Extracts the results of the experiments. This is specific to the format of the baseline results
    generated using run_baselines.py. New types of outputs may require modification of this function.
    We strongly recommend manual inspection of the outputs to ensure the parsing is done correctly.

    Args:
        results_json: (list) loaded JSON output from a single model run
        query_key: (str) key denoting the query
        name_key: (str) key denoting the name / source of the study
        method_key: (str) key denoting the true causal inference method
        effect_key: (str) key denoting the true causal effect value
        path_key: (str) key denoting the path to the dataset
        pred_results_key: (str) key denoting the predicted results
        pred_results_summary_key: (str) key denoting the summary nested within pred_results_key

    Returns:
        (pd.DataFrame, dict) DataFrame of results and dict of parsing errors
    """

    df_dict = {"query": [], "name": [], "method": [], "effect": [], "path": [],
               "pred_method": [], "pred_effect": [], "pred_method_ini": [],
               "pred_se": [], "pred_sig": [],
               "pred_treatment": [], "pred_outcome": [], "pred_covariates": [],
               "num_tries": []}

    errors_info = {"query number": [], "query": [], "pred_effect": [], "error": []}

    count = 0
    for result in results_json:
        query = result[query_key]
        name = result[name_key]
        method = result[method_key]
        effect = result[effect_key]
        path = result[path_key]

        try:
            pred_results = result.get(pred_results_key, {}).get(pred_results_summary_key, {})
        except AttributeError:
            pred_results = {}

        pred_method = pred_results.get("method", np.nan)
        pred_effect = pred_results.get("causal_effect", np.nan)
        pred_se = pred_results.get("standard_deviation", np.nan)

        df_dict["query"].append(query)
        df_dict["name"].append(name)
        # Standardize the true method name
        df_dict["method"].append(standardize_method_name(method))
        df_dict["path"].append(path)
        df_dict["effect"].append(effect)
        # Standardize the predicted method name; keep raw version for inspection
        df_dict["pred_method"].append(standardize_method_name(pred_method))
        df_dict["pred_method_ini"].append(pred_method)
        df_dict["pred_treatment"].append(pred_results.get("treatment_variable", None))
        df_dict["pred_outcome"].append(pred_results.get("outcome_variable", None))
        df_dict["pred_covariates"].append(pred_results.get("covariates", None))

        try:
            df_dict["pred_effect"].append(float(pred_effect))
        except (ValueError, TypeError):
            df_dict["pred_effect"].append(np.nan)
            errors_info["query number"].append(count)
            errors_info["query"].append(query)
            errors_info["pred_effect"].append(pred_effect)
            errors_info["error"].append("Could not convert to float")

        try:
            df_dict["pred_se"].append(float(pred_se))
        except (ValueError, TypeError):
            df_dict["pred_se"].append(np.nan)

        df_dict["pred_sig"] = compute_significance(df_dict["pred_effect"], df_dict["pred_se"])

        # Number of retries, capped at 3
        try:
            n = int(np.abs(result.get(pred_results_key).get("retries")))
            df_dict["num_tries"].append(np.minimum(n, 3))
        except (TypeError, AttributeError, ValueError):
            df_dict["num_tries"].append(np.nan)

        count += 1

    return pd.DataFrame(df_dict), errors_info


def main_output_extraction(output_folder, print_error=False):
    """
    Scans a folder for JSON output files and parses each one.
    Expects filenames of the form: {source}_{prompt}_{model}.json

    Args:
        output_folder: (str) path to folder containing JSON output files
        print_error: (bool) whether to print a summary of parsing errors

    Returns:
        (dict) { source: { prompt: { model: pd.DataFrame } } }
    """

    results_all = {}
    error_all = {}
    total = 0

    for filename in sorted(os.listdir(output_folder)):
        if not filename.endswith(".json"):
            continue

        print(f"Processing file: {filename}")
        filepath = os.path.join(output_folder, filename)
        with open(filepath) as f:
            result = json.load(f)

        strip = filename.rstrip(".json").split("_")
        data_source = strip[0]
        prompt_name = strip[1]
        model_name = strip[2]

        if data_source not in results_all:
            results_all[data_source] = {}
        if prompt_name not in results_all[data_source]:
            results_all[data_source][prompt_name] = {}

        df_result, errors_info = extract_results(result)
        error_all[filename] = errors_info
        results_all[data_source][prompt_name][model_name] = df_result
        total = df_result.shape[0]

    if print_error:
        print("--------------------------------------------------------------------")
        print("Summary of errors:")
        for key in error_all:
            print(key, len(error_all[key]["query number"]), "/", total)
        print("--------------------------------------------------------------------")

    return results_all


def extract_metadata(location, source, standardize=True):
    """
    Loads a metadata CSV and optionally standardizes the method column

    Args:
        location: (str) path to the metadata CSV file
        source: (str) one of 'real', 'qrdata', 'synthetic'
        standardize: (bool) whether to standardize method names

    Returns:
        (pd.DataFrame) metadata DataFrame
    """

    df = pd.read_csv(location)

    if standardize:
        df['method'] = [standardize_method_name(m) for m in df['method'].values]

    df['answer'] = pd.to_numeric(df['answer'], errors='coerce')

    # is_significant is only available for real and qrdata
    if source != 'synthetic' and 'is_significant' in df.columns:
        df['is_significant'] = pd.to_numeric(df['is_significant'], errors='coerce')

    return df

## This script contains the functions to compile the json output from the baselines. The focus is on extracting the predicted method and effect from the json output.
## The final output is a csv file that contains the results: method + effect for different models for a give prompting strategy.
## For example, the file real_basic.csv contains the results from different models using the basic prompting strategy on the real data.

## Example: python compile_results.py -if output/qrdata -of errors/qrdata -sd qrdata
## python compile_results.py -if output/synthetic -of errors/synthetic -sd synthetic
## python compile_results.py -if output/real -of errors/real -sd real

## Please see baselines/run_baseline.sh for an example of how to run the baselines and generate the json output

import json 
import pandas as pd
import numpy as np
import argparse 
import os
from pathlib import Path

def parse_arguments():

    parser = argparse.ArgumentParser(description="Arguments for analyzing the errors")
    parser.add_argument("-if", "--input_folder", type=str, 
                      help="Path the the folder containing the results")
    parser.add_argument("-of", "--output_folder", type=str,
                      help="Path the the folder to save the results of the analysis")
    parser.add_argument("-sd", "--source_data", type=str, 
                       help="Source data")

    return parser.parse_args()

def standardize_method_name(method):
    """
    Standardize the method names to a common format. The standardization is based on the outputs from LLMs. 
    We strongly recommend manual inspecttion of the outputs, and update this function to account for method names not described here. 
    Likewise, please check the mappings for inconsistencies. Since the design is based on heuristics, it may not be perfect. 

    Args:
        method (str): The method name to standardize

    Returns:
        (str): The standardized method name
    """

    if method is None:
        return np.nan
    if type(method) != str:
        return np.nan
    method = method.lower()
    
    if "weighting" in method or 'ipw' in method or 'propensity' in method:
        return "ps"
    elif "front" in method or 'frontdoor' in method:
        return "fd"
    elif "discontinuity" in method or 'fuzzy' in method or 'rdd' in method:
        return "rdd"
    elif "in-difference" in method or "did" in method or "in-diff" in method or 'fixed effects' in method or 'panel' in method:
        return "did"
    elif "matching" in method or "observational" in method:
        return "ps"
    elif "logistic" in method or 'probit' in method or 'logit' in method or 'glm' in method:
        return "glm"
    elif "linear" in method or "means" in method or 'ordinary' in method or 'rct' in method or 'ols' in method or 'wls' in method:
        return "ols"
    elif "instrument" in method or "encouragement" in method or "2sls" in method or "iv" in method:
        return "iv"
    elif 'null' in method or 'na' in method or 'n/a' in method or 'none' in method:
        return np.nan
    else:
        return 'other'
    

def analyze_results(results_json, query_key="query", name_key= "paper_name",
                    method_key="method", effect_key="effect",
                    path_key="dataset_path", pred_results_key="result", 
                    pred_results_summary_key="final_result"):
    """
    Analyze the results of the experiments. This is specific to the format of the baseline results
    generated using run_baselines.py. New types of outputs may require modification of this function. 
    We strongly recommend manual inspection of the outputs to ensure the parsing is done correctly.

    Args:
        results_json (dict): The output of baseline experiments in json format
        query_key (str): The key denoting the query
        name_key (str): The key denoting the name / source of the study associated with the query
        method_key (str): The key denoting the causal inference method used
        effect_key (str): The key denoting the causal effect value 
        path_key (str): The key denoting the path to the dataset
        pred_results_key (str): The key denoting the predicted results
        pred_results_summary_key (str): The key denoting the summary of the predicted results. This 
                                        is nested within the pred_results_key. 
    """

    df_dict = {"query": [], "name": [], "method": [], "effect": [], "path": [], 
               "pred_method": [], "pred_effect": [], "pred_method_ini": []}
    errors_info = {"query number": [], "query": [], "pred_effect": [], "error": []}
    true_method_mapping = {}
    pred_method_mapping = {}
    count = 0
    for result in results_json:
        query = result[query_key]
        name = result[name_key]
        method = result[method_key]
        effect = result[effect_key]
        path = result[path_key]
        try:
            pred_results = result.get(pred_results_key, []).get(pred_results_summary_key, [])
        except AttributeError as e:
            print(f"Error in accessing the predicted results for query: {query}. Error: {e}", 
                  result.get(pred_results_key, []))
            #print(result)
            #continue
        pred_method = pred_results.get("method", np.nan)
        pred_effect = pred_results.get("causal_effect", np.nan)
        if method not in true_method_mapping:
            true_method_mapping[method] = []
        true_method_mapping[method].append(standardize_method_name(method))
        df_dict["query"].append(query)
        df_dict["name"].append(name)
        df_dict["method"].append(standardize_method_name(method))
        #if df_dict["method"][-1] == 'other':
        #    print(f"Unrecognized True method name: {method} for query: {query}")
        df_dict["path"].append(path)
        df_dict["effect"].append(effect)
        df_dict["pred_method"].append(standardize_method_name(pred_method))
        df_dict["pred_method_ini"].append(pred_method)
        if pred_method not in pred_method_mapping:
            pred_method_mapping[pred_method] = []
        pred_method_mapping[pred_method].append(standardize_method_name(pred_method))
        #if df_dict["pred_method"][-1] == 'other':
         #   print(f"Unrecognized Predicted method name: {pred_method} for query: {query}")
        try:
            df_dict["pred_effect"].append(float(pred_effect))
        except (ValueError, TypeError):
            #print("Query name:", query)
            #print(f"Could not convert predicted effect '{pred_effect}' to float. Setting as NaN.")
            df_dict["pred_effect"].append(np.nan)
            errors_info["query number"].append(count)
            errors_info["query"].append(query)
            errors_info["pred_effect"].append(pred_effect)
            errors_info["error"].append("Could not convert to float")
        count += 1

    df = pd.DataFrame(df_dict)
    #print(df["pred_method_ini"].unique())
    #print(df["pred_method"].unique())
    print("True method mapping:")
    for key in true_method_mapping:
        true_method_mapping[key] = list(set(true_method_mapping[key]))
    print(true_method_mapping)
    print("Predicted method mapping:")
    for key in pred_method_mapping:
        pred_method_mapping[key] = list(set(pred_method_mapping[key]))
    print(pred_method_mapping)
    print()

    return df, errors_info
    


def combine_dfs(df_all, source, prompt_name):
    """
    Combines the dataframe from different models into a single dataframe. 

    Args:
        df_all (dict): A dictionary containing the dataframes from different models
        source (str): The source of the data
        prompt_name (str): The name of the prompt used
    """

    df_sub = df_all[source][prompt_name]
   
    df_combined = {"query": [], "name": [], "method": [], "effect": [], "path": []}
    for model in df_sub:
        pred_method_key = f"pred_method_{model}"
        pred_effect_key = f"pred_effect_{model}"
        pred_effect_ini_key = f"pred_method_ini_{model}"
        
        if len(df_combined["query"]) == 0:
            df_combined["query"] = list(df_sub[model]["query"])
            df_combined["name"] = list(df_sub[model]["name"])
            df_combined["method"] = list(df_sub[model]["method"])
            df_combined["effect"] = list(df_sub[model]["effect"])
            df_combined["path"] = list(df_sub[model]["path"])
        df_combined[pred_method_key] = df_sub[model]["pred_method"]
        df_combined[pred_effect_key] = df_sub[model]["pred_effect"]
        df_combined[pred_effect_ini_key] = df_sub[model]["pred_method_ini"]
    #for key in df_combined:
    #    print(f"{key}: {len(df_combined[key])}" )
    df_final = pd.DataFrame(df_combined)

    return df_final


def main(args):

    results_all = {}
    error_all = {}
    total = 0

    ## the input folder is the folder containing the json output from the experiments. If conditions are not specific, 
    ## the program will try to process all the json files in the folder.

    for files in os.listdir(args.input_folder):
        if files.endswith(".json"):
            print(f"Processing file: {files}")
            with open(os.path.join(args.input_folder, files), 'r') as f:
                result = json.load(f)
 
            strip = files.rstrip(".json").split("_")
            data_source = strip[0]
            prompt_name = strip[1]
            model_name = strip[2]
            if data_source not in results_all:
                results_all[data_source] = {}
            if prompt_name not in results_all[data_source]:
                results_all[data_source][prompt_name] = {}
            if model_name not in results_all[data_source][prompt_name]:
                results_all[data_source][prompt_name][model_name] = {}
            df_result, errors_info = analyze_results(result)
            error_all[files] = errors_info
            results_all[data_source][prompt_name][model_name] = df_result
            total = df_result.shape[0]

    print("--------------------------------------------------------------------")
    print("Summary of errors:")
    for key in error_all:
        print(key, len(error_all[key]["query number"]), "/", total)
    print("--------------------------------------------------------------------")
    print()

    return results_all

if __name__ == "__main__":
    args = parse_arguments()
    results_processed = main(args)
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(results_processed[args.source_data].keys())

    for pt in ["basic", "cot", "pot", "react"]:
        results_qr = combine_dfs(results_processed, source=args.source_data, prompt_name=pt)
        full_path = output_path / f"{args.source_data}_{pt}.csv"
        results_qr.to_csv(full_path, index=True)
        print()

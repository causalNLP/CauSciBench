## This program creates a json file from a csv file containing the metadata 
## of the queries. 

import pandas as pd
import json 
import argparse
import os 
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert CSV to JSON")
    parser.add_argument("-i", "--input_csv", type=str, required=True, 
                        help="Path to the input CSV file")
    parser.add_argument("-of","--output_folder", type=str, required=True, 
                        help="Path to the output folder")
    parser.add_argument("-on","--output_name", type=str, required=True,
                        help="Name of the output JSON file")   
    
    parser.add_argument("-nc","--name_column", type=str, default="paper_name",
                        help="Name of the column containing the paper name")
    parser.add_argument("-qc","--query_column", type=str, default="natural_language_query",
                        help="Name of the column containing the queries")
    parser.add_argument("-dc","--description_column", type=str, default="data_description",
                        help="Name of the column containing the descriptions")
    parser.add_argument("-fc","--filename_column", type=str, default="data_files",
                        help="Name of the column containing the filenames") 
    
    parser.add_argument("-mc", "--method_column", type=str, default="method",
                        help="Name of the column describing method that is used")
    parser.add_argument("-ec", "--effect_column", type=str, default="answer",
                        help="Name of the column containing the causal effect reference value")
    parser.add_argument("-sc", "--significance_column", type=str, default="is_significant",
                        help="Name of the column containing the statistical significance")
    parser.add_argument("-std", "--error_column", type=str, default="std_error",
                        help="Name of the column containing the standard error")
    
    parser.add_argument("-tc", "--treatment_column", type=str, default="treatment",
                        help="Name of the column containing the treatment variable")
    parser.add_argument("-oc", "--outcome_column", type=str, default="outcome",
                        help="Name of the column containing the outcome variable")
    parser.add_argument("-cc", "--controls_column", type=str, default="control_variables",
                        help="Name of the column containing the control variables")
    parser.add_argument("-rc", "--running_column", type=str, default="running_var",
                        help="Name of the column containing the running variable")
    parser.add_argument("-ic", "--instrument_column", type=str, default="instrument_var",
                        help="Name of the column containing the instrument variable")
    parser.add_argument("-tpc", "--temporal_column", type=str, default="temporal_var",
                        help="Name of the column containing the temporal variable")
    parser.add_argument("-stc", "--state_column", type=str, default="state_var",
                        help="Name of the column containing the state variable")
    parser.add_argument("-itc", "--interaction_column", type=str, default="interaction_var",
                        help="Name of the column containing the interaction variable")

    parser.add_argument("-pc", "--publication_column", type=str, default="publication_year",
                        help="Name of the column containing the publication year")
    parser.add_argument("-doc", "--domain_column", type=str, default="domain",
                        help="Name of the column containing the domain information")



    return parser.parse_args()


def csv_to_json(df, name_column, query_column, description_column, filename_column, 
                method_column, effect_column, std_column, significance_column, 
                treatment_column, outcome_column, controls_column, 
                running_column, instrument_column, temporal_column, state_column, interaction_column,
                publication_column, domain_column, file_path=None):
    """
    Convert a DataFrame to json format
    Args:
        df (pd.DataFrame): The input DataFrame 
        name_column (str): The name of the column containing the names associated to the queries
        query_column (str): The name of the column containing the queries
        description_column (str): The name of the column containing the descriptions
        filename_column (str): The name of the column containing the filenames

        method_column (str): The name of the column containing the method names
        effect_column (str): The name of the column containing the causal effect reference values
        std_column (str): The name of the column containing the standard error
        significance_column (str): The name of the column containing the statistical significance

        treatment_column (str): The name of the column containing the treatment variable
        outcome_column (str): The name of the column containing the outcome variable
        controls_column (str): The name of the column containing the control variables
        running_column (str): The name of the column containing the running variable
        instrument_column (str): The name of the column containing the instrument variable
        temporal_column (str): The name of the column containing the temporal variable
        state_column (str): The name of the column containing the state variable
        interaction_column (str): The name of the column containing the interaction variable
        
        publication_column (str): The name of the column containing the publication year
        domain_column (str): The name of the column containing the domain information
        file_path (str, optional): The path to the file to save the json. Defaults to None.

    Returns:
        (dict): the json object 
    """
    json_list = []
    method_count = {}
    for _, row in df.iterrows():
        try:
            ## general information that serves as input 
            query = row[query_column] 
            description = row[description_column]
            filename = row[filename_column] if file_path is None else f"{file_path}/{row[filename_column]}"
            name = row[name_column]

            ## causal estimation specific
            method = row[method_column]
            if method not in method_count:
                method_count[method] = 0
            method_count[method] += 1
            effect = float(row[effect_column])
            std = float(row[std_column])
            significance = int(row[significance_column]) if pd.notna(row[significance_column]) else None

            ## model specific variables 
            treatment_var = row[treatment_column]
            outcome_var = row[outcome_column]
            controls_var = row[controls_column] if pd.notna(row[controls_column]) else None
            running_var = row[running_column]
            instrument_var = row[instrument_column]
            temporal_var = row[temporal_column]
            state_var = row[state_column]
            interaction_var = row[interaction_column]

            publication = row[publication_column]
            domain = row[domain_column]
        except KeyError as e:
            print(f"Missing column in the input data: {e}")
            continue
        if not os.path.exists(filename):
            print(f"Warning: File {filename} does not exist.")
            continue
        json_list.append({"name": name, "query": query, "dataset_description": description, 
                         "method": method, "dataset_path": filename,
                         "effect":effect, "std_error":std, "is_significant":significance,
                            "treatment_var":treatment_var, "outcome_var":outcome_var,
                            "control_variables":controls_var, "running_var":running_var,
                            "instrument_var":instrument_var, "temporal_var":temporal_var,
                            "state_var":state_var, "interaction_var":interaction_var,
                            "publication_year":publication, "domain":domain
                         })

    return json_list

if __name__ == "__main__":

    args = parse_arguments()
    try:
        df = pd.read_csv(args.input_csv)
        df["answer"] = df["answer"].astype(str).str.replace("−", "-")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input CSV file {args.input_csv} not found.")
    
    ## this is specific to our use case
    path = "data/"
    if "synthetic" in args.input_csv:
        path += "synthetic_data"
    elif "real" in args.input_csv:
        path += "real_data"
    elif "qr" in args.input_csv:
        path += "qrdata"


    dict_data = csv_to_json(df, args.name_column, args.query_column, args.description_column,
                            args.filename_column, args.method_column, args.effect_column,
                            args.error_column, args.significance_column, args.treatment_column,
                            args.outcome_column, args.controls_column, args.running_column,
                            args.instrument_column, args.temporal_column, args.state_column,
                            args.interaction_column, args.publication_column, args.domain_column,
                            file_path=path)  
    
    json_data = json.dumps(dict_data, indent=4)
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name
    if ".json" not in args.output_name:
        output_name += ".json"
    output_file = output_path / f"{args.output_name}"
    with open(output_file, "w") as f:
        f.write(json_data)
    print(f"JSON file saved to {output_file}")

    

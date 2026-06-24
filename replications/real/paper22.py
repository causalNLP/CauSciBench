## This file contains the replication of paper 22
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 22

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper22(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 22
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "classize"
    outcome_vars = ["avgverb", "avgmath"]
    instrument_var = "class_section"
    control_vars = ["tipuach"]
    method = "iv"

    df.loc[df["avgverb"] > 100, "avgverb"] = df.loc[df["avgverb"] > 100, "avgverb"] - 100
    df.loc[df["avgmath"] > 100, "avgmath"] = df.loc[df["avgmath"] > 100, "avgmath"] - 100
    df.loc[df["verbsize"] == 0, "avgverb"] = np.nan
    df.loc[df["mathsize"] == 0, "avgmath"] = np.nan

    sample_filter = ((df["classize"] > 1) & (df["classize"] < 45) &
        (df["c_size"] > 5) & (df["c_leom"] == 1) &
        (df["c_pik"] < 3) & (df["avgverb"].notna()))
    df_sample = df[sample_filter].copy()

    solutions = {}
    for i, outcome_var in enumerate(outcome_vars):
        data = df_sample[df_sample[outcome_var].notna()].copy()
        data = data[[outcome_var, treat_var, instrument_var] + control_vars + ["schlcode"]].dropna()
        exog = pd.DataFrame({"const": 1, "tipuach": data["tipuach"]})
        model = IV2SLS(dependent=data[outcome_var], exog=exog, endog=data[[treat_var]],
                       instruments=data[[instrument_var]]).fit(cov_type="clustered", clusters=data["schlcode"])
        if debug:
            print(model.summary)
        solutions[id_li[i]] = Solution(id_li[i], title, query_li[i], method, dataset_name,
                                       model.params[treat_var], model.std_errors[treat_var],
                                       treat_var=treat_var, outcome_var=outcome_var,
                                       control_vars=control_vars, instrument_var=instrument_var, is_rct=False)

    return solutions


def build_paper22(debug=False):
    """
    Builds the representation of paper 22

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Using Maimonides rule to estimate the effect of class size on scholastic achievementt"
    dataset_name = "angrist_using_maimonides"
    year = 1999
    domain = "economics"
    n_solutions = 2

    query1 = "Does class size affect reading scores of students?"
    query2 = "Does smaller class sizes lead to higher scores in math tests?"

    solutions = replicated_paper22(title, dataset_name, [query1, query2], [42, 43], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

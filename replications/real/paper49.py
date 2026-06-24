## This file contains the replication of paper 49
from pathlib import Path
import pandas as pd
from rdrobust import rdrobust
from solution import Solution, Paper

PAPER_ID = 49

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper49(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 49
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    outcome_var = "mort_age59_related_postHS"
    running_var = "povrate60"
    method = "rdd"

    y = df[outcome_var].values
    x = df[running_var].values
    c = 59.1984

    res = rdrobust(y, x, c=c, kernel="triangular")

    if debug:
        print(res)

    answer = res.coef.loc["Conventional", "Coeff"]
    std_err = res.se.loc["Conventional", "Std. Err."]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=None, outcome_var=outcome_var,
                         running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper49(debug=False):
    """
    Builds the representation of paper 49

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Regression discontinuity designs using covariates"
    dataset_name = "calonico_regression"
    year = 2019
    domain = "health economics"
    n_solutions = 1

    query1 = "Did the Head Start program help reduce the child mortality rates?"

    solutions = replicated_paper49(title, dataset_name, [query1], [88], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

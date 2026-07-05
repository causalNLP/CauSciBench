## This file contains the replication of paper 78
from pathlib import Path
import numpy as np
import pandas as pd
from rdrobust import rdrobust
from solution import Solution, Paper

PAPER_ID = 78

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper78(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 78
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df['corr7_11_diff'] = df['corr0711'] - df['corr0307']
    df['popdiff'] = df['pop'] - 5000
    filtered = df[df['pop'] > 1000].copy()

    treat_var = 'popdiff'
    method = 'rdd'

    ## Solution 1: RDD effect of electing women on corruption change at 5000 population threshold
    outcome_var1 = 'corr7_11_diff'
    d1 = filtered.dropna(subset=[outcome_var1, treat_var])
    result1 = rdrobust(y=d1[outcome_var1].values, x=d1[treat_var].values, all=True)
    if debug:
        print(result1)
    answer1  = float(np.ravel(result1.coef)[2])
    std_err1 = float(np.ravel(result1.se)[2])
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var1,
                         running_var=treat_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper78(debug=False):
    """
    Builds the representation of paper 78

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Does electing women reduce corruption? A regression discontinuity approach"
    dataset_name = "pereira_does_electing"
    year = 2013
    domain = "political science"
    n_solutions = 1

    query1 = "What is the effect of the gender quota enacted in 2007 on corruption?"

    solutions = replicated_paper78(title, dataset_name, [query1], [126], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

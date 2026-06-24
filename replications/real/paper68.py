## This file contains the replication of paper 68
from pathlib import Path
import numpy as np
import pandas as pd
from rdrobust import rdrobust
from solution import Solution, Paper

PAPER_ID = 68

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper68(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 68
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    mask = df[['general2p', 'minorityprimarymargin']].notna().all(axis=1)
    y = df.loc[mask, 'general2p'].to_numpy(dtype=float)
    x = df.loc[mask, 'minorityprimarymargin'].to_numpy(dtype=float)

    treat_var = 'minorityprimarymargin'
    outcome_var = 'general2p'
    method = 'rdd'

    ## Solution 1: RDD effect of minority nominee on general election vote share
    res = rdrobust(y, x)
    if debug:
        print(res.coef)
        print(res.se)
    answer1  = float(np.ravel(res.coef)[0])
    std_err1 = float(np.ravel(res.se)[0])
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         running_var=treat_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper68(debug=False):
    """
    Builds the representation of paper 68

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Evaluating the minority candidate penalty with a regression discontinuity approach"
    dataset_name = "white_evaluating_the_minority"
    year = 2024
    domain = "political science"
    n_solutions = 1

    query1 = "How does effect of nominating a minority candidate affect the general election vote share?"

    solutions = replicated_paper68(title, dataset_name, [query1], [114], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

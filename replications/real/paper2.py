## This file contains the replication of paper 2
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 2

BASE_DIR = Path("../data/csv_files/realdata/")

import numpy as np

def replicated_paper2(title, dataset_name, query_li, id_li,
                      debug=False):
    """
    Replicates the analysis of paper 2
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0)

    treatment_var = "treat"
    outcome_var = "re78"
    method = "ols"

    control_vars = ["age", "education", "black", "hispanic", "nodegree", "re74", "re75"]
    formula = f"{outcome_var} ~ {treatment_var} + I(age**2) + {' + '.join(control_vars)}"
    model = smf.ols(formula, data=df).fit()
    if debug:
        print(model.summary())
    answer1 = model.params[treatment_var]
    std_err1 = model.bse[treatment_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treatment_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=True)

    solution_dict = {id_li[0]: solution1}

    return solution_dict


def build_paper2(debug=False):
    """
    Builds the representation of paper 2

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Propensity score matching for methods for non-experimental causal studies"
    dataset_name = "dehejia_propensity_score"
    year = 2002
    domain = "labor economics"
    is_rct = True
    is_multirct = False
    n_solutions = 1

    query1 = "What is the effect of the training program on job earnings for 1978?"

    solutions = replicated_paper2(title, dataset_name, [query1], [5], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, is_multirct,
                  is_rct, n_solutions)
    return paper

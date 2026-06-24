## This file contains the replication of paper 69
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 69

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper69(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 69
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = 'interaction'
    outcome_var = 'Rate'
    state_var = 'trt'
    time_var = 'year'
    method = 'did'

    ## Solution 1: DiD effect of Zika epidemic on municipal birth rates
    formula1 = 'Rate ~ trt + year + interaction'
    model1 = smf.ols(formula1, data=df).fit()
    if debug:
        print(model1.summary())
    answer1  = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         state_var=state_var, time_var=time_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper69(debug=False):
    """
    Builds the representation of paper 69

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Causal measures using generalized difference-in-difference approach with nonlinear models"
    dataset_name = "taddeo_causal"
    year = 2022
    domain = "public health"
    n_solutions = 1

    query1 = "Does the presence of a Zika epidemic in a municipality lead to lower birth rates?"

    solutions = replicated_paper69(title, dataset_name, [query1], [115], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, 
                  n_solutions=n_solutions)
    
    return paper

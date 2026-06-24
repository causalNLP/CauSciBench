## This file contains the replication of paper 61
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 61

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper61(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 61
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = 'postxapple'
    outcome_var = 'norm_purch_users'
    state_var = 'apple'
    time_var = 'post'
    control_vars = ['game', 'country', 'year', 'month']
    method = 'did'

    ## Solution 1: DiD effect of Apple payment policy change on in-app purchases
    formula1 = 'norm_purch_users ~ post + apple + postxapple + C(game) + C(country) + C(year) + C(month)'
    model1 = smf.ols(formula1, data=df).fit()
    if debug:
        print(model1.summary())
    answer1 = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         is_rct=False)

    return {id_li[0]: solution1}


def build_paper61(debug=False):
    """
    Builds the representation of paper 61

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Demand for in-app purchases in mobile apps—a difference-in-difference approach"
    dataset_name = "enache_demand_for"
    year = 2023
    domain = "economics"
    n_solutions = 1

    query1 = "Does the number of users who purchase premium change after the price of premium is altered?"

    solutions = replicated_paper61(title, dataset_name, [query1], [106], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

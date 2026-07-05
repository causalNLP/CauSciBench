## This file contains the replication of paper 63
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 63

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper63(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 63
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = 'invertedplc'
    outcome_var = 'qxall'
    control_vars = ['pop', 'hc', 'rgdpe']
    method = 'ols'

    ## Solution 1: effect of income inequality on mortality rates
    formula1 = 'qxall ~ invertedplc + pop + hc + rgdpe'
    model1 = smf.ols(formula1, data=df).fit()
    if debug:
        print(model1.summary())
    answer1 = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper63(debug=False):
    """
    Builds the representation of paper 63

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Does rising income inequality affect mortality rates in advanced economies?"
    dataset_name = "rebeira_does_rising"
    year = 2017
    domain = "economics"
    n_solutions = 1

    query1 = "Does a greater inequality in income lead to a higher average mortality rate among males and females?"

    solutions = replicated_paper63(title, dataset_name, [query1], [108], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

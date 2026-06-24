## This file contains the replication of paper 15
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 15

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper15(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 15
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0)

    treat_var = "post:state"
    outcome_var = "fte"
    state_var = "state"
    time_var = "post"
    method = "did"

    formula = f"{outcome_var} ~ {time_var} * {state_var}"
    model = smf.ols(formula, data=df).fit()
    if debug:
        print(model.summary())

    answer1 = model.params[treat_var]
    std_err1 = model.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         state_var=state_var, time_var=time_var,
                         canonical_did=True, is_rct=False)

    return {id_li[0]: solution1}


def build_paper15(debug=False):
    """
    Builds the representation of paper 15

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania"
    dataset_name = "card_minimum_wages"
    year = 1993
    domain = "labor economics"
    n_solutions = 1

    query1 = "Did an increase in minimum wage decrease employment?"

    solutions = replicated_paper15(title, dataset_name, [query1], [28], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper



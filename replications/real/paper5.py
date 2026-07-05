## This file contains the replication of paper 5
from pathlib import Path
import pandas as pd
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 5

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper5(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 5
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0)

    treatment_var = "educ"
    outcome_var = "lwage"
    instrument_var = "nearc4"
    control_vars = ["exper", "black", "south", "smsa"]
    method = "iv"

    formula = "lwage ~ 1 + exper + black + south + smsa + [educ ~ nearc4]"
    model = IV2SLS.from_formula(formula, data=df).fit()
    if debug:
        print(model.summary)
    answer1 = model.params[treatment_var]
    std_err1 = model.std_errors[treatment_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treatment_var, outcome_var=outcome_var,
                         instrument_var=instrument_var, control_vars=control_vars, is_rct=False)

    solution_dict = {id_li[0]: solution1}

    return solution_dict


def build_paper5(debug=False):
    """
    Builds the representation of paper 5

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Using geographic variation in college proximity to estimate the return to schooling"
    dataset_name = "card_using_geographic"
    year = 1993
    domain = "labor economics"
    n_solutions = 1

    query1 = "What is the effect of education on earnings?"

    solutions = replicated_paper5(title, dataset_name, [query1], [9], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions,  n_solutions=n_solutions)
    return paper

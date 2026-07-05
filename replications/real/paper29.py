## This file contains the replication of paper 29
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 29

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper29(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 29
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "HASCOMPSTAT"
    outcome_var = "PART2arrests"
    state_var = "AGENCY"
    time_var = "YEAR"
    control_vars = ["POP"]
    method = "did"

    data = df[[outcome_var, treat_var, state_var, time_var] + control_vars].dropna()

    formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)} + C({state_var}) + C({time_var})"
    model = smf.ols(formula, data=data).fit()

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         canonical_did=False, is_rct=False)

    return {id_li[0]: solution1}


def build_paper29(debug=False):
    """
    Builds the representation of paper 29

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Metrics management and bureaucratic accountability: Evidence from policing"
    dataset_name = "eckhouse_metrics_management"
    year = 2019
    domain = "political science"
    n_solutions = 1

    query1 = "What is the effect of CompStat adoption on the number of Part 2 arrests?"

    solutions = replicated_paper29(title, dataset_name, [query1], [50], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

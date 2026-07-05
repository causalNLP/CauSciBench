## This file contains the replication of paper 17
from pathlib import Path
from linearmodels.iv import IV2SLS
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 17

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper17(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 17
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path)

    treat_var = "p"
    outcome_var = "q"
    instrument_var = "stormy"
    control_vars = ["mon", "tue", "wed", "thu", "cold", "rainy"]
    method = "iv"

    formula = f"{outcome_var} ~ 1 + {' + '.join(control_vars)} + [{treat_var} ~ {instrument_var}]"
    model = IV2SLS.from_formula(formula, data=df).fit(cov_type="robust")
    if debug:
        print(model.summary)

    answer1 = model.params[treat_var]
    std_err1 = model.std_errors[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var,
                         is_rct=False)

    return {id_li[0]: solution1}


def build_paper17(debug=False):
    """
    Builds the representation of paper 17

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Markets: The fulton fish market"
    dataset_name = "graddy_markets_fulton"
    year = 2006
    domain = "economics"
    n_solutions = 1

    query1 = "What is the effect of fish prices on quantity demanded?"

    solutions = replicated_paper17(title, dataset_name, [query1], [30], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

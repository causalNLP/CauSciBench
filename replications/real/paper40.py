## This file contains the replication of paper 40
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 40

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper40(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 40
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "x"
    outcome_var = "y_us"
    control_vars = ["usaid", "gdppc", "wbgi", "fhscore", "africa", "trade_i", "trade_e", "unvote_key"]
    method = "iv"

    instruments = [col for col in df.columns if col.startswith("z_")]

    data = pd.concat([df[outcome_var], df[treat_var], sm.add_constant(df[control_vars]),
                      df[instruments]], axis=1).dropna()

    model = IV2SLS(dependent=data[outcome_var],
                   exog=data[["const"] + control_vars],
                   endog=data[[treat_var]],
                   instruments=data[instruments]).fit(cov_type="robust")

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.std_errors[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=", ".join(instruments), is_rct=False)

    return {id_li[0]: solution1}


def build_paper40(debug=False):
    """
    Builds the representation of paper 40

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Doing well by doing good: The impact of foreign aid on foreign public opinion"
    dataset_name = "goldsmith_doing_well"
    year = 2014
    domain = "political economics"
    n_solutions = 1

    query = "Does PEPFAR funding increase approval of U.S. leadership?"

    solutions = replicated_paper40(title, dataset_name, [query], [69], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

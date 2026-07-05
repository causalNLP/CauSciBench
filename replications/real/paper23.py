## This file contains the replication of paper 23
from pathlib import Path
import pandas as pd
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 23

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper23(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 23
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "d_tradeusch_pw"
    outcome_var = "d_sh_empl_mfg"
    instrument_var = "d_tradeotch_pw_lag"
    control_vars = ["t2"]
    method = "iv"

    data = df[[outcome_var, treat_var, instrument_var, "timepwt48"] + control_vars].dropna()
    data = data.copy()
    data["const"] = 1.0

    model = IV2SLS(dependent=data[outcome_var], exog=data[["const"] + control_vars],
                   endog=data[[treat_var]], instruments=data[[instrument_var]],
                   weights=data["timepwt48"]).fit()

    if debug:
        print(model.summary)

    answer = model.params[treat_var]
    std_err = model.std_errors[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper23(debug=False):
    """
    Builds the representation of paper 23

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The China syndrome: Local labor market effects of import competition in the United States"
    dataset_name = "autor_china_syndrome"
    year = 2013
    domain = "labor economics"
    n_solutions = 1

    query = "Does an increase in Chinese import competition reduce the manufacturing employment share in local labor markets?"

    solutions = replicated_paper23(title, dataset_name, [query], [44], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    
    return paper

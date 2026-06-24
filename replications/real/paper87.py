## This file contains the replication of paper 87
from pathlib import Path
import pandas as pd
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 87

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper87(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 87
    """

    data = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "EMST"
    outcome_var = "y1"
    instrument_var = "meanwSIM"
    control_vars = ["newmopan"]
    method = "iv"

    df = data[["y1", "EMST", "newmopan", "meanwSIM", "hq"]].dropna()
    df["hq"] = df["hq"].astype("category")
    df["newmopan"] = df["newmopan"].astype("float")

    ## Solution 1: IV effect of earmarked funding share on IO performance
    formula = "y1 ~ 1 + newmopan + [EMST ~ meanwSIM]"
    res = IV2SLS.from_formula(formula, data=df).fit()
    if debug:
        print(res)
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         float(res.params[treat_var]), float(res.std_errors[treat_var]),
                         treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper87(debug=False):
    """
    Builds the representation of paper 87

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Does earmarked funding affect the performance of international organisations?"
    dataset_name = "reinsberg_does_earmarked"
    year = 2024
    domain = "international relations"
    n_solutions = 1

    query1 = "How much does earmarked funding affect the performance of international organisations?"

    solutions = replicated_paper87(title, dataset_name, [query1], [150], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

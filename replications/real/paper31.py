## This file contains the replication of paper 31
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 31

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper31(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 31
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "mwdef"
    outcome_var = "probit1620"
    state_var = "state"
    time_var = "year"
    control_vars = ["lnorate1620", "lpop", "realbeertax", "bac", "lpc_perinc", "ur1620_r"]
    method = "ols"

    zero_replace_vars = ["DUI_1620", "nonDUI_1620"]
    for var in zero_replace_vars:
        df[var] = df[var].replace(0, 0.1)

    df["probit1620"] = pd.to_numeric(stats.norm.ppf(df["DUI_1620"] / df["pop1620"]), errors="coerce")
    df["lnorate1620"] = np.log(df["nonDUI_1620"] / df["pop1620"])

    needed = [outcome_var, treat_var, state_var, time_var] + control_vars
    data = df[needed].dropna()

    formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)} + C({state_var}) + C({time_var})"
    cluster = pd.factorize(data[state_var])[0]
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": cluster})

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         is_rct=False)

    return {id_li[0]: solution1}


def build_paper31(debug=False):
    """
    Builds the representation of paper 31

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Are minimum wages a silent killer? New evidence on drunk driving fatalities"
    dataset_name = "sabia_are_minimum"
    year = 2019
    domain = "labor economics"
    n_solutions = 1

    query1 = "What is the effect of the minimum wage on alcohol‑related fatal accidents involving drivers ages 16–20?"

    solutions = replicated_paper31(title, dataset_name, [query1], [52], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

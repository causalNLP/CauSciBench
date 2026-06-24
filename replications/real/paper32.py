## This file contains the replication of paper 32
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 32

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper32(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 32
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "ever_treated"
    outcome_var = "trnt"
    state_var = "loc"
    time_var = "year"
    method = "did"

    df = df[(df["count"] == 5) & (df["loc"] != "Tower Hamlets")].copy()
    df["ever_treated"] = np.where(df["loc"] == "Bromley", 1, 0)

    data = df[df["trnt"] > 0].copy()
    data["year_factor"] = data["year"].astype("category")

    formula = "trnt ~ ever_treated * C(year_factor)"
    model = smf.ols(formula, data=data).fit()

    if debug:
        print(model.summary())

    coef_name = [c for c in model.params.index if "ever_treated:C(year_factor)" in c and "2018" in c][0]
    answer = model.params[coef_name]
    std_err = model.bse[coef_name]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         state_var=state_var, time_var=time_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper32(debug=False):
    """
    Builds the representation of paper 32

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Understanding the impact of the 2018 voter id pilots on turnout at the london local elections: A synthetic difference-in-difference approach"
    dataset_name = "barton_understanding"
    year = 2025
    domain = "political science"
    n_solutions = 1

    query1 = "What is the effect of requiring one photo ID or two non‑photo IDs at polling stations on turnout in Bromley's 2018 local election?"

    solutions = replicated_paper32(title, dataset_name, [query1], [53], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

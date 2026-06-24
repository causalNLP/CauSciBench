## This file contains the replication of paper 34
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from solution import Solution, Paper

PAPER_ID = 34

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper34(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 34
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "treat"
    outcome_var = "share_detained_sheriff"
    running_var = "rv"
    control_vars = ["rv", "treat_rv"]
    method = "rdd"

    data = df[np.abs(df["rv"]) < 10].copy()
    data = data.dropna(subset=[outcome_var, treat_var, "rv", "treat_rv", "election_id"])

    y = data[outcome_var]
    X = sm.add_constant(data[[treat_var] + control_vars])
    cluster = pd.factorize(data["election_id"])[0]

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster})

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper34(debug=False):
    """
    Builds the representation of paper 34

    Returns:
        (Paper): The constructed Paper object
    """

    title = "How partisan is local law enforcement? Evidence from sheriff cooperation with immigration authorities"
    dataset_name = "thompson_how_partisan"
    year = None
    domain = "criminology"
    n_solutions = 1

    query1 = "What is the effect of electing a Democratic on cooperation with federal immigration authorities?"

    solutions = replicated_paper34(title, dataset_name, [query1], [55], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

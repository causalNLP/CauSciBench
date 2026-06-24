## This file contains the replication of paper 56
from pathlib import Path
import pandas as pd
import numpy as np
from solution import Solution, Paper

PAPER_ID = 56

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper56(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 56
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df["tx"] = (df["group"] == "SH:24").astype(int)
    df = df.dropna(subset=["anytest"])

    treat_var = "tx"
    outcome_var = "anytest"
    method = "ols"

    rates = df.groupby(treat_var)[outcome_var].mean()
    n0 = (df[treat_var] == 0).sum()
    n1 = (df[treat_var] == 1).sum()

    p0 = rates.loc[0]
    p1 = rates.loc[1]
    answer = p1 - p0
    std_err = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)

    if debug:
        print(f"Control: {p0:.4f}, Treated: {p1:.4f}, ATE: {answer:.4f}, SE: {std_err:.4f}")

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         is_rct=True)

    return {id_li[0]: solution1}


def build_paper56(debug=False):
    """
    Builds the representation of paper 56

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Internet-accessed sexually transmitted infection (e-sti) testing and results service: A randomised, single-blind, controlled trial"
    dataset_name = "wilson_internet"
    year = 2017
    domain = "public health"
    n_solutions = 1

    query1 = "What is the effect of the e-STI testing service on the probability of getting tested for an STI by 6 weeks?"

    solutions = replicated_paper56(title, dataset_name, [query1], [97], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    
    return paper

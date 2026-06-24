## This file contains the replication of paper 18
from pathlib import Path
from rdd.rdd import rdd
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 18

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper18(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 18
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path)

    treat_var = "dui"
    outcome_var = "recidivism"
    running_var = "bac1"
    method = "rdd"

    df[treat_var] = (df[running_var] > 0.08).astype(int)
    df = df[(df[running_var] > 0.03) & (df[running_var] < 0.13)]
    data = df[[outcome_var, running_var]].dropna()

    model = rdd(data, running_var, outcome_var, cut=0.08).fit()
    if debug:
        print(model.summary())

    answer1 = model.params["TREATED"]
    std_err1 = model.bse["TREATED"]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper18(debug=False):
    """
    Builds the representation of paper 18

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Punishment and deterrence: Evidence from drunk driving"
    dataset_name = "hansen_punishment"
    year = 2015
    domain = "law"
    n_solutions = 1

    query1 = "Does receiving a DUI conviction (BAC ≥ 0.08) reduce the likelihood of future drunk driving compared to those just below the legal limit?"

    solutions = replicated_paper18(title, dataset_name, [query1], [31], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

## This file contains the replication of paper 57
from pathlib import Path
import pandas as pd
from rdrobust import rdrobust
from solution import Solution, Paper

PAPER_ID = 57

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper57(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 57
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df = df[df["office_num"] == "UK, House"].dropna(subset=["win", "l_vmargin"])

    outcome_var = "win"
    running_var = "l_vmargin"
    method = "rdd"

    res = rdrobust(y=df[outcome_var].values, x=df[running_var].values, kernel="uniform")

    if debug:
        print(res)

    answer = res.coef.loc["Conventional", "Coeff"]
    std_err = res.se.loc["Conventional", "Std. Err."]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=None, outcome_var=outcome_var,
                         running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper57(debug=False):
    """
    Builds the representation of paper 57

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Estimating incumbency effects using regression discontinuity design"
    dataset_name = "song_estimating_incumbency"
    year = 2018
    domain = "political science"
    n_solutions = 1

    query1 = "In the UK, in very close elections, does just barely winning make it more likely that the party's candidate will stand again as the incumbent in the next election?"

    solutions = replicated_paper57(title, dataset_name, [query1], [98], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

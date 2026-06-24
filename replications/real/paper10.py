## This file contains the replication of paper 10
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 10

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper10(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 10
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0).reset_index()

    treat_var = "participation"
    outcome_var = "support"
    running_var = "income_centered"
    method = "rdd"

    model = smf.ols("support ~ income_centered * participation + I(income_centered**2) * participation",
                    data=df).fit()
    if debug:
        print(model.summary())

    answer1 = model.params[treat_var]
    std_err1 = model.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper10(debug=False):
    """
    Builds the representation of paper 10

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Government transfers and political support"
    dataset_name = "manacorda_gov_transfers"
    year = 2011
    domain = "political economics"
    n_solutions = 1

    query1 = "Did receiving financial assistance from Uruguay's poverty alleviation program cause recipients to view the government more favorably?"

    solutions = replicated_paper10(title, dataset_name, [query1], [19], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

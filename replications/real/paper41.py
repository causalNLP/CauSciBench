## This file contains the replication of paper 41
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from solution import Solution, Paper

PAPER_ID = 41

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper41(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 41
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "mccain"
    outcome_var = "interest_in_letter"
    control_vars = ["circulation", "unemployment_metro"]
    method = "glm"

    data = df[[outcome_var, treat_var] + control_vars].dropna()

    y = data[outcome_var]
    X = sm.add_constant(data[[treat_var] + control_vars])

    model = sm.Probit(y, X).fit(disp=0)

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=True)

    return {id_li[0]: solution1}


def build_paper41(debug=False):
    """
    Builds the representation of paper 41

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Were newspapers more interested in pro-Obama letters to the editor in 2008? Evidence from a field experiment"
    dataset_name = "butler_were_newspapers"
    year = 2010
    domain = "political science"
    n_solutions = 1

    query1 = "Does receipt of a pro-McCain letter affect a paper's probability of expressing interest in publishing it?"

    solutions = replicated_paper41(title, dataset_name, [query1], [70], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

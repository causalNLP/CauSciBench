## This file contains the replication of paper 37
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 37

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper37(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 37
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "prezg"
    outcome_var = "turn00"
    control_vars = ["senate"]
    method = "ols"

    data = df[[outcome_var, treat_var] + control_vars].dropna()

    formula = f"{outcome_var} ~ {treat_var} + {' + '.join(control_vars)}"
    model = smf.ols(formula, data=data).fit()

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper37(debug=False):
    """
    Builds the representation of paper 37

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Do televised presidential ads increase voter turnout? Evidence from a natural experiment"
    dataset_name = "krasno_do_televised"
    year = 2008
    domain = "political science"
    n_solutions = 1

    query1 = "How does an increase in presidential advertising intensity (measured in Gross Ratings Points) affect voter turnout in the presidential election?"

    solutions = replicated_paper37(title, dataset_name, [query1], [58], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

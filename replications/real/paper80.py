## This file contains the replication of paper 80
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 80

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper80(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 80
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df['income_change'] = df['re78'] - df['re75']

    treat_var = 'treat'
    outcome_var = 're78'
    control_vars = ['re75', 'age', 'I(age**2)']
    method = 'ols'

    formula1 = 'income_change ~ treat + age + I(age**2)'
    model1 = smf.ols(formula1, data=df).fit(cov_type='nonrobust')
    if debug:
        print(model1.summary())
    answer1  = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=True)

    return {id_li[0]: solution1}


def build_paper80(debug=False):
    """
    Builds the representation of paper 80

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Evaluating the Econometric Evaluations of Training Programs with Experimental Data"
    dataset_name = "lalonde_evaluating"
    year = 1986
    domain = "labor economics"
    n_solutions = 1

    query1 = "What is the effect of the training program on earnings when considering earnings in 1978 and 1975?"

    solutions = replicated_paper80(title, dataset_name, [query1], [128], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions,
                  n_solutions=n_solutions, is_rct=True)
    return paper

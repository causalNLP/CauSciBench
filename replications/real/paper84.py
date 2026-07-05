## This file contains the replication of paper 84
from pathlib import Path
import pandas as pd
from statsmodels.regression.linear_model import OLS
from solution import Solution, Paper

PAPER_ID = 84

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper84(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 84
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    bandwidth = 0.15
    y_name = 'femaleonballotnextyear'
    treat_var = 'womanwon'
    method = 'rdd'

    data = df[df['absolute_margin'] < bandwidth].copy()
    treatment_and_covariates = data.iloc[:1].loc[:, 'womanwon':'winXfem_v_c_p_4'].columns.tolist()
    running_var = 'femalecand_margin_of_victory'
    control_vars = [c for c in treatment_and_covariates if c not in (treat_var, running_var)]
    data = data[[y_name] + treatment_and_covariates]

    ## Solution 1: RDD effect of woman winning on female candidates on ballot next year
    formula = f'{y_name} ~ {" + ".join(treatment_and_covariates)}'
    results = OLS.from_formula(formula, data=data).fit()
    if debug:
        print(results.summary())
    answer1  = float(results.params[treat_var])
    std_err1 = float(results.bse[treat_var])
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=y_name,
                         control_vars=control_vars, running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper84(debug=False):
    """
    Builds the representation of paper 84

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Do female politicians empower women to vote or run for office? A regression discontinuity approach"
    dataset_name = "broockman_do_female"
    year = 2014
    domain = "political science"
    n_solutions = 1

    query1 = "Does electing women in districts with close races empower other women to vote at the next election?"

    solutions = replicated_paper84(title, dataset_name, [query1], [145], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

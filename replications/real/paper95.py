## This file contains the replication of paper 95
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from solution import Solution, Paper

PAPER_ID = 95

BASE_DIR = Path("../data/csv_files/realdata/")

X_VARS = ['post', 'linear', 'square', 'linear_post', 'square_post',
          'birthday_19', 'birthday_19_1', 'birthday_20', 'birthday_20_1',
          'birthday_21', 'birthday_21_1', 'birthday_22', 'birthday_22_1',
          'birthday_23', 'birthday_23_1']


def replicated_paper95(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 95
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv", encoding="latin-1")
    data = df[(df['days_to_21'] >= -2*365) & (df['days_to_21'] <= 2*365 - 1)]

    treat_var = "post"
    running_var = "days_to_21"
    method = "rdd"

    def _fit(outcome_var):
        X = sm.add_constant(data[X_VARS])
        m = sm.OLS(data[outcome_var], X).fit(cov_type='HC1')
        if debug:
            print(f"{outcome_var}: coef={m.params['post']:.4f}, se={m.bse['post']:.4f}")
        return float(m.params['post']), float(m.bse['post'])

    outcomes = ['all_r', 'violent_r', 'property_r', 'ill_drugs_r', 'alcohol_r']
    solutions = {}
    for i, (ov, id_, query) in enumerate(zip(outcomes, id_li, query_li)):
        coef, se = _fit(ov)
        solutions[id_] = Solution(id_, title, query, method, dataset_name,
                                  coef, se, treat_var=treat_var, outcome_var=ov,
                                  running_var=running_var, is_rct=False)
    return solutions


def build_paper95(debug=False):
    """
    Builds the representation of paper 95

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The minimum legal drinking age and crime"
    dataset_name = "carpenter_the_minimum_legal_drinking"
    year = 2015
    domain = "criminology"
    n_solutions = 5

    query1 = "How does gaining legal access to alcohol at age 21 affect overall arrest rates?"
    query2 = "How does gaining legal access to alcohol at age 21 affect violent crime arrest rates?"
    query3 = "How does gaining legal access to alcohol at age 21 affect property crime arrest rates?"
    query4 = "How does gaining legal access to alcohol at age 21 affect illegal drugs arrest rates?"
    query5 = "How does gaining legal access to alcohol at age 21 affect alcohol related crime arrest rates?"

    solutions = replicated_paper95(title, dataset_name,
                                   [query1, query2, query3, query4, query5],
                                   [169, 170, 171, 172, 173], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

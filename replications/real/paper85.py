## This file contains the replication of paper 85
from pathlib import Path
import pandas as pd
from statsmodels.regression.linear_model import OLS
from solution import Solution, Paper

PAPER_ID = 85

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper85(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 85
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    j = 2  # closeness threshold: +/- 2%
    df = df[~df['dist'].astype(str).str.contains('97')]
    df = df[df['type'] == 'G']
    df = df[df['office'] != 'H']
    df = df[(df['year'] >= 1946) & (df['year'] <= 2010)]
    df['stateyear'] = df['state'] + df['year'].astype(str)
    df['dem_share_d1'] = (df['dem_share'] - 0.5) * (df['dem_share'] > 0.5)
    df['close'] = (abs(df['dem_share'] - 0.5) < j / 100) & df['dem_share'].notna()

    treat_var = 'dem_win'
    running_var = 'dem_share'
    control_vars = ['dem_share_d1']
    method = 'rdd'

    def _fit(outcome_var):
        sub = df[df['close']][[outcome_var, treat_var, 'dem_share', 'dem_share_d1', 'stateyear']].dropna()
        res = OLS.from_formula(f'{outcome_var} ~ {treat_var} + dem_share + dem_share_d1', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['stateyear']})
        if debug:
            print(res.summary())
        return res

    ## Solution 1: effect of dem winning on dem winning next election
    res1 = _fit('dem_share_next')
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         float(res1.params[treat_var]), float(res1.bse[treat_var]),
                         treat_var=treat_var, outcome_var='dem_share_next',
                         control_vars=control_vars, running_var=running_var, is_rct=False)

    ## Solution 2: effect of dem winning on sitting governor being dem
    res2 = _fit('gov_dem')
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         float(res2.params[treat_var]), float(res2.bse[treat_var]),
                         treat_var=treat_var, outcome_var='gov_dem',
                         control_vars=control_vars, running_var=running_var, is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper85(debug=False):
    """
    Builds the representation of paper 85

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Partisan imbalance in regression discontinuity studies based on electoral thresholds"
    dataset_name = "snyder_partisan_imbalance"
    year = 2015
    domain = "political science"
    n_solutions = 2

    query1 = "What is the effect of the democratic candidate winning the election on the democratic candidate winning the next election in statewide races?"
    query2 = "What is the effect of the democratic candidate winning the election on the sitting governor being democratic in statewide races?"

    solutions = replicated_paper85(title, dataset_name, [query1, query2], [146, 147], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

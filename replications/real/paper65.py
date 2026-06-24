## This file contains the replication of paper 65
from pathlib import Path
import statsmodels.api as sm
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 65

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper65(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 65
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    df = df[df['period'] >= 2]
    df = df.dropna(subset=['log_fn', 'CAO_period_3', 'CAO', 'period_3', 'codeinsee'])

    treat_var = 'CAO_period_3'
    outcome_var = 'log_fn'
    state_var = 'CAO'
    time_var = 'period_3'
    method = 'did'

    ## Solution 1: DiD effect of migrant relocation on extreme right voting (no controls, no FE)
    X1 = sm.add_constant(df[['CAO_period_3', 'CAO', 'period_3']].astype(float))
    y1 = df['log_fn'].astype(float)
    model1 = sm.OLS(y1, X1).fit(cov_type='cluster', cov_kwds={'groups': df['codeinsee']})
    if debug:
        print(model1.summary())
    answer1 = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         state_var=state_var, time_var=time_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper65(debug=False):
    """
    Builds the representation of paper 65

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Dismantling the jungle: Migrant relocation and extreme voting in France"
    dataset_name = "vertier_dismantling"
    year = 2023
    domain = "political science"
    n_solutions = 1

    query1 = "What is the causal effect of having a migrant reception center (CAO) on the vote share of the Front National (FN) between 2012 and 2017?"

    solutions = replicated_paper65(title, dataset_name, [query1], [111], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

## This file contains the replication of paper 77
from pathlib import Path
from linearmodels.iv import IV2SLS
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 77

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper77(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 77
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df_clean = df.dropna(subset=['sow3', 'insuredpig', 'groupd2', 'groupd3'])

    treat_var = 'insuredpig'
    outcome_var = 'sow3'
    instrument_var = 'groupd2, groupd3'
    method = 'iv'

    ## Solution 1: IV effect of pig insurance take-up on sow count
    iv_formula1 = 'sow3 ~ 1 + [insuredpig ~ groupd2 + groupd3]'
    model1 = IV2SLS.from_formula(iv_formula1, data=df_clean).fit(cov_type='robust')
    if debug:
        print(model1.summary)
    answer1  = model1.params[treat_var]
    std_err1 = model1.std_errors[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         instrument_var=instrument_var, is_rct=True)

    return {id_li[0]: solution1}


def build_paper77(debug=False):
    """
    Builds the representation of paper 77

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The effect of microinsurance on economic activities: Evidence from a randomized field experiment"
    dataset_name = "cai_effect_microinsurance"
    year = 2015
    domain = "economics"
    n_solutions = 1

    query1 = "What is the causal effect of having a sow insured on the subsequent number of sows owned by a farmer?"

    solutions = replicated_paper77(title, dataset_name, [query1], [125], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, is_rct=True, n_solutions=n_solutions)
    return paper

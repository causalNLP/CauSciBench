## This file contains the replication of paper 82
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from solution import Solution, Paper

PAPER_ID = 82

BASE_DIR = Path("../data/csv_files/realdata/")


def _load(dataset_name):
    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    for col in ['ethnicity', 'workstatus']:
        df[col] = df[col].fillna(-1)
    return df


def _regress(df, outcome, treatments, controls):
    cols = [outcome] + treatments + controls
    d = df[cols].dropna()
    y = d[outcome]
    X = sm.add_constant(d[treatments + controls])
    return OLS(y, X).fit(cov_type='HC1')


def replicated_paper82(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 82
    """

    df = _load(dataset_name)

    method = 'ols'
    base_controls = ['age', 'female', 'lucid_hhi', 'lucid_incmissing', 'ethnicity',
                     'education', 'workstatus', 'fluvaccine'] + \
                    [f'partisanship_{i}' for i in range(1, 8) if i != 4]

    beliefs_controls = base_controls
    self_controls = ['pool_self_scenario_atm', 'pool_self_scenario_meet'] + base_controls
    other_controls = ['pool_other_scenario_atm', 'pool_other_scenario_meet'] + base_controls

    ## Beliefs: mskblfs_prtctothers
    res_others = _regress(df, 'mskblfs_prtctothers',
                          ['T_mask_protectyou', 'T_mask_protectother'], beliefs_controls)
    if debug:
        print(res_others.summary())

    ## Beliefs: mskblfs_prtctyou
    res_you = _regress(df, 'mskblfs_prtctyou',
                       ['T_mask_protectyou', 'T_mask_protectother'], beliefs_controls)
    if debug:
        print(res_you.summary())

    ## Pool self
    res_self = _regress(df, 'pool_self',
                        ['T_mask_protectyou', 'T_mask_protectother', 'pool_self_others'],
                        self_controls)
    if debug:
        print(res_self.summary())

    ## Pool other
    res_other = _regress(df, 'pool_other',
                         ['T_mask_protectyou', 'T_mask_protectother', 'pool_other_others'],
                         other_controls)
    if debug:
        print(res_other.summary())

    def sol(idx, tv, ov, res, ctrl):
        return Solution(id_li[idx], title, query_li[idx], method, dataset_name,
                        float(res.params[tv]), float(res.bse[tv]),
                        treat_var=tv, outcome_var=ov, control_vars=ctrl, is_rct=True)

    ## pool_other (3 solutions)
    s1 = sol(0, 'pool_other_others',  'pool_other', res_other, other_controls)
    s2 = sol(1, 'T_mask_protectother','pool_other', res_other, other_controls)
    s3 = sol(2, 'T_mask_protectyou',  'pool_other', res_other, other_controls)
    ## pool_self (3 solutions)
    s4 = sol(3, 'pool_self_others',   'pool_self',  res_self,  self_controls)
    s5 = sol(4, 'T_mask_protectother','pool_self',  res_self,  self_controls)
    s6 = sol(5, 'T_mask_protectyou',  'pool_self',  res_self,  self_controls)
    ## mskblfs_prtctyou (2 solutions)
    s7 = sol(6, 'T_mask_protectother','mskblfs_prtctyou', res_you, beliefs_controls)
    s8 = sol(7, 'T_mask_protectyou',  'mskblfs_prtctyou', res_you, beliefs_controls)
    ## mskblfs_prtctothers (2 solutions)
    s9  = sol(8, 'T_mask_protectother','mskblfs_prtctothers', res_others, beliefs_controls)
    s10 = sol(9, 'T_mask_protectyou', 'mskblfs_prtctothers', res_others, beliefs_controls)

    return {id_li[i]: s for i, s in enumerate([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10])}


def build_paper82(debug=False):
    """
    Builds the representation of paper 82

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Experimental evidence that changing beliefs about mask efficacy and social norms increase mask wearing for covid-19 risk reduction: Results from the United states and Italy"
    dataset_name = "bokemper_experimental_evidence"
    year = 2021
    domain = "public health"
    n_solutions = 10

    query1 = "How much does greater mask use increase the likelihood that someone would encourage others to wear a mask?"
    query2 = "How much does providing information about how masks protect others increase the likelihood that someone would encourage others to wear a mask?"
    query3 = "How much does providing information about how masks protect the wearer increase the likelihood that someone would encourage others to wear a mask?"
    query4 = "How much does greater mask use increase the likelihood that someone would wear a mask?"
    query5 = "How much does providing information about how masks protect others increase the likelihood that someone would wear a mask?"
    query6 = "How much does providing information about how masks protect the wearer increase the likelihood that someone would wear a mask?"
    query7 = "How much does providing information about how masks protect others increase the belief that masks protect the wearer?"
    query8 = "How much does providing information about how masks protect the wearer increase the belief that masks protect the wearer?"
    query9 = "How much does providing information about how masks protect others increase the belief that masks protect others?"
    query10 = "How much does providing information about how masks protect the wearer increase the belief that masks protect others?"

    queries = [query1, query2, query3, query4, query5, query6, query7, query8, query9, query10]

    solutions = replicated_paper82(title, dataset_name, queries,
                                   list(range(133, 143)), debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions,
                  n_solutions=n_solutions, is_rct=True)
    
    return paper

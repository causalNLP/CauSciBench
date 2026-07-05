## This file contains the replication of paper 59
from pathlib import Path
import statsmodels.api as sm
import numpy as np
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 59

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper59(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 59
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    male_df   = df[df['male'] == 1].copy()
    female_df = df[df['male'] == 0].copy()

    outcome_var    = 'income'
    treat_var      = 'JTPA training participation'
    instrument_var = 'JTPA training offer'
    method         = 'iv'

    control_vars1 = ['hsorged', 'black', 'hispanic', 'married', 'wkless13',
        'class_tr', 'ojt_jsa', 'age2225', 'age2629', 'age3035', 'age3644', 'age4554', 'f2sms']
    control_vars2 = ['hsorged', 'black', 'hispanic', 'married', 'wkless13', 'afdc',
        'class_tr', 'ojt_jsa', 'age2225', 'age2629', 'age3035', 'age3644', 'age4554', 'f2sms']

    ## Solution 1: IV effect of JTPA training on men's earnings
    d1 = male_df.dropna(subset=[outcome_var, treat_var, instrument_var] + control_vars1)
    O1 = np.ones(len(d1))
    X1_fs = np.column_stack([O1, d1[instrument_var].values, d1[control_vars1].values])
    first1 = sm.OLS(d1[treat_var].values, X1_fs).fit()
    D_hat1 = first1.predict(X1_fs)
    X1_ss = np.column_stack([O1, D_hat1, d1[control_vars1].values])
    model1 = sm.OLS(d1[outcome_var].values, X1_ss).fit(cov_type='HC1')
    if debug:
        print(model1.summary())
    answer1 = model1.params[1]
    std_err1 = model1.bse[1]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars1, instrument_var=instrument_var, is_rct=True)

    ## Solution 2: IV effect of JTPA training on women's earnings
    d2 = female_df.dropna(subset=[outcome_var, treat_var, instrument_var] + control_vars2)
    O2 = np.ones(len(d2))
    X2_fs = np.column_stack([O2, d2[instrument_var].values, d2[control_vars2].values])
    first2 = sm.OLS(d2[treat_var].values, X2_fs).fit()
    D_hat2 = first2.predict(X2_fs)
    X2_ss = np.column_stack([O2, D_hat2, d2[control_vars2].values])
    model2 = sm.OLS(d2[outcome_var].values, X2_ss).fit(cov_type='HC1')
    if debug:
        print(model2.summary())
    answer2 = model2.params[1]
    std_err2 = model2.bse[1]
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         answer2, std_err2, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars2, instrument_var=instrument_var, is_rct=True)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper59(debug=False):
    """
    Builds the representation of paper 59

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Instrumental variables estimates of the effect of subsidized training on the quantiles of trainee earnings"
    dataset_name = "abadie_instrumental_variables"
    year = 1999
    domain = "labor economics"
    is_rct = True
    n_solutions = 2

    query1 = "What is the effect of receiving subsidized training on 30-month income of men?"
    query2 = "What is the effect of receiving subsidized training on 30-month income of women?"

    solutions = replicated_paper59(title, dataset_name,
                                   [query1, query2],
                                   [103, 104], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions,
                  is_rct=is_rct, n_solutions=n_solutions)
    return paper

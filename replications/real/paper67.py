## This file contains the replication of paper 67
from pathlib import Path
import statsmodels.api as sm
from linearmodels import IV2SLS
import numpy as np
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 67

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper67(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 67
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv", encoding='utf-8')
    df['inflation'] = np.log(df['inflation'].replace([0, np.inf, -np.inf], np.nan))

    needed = ['approval', 'lagged_dism_tainted', 'lagged_nonpartisan', 'lagged_age',
        'lagged_rc', 'visibility', 'allhouse', 'coalition',
        'reelect_last_year', 'govfrac', 'gdp', 'inflation',
        'honeymoon_2q', 'honeymoon_3q', 'preelection_1q', 'preelection_2q']
    iv_data = df.dropna(subset=needed).copy()
    iv_data['lagged_age_sq'] = iv_data['lagged_age'] ** 2
    iv_data['lagged_rc_sq']  = iv_data['lagged_rc'] ** 2

    treat_var = 'lagged_dism_tainted'
    outcome_var = 'approval'
    instrument_var = 'lagged_nonpartisan, lagged_age, lagged_age_sq'
    control_vars = ['lagged_rc', 'lagged_rc_sq', 'visibility', 'allhouse', 'coalition',
        'reelect_last_year', 'govfrac', 'gdp', 'inflation',
        'honeymoon_2q', 'honeymoon_3q', 'preelection_1q', 'preelection_2q']
    method = 'iv'

    y    = iv_data[outcome_var]
    endog = iv_data[[treat_var]]
    instr = iv_data[['lagged_nonpartisan', 'lagged_age', 'lagged_age_sq']]
    exog  = sm.add_constant(iv_data[['lagged_rc', 'lagged_rc_sq', 'visibility', 'allhouse',
        'coalition', 'reelect_last_year', 'govfrac', 'gdp', 'inflation',
        'honeymoon_2q', 'honeymoon_3q', 'preelection_1q', 'preelection_2q']])
    model1 = IV2SLS(y, exog, endog, instr).fit(cov_type='robust')
    if debug:
        print(model1.summary)
    answer1  = model1.params[treat_var]
    std_err1 = model1.std_errors[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var,
                         is_rct=False)

    return {id_li[0]: solution1}


def build_paper67(debug=False):
    """
    Builds the representation of paper 67

    Returns:
        (Paper): The constructed Paper object
    """

    title = "A blame shifting in presidential systems: Ministerial terminations corrective effect on approval"
    dataset_name = "gonzalez_blame_shifting"
    year = 2025
    domain = "government"
    n_solutions = 1

    query1 = "How does dismissing tainted ministers affect presidential approval?"

    solutions = replicated_paper67(title, dataset_name, [query1], [113], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    
    return paper

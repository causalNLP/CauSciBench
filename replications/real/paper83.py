## This file contains the replication of paper 83
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 83

BASE_DIR = Path("../data/csv_files/realdata/")


def _run_iv(dataset_name, outcome_raw, suffix):
    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df['ln_citylob']   = np.log(df['city_lobbying'] + 1)
    df['ln_countylob'] = np.log(df['county_lobbying'] + 1)
    df[f'pop_{suffix}'] = df[f'pop_{suffix}'] / 1000
    df['state']  = df['state'].astype('category')
    df['state2'] = df['state'].cat.codes
    df[f'{outcome_raw}_scaled'] = df[outcome_raw] * 1000
    df[f'ln_{outcome_raw}'] = np.log(df[f'{outcome_raw}_scaled'] + 1)

    s = suffix
    exog_X_vars = (
        [f'pop_{s}', f'land_{s}', f'water_{s}', f'senior_{s}', f'student_{s}',
         f'ethnic_{s}', f'mincome_{s}', f'unemp_{s}', f'poverty_{s}', f'gini_{s}'] +
        [f'city_propertytaxshare_{s}', f'city_intgovrevenueshare_{s}', f'city_airexp_{s}'] +
        [f'houdem_{s}', 'ln_countylob']
    )
    instrument_vars = ['direct_flight_dc', f'diverge2_{s}']
    outcome_var = f'ln_{outcome_raw}'
    formula = (f'{outcome_var} ~ 1 + state + {" + ".join(exog_X_vars)} + '
               f'[ln_citylob ~ {" + ".join(instrument_vars)}]')
    needed = list(set(instrument_vars + exog_X_vars + [outcome_var, 'ln_citylob', 'state']))
    iv_data = df[needed].dropna()
    res = IV2SLS.from_formula(formula, data=iv_data).fit(
        cov_type='clustered', clusters=df.loc[iv_data.index, 'state2'])
    return res, exog_X_vars, f'direct_flight_dc, diverge2_{s}', outcome_var


def replicated_paper83(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 83
    """

    treat_var = 'ln_citylob'
    method = 'iv'

    ## Solution 1: IV effect of city lobbying on recovery act grants
    ds_recovery = f"{dataset_name}_recovery"
    res1, ctrl1, instr1, ov1 = _run_iv(ds_recovery, 'recovery', 'r')
    if debug:
        print(res1.summary)
    solution1 = Solution(id_li[0], title, query_li[0], method, ds_recovery,
                         float(res1.params[treat_var]), float(res1.std_errors[treat_var]),
                         treat_var=treat_var, outcome_var=ov1,
                         control_vars=ctrl1, instrument_var=instr1,
                         state_var='state', is_rct=False)

    ## Solution 2: IV effect of city lobbying on earmarks
    ds_earmark = f"{dataset_name}_earmark"
    res2, ctrl2, instr2, ov2 = _run_iv(ds_earmark, 'earmark', 'e')
    if debug:
        print(res2.summary)
    solution2 = Solution(id_li[1], title, query_li[1], method, ds_earmark,
                         float(res2.params[treat_var]), float(res2.std_errors[treat_var]),
                         treat_var=treat_var, outcome_var=ov2,
                         control_vars=ctrl2, instrument_var=instr2,
                         state_var='state', is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper83(debug=False):
    """
    Builds the representation of paper 83

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Cities as lobbyists"
    dataset_name = "goldstein_lobbyists"
    year = 2017
    domain = "political science"
    n_solutions = 2

    query1 = "How much does the money spent on lobbying increase the received amount of recovery act grants?"
    query2 = "How much does the money spent on lobbying increase the number of earmarks received?"

    solutions = replicated_paper83(title, dataset_name, [query1, query2], [143, 144], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

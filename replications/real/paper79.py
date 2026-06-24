## This file contains the replication of paper 79
from pathlib import Path
from statsmodels.regression.linear_model import OLS
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 79

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper79(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 79
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df['fips'] = df['fips'].astype('category')
    df['year'] = df['year'].astype('category')

    treat_var = 'tl_onset'
    outcome_var = 'exp_real_pc'
    state_var = 'fips'
    time_var = 'year'
    control_vars = ['population', 'popgrowth', 'populationdensity', 'grants_real_pc',
        'gsp_real_inc', 'fedempciv_pc', 'fedempmil_pc', 'govtempsandl_pc',
        'seats_house', 'seats_senate', 'unemployment', 'gopshare', 'demcontrol',
        'dividedgov', 'squire_score', 'initiatives', 'tel', 'debtlimit']
    method = 'did'

    ## Solution 1: DiD effect of term limit onset on state real per capita expenditures
    formula1 = ('exp_real_pc ~ tl_onset + population + popgrowth + populationdensity + '
                'grants_real_pc + gsp_real_inc + fedempciv_pc + fedempmil_pc + '
                'govtempsandl_pc + seats_house + seats_senate + unemployment + gopshare + '
                'demcontrol + dividedgov + squire_score + initiatives + tel + debtlimit + '
                'fips + year')
    model1 = OLS.from_formula(formula1, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['fips']})
    if debug:
        print(model1.summary())
    answer1  = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         is_rct=False)

    return {id_li[0]: solution1}


def build_paper79(debug=False):
    """
    Builds the representation of paper 79

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Do term limits restrain state fiscal policy? Approaches for causal inference in assessing the effects of legislative institutions"
    dataset_name = "keele_do_term_limits"
    year = 2013
    domain = "political economics"
    n_solutions = 1

    query1 = "What is the effect of term limits on state expenditure per capita?"

    solutions = replicated_paper79(title, dataset_name, [query1], [127], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

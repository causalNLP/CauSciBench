## This file contains the replication of paper 76
from pathlib import Path
from statsmodels.discrete.discrete_model import Poisson
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 76

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper76(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 76
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    cols_needed = ['appeals_count', 'radical_islamist', 'secessionist', 'left_wing',
        'multi_ethnic', 'any_nr_binary', 'supporttype_funds_rebel', 'group_age',
        'reb_strength_ord', 'terr_cont', 'other_group_binary', 'post_2003',
        'v2x_polyarchy', 'mean_log10_gdp_pc', 'ccode']
    df = df.dropna(subset=cols_needed)
    df['ccode_int'] = pd.factorize(df['ccode'])[0]

    treat_var = 'radical_islamist'
    outcome_var = 'appeals_count'
    control_vars = ['secessionist', 'left_wing', 'multi_ethnic', 'any_nr_binary',
        'supporttype_funds_rebel', 'group_age', 'reb_strength_ord', 'terr_cont',
        'other_group_binary', 'post_2003', 'v2x_polyarchy', 'mean_log10_gdp_pc']
    method = 'ols'

    ## Solution 1: Poisson effect of radical Islamist ideology on recruitment appeal count
    formula1 = ('appeals_count ~ radical_islamist + secessionist + left_wing + multi_ethnic + '
                'any_nr_binary + supporttype_funds_rebel + group_age + reb_strength_ord + '
                'terr_cont + other_group_binary + post_2003 + v2x_polyarchy + mean_log10_gdp_pc')
    model1 = Poisson.from_formula(formula1, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['ccode_int']}, disp=False)
    if debug:
        print(model1.summary())
    answer1  = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper76(debug=False):
    """
    Builds the representation of paper 76

    Returns:
        (Paper): The constructed Paper object
    """

    title = "A Call to Arms: How Rebel Groups Choose Their Recruitment Appeals"
    dataset_name = "soules_call_to"
    year = 2025
    domain = "political science"
    n_solutions = 1

    query1 = "Does being a radical Islamist group causally increase the number of recruitment appeals they make?"

    solutions = replicated_paper76(title, dataset_name, [query1], [124], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

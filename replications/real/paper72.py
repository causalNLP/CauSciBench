## This file contains the replication of paper 72
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 72

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper72(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 72
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = 'SC8609_ma'
    outcome_var = 'prison_BWratio'
    vars1 = ['prison_BWratio', 'SC8609_ma', 'racialdiversity', 'SC_RD', 'pop_pctblk',
             'VCRate_Total', 'PCRate_Total', 'govideo', 'totdempct', 'women_leg',
             'blk_leg', 'threestrikes', 'unemp', 'bwpovratio', 'eduattain_ma',
             'bwcolratio', 'divorcerate', 'felonspc', 'statename']
    control_vars = ['VCRate_Total', 'PCRate_Total', 'divorcerate', 'women_leg',
        'pop_pctblk', 'felonspc', 'bwpovratio', 'bwcolratio', 'unemp',
        'racialdiversity', 'govideo', 'threestrikes', 'blk_leg', 'totdempct',
        'eduattain_ma', 'SC_RD']
    method = 'did'

    df1 = df[vars1].dropna()

    ## Solution 1: pooled OLS with clustered SE (equivalent to Stata xtpcse) — effect of social capital on Black-White incarceration ratio
    formula1 = ('prison_BWratio ~ SC8609_ma + racialdiversity + SC_RD + pop_pctblk + '
                'VCRate_Total + PCRate_Total + govideo + totdempct + women_leg + '
                'blk_leg + threestrikes + unemp + bwpovratio + eduattain_ma + '
                'bwcolratio + divorcerate + felonspc')
    model1 = smf.ols(formula=formula1, data=df1).fit(
        cov_type='cluster', cov_kwds={'groups': df1['statename']})
    if debug:
        print(model1.summary())
    answer1 = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var='statename', time_var='year', is_rct=False)

    return {id_li[0]: solution1}


def build_paper72(debug=False):
    """
    Builds the representation of paper 72

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Social capital, racial context, and incarcerations in the American states"
    dataset_name = "hawes_social_capital"
    year = 2017
    domain = "criminology"
    n_solutions = 1

    query1 = "Does an increase in social capital increase the ratio of Black/White prisoners, while accounting for other relevant factors?"

    solutions = replicated_paper72(title, dataset_name, [query1], [119], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

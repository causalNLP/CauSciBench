## This file contains the replication of paper 73
from pathlib import Path
from linearmodels import PanelOLS
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 73

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper73(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 73
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    model_df = df.set_index(['statename', 'year'])

    treat_var = 'SC8609_ma'
    outcome_var = 'tanfbenefit'
    control_vars = ['pctfb_ma', 'tanfcaseloadK', 'avgnbrtanf', 'unmarried_births',
        'pov_rtfull', 'cit_ideology', 'totdempct', 'hvd_4yr', 'women_leg',
        'unemp', 'adj_GSPpcK', 'adj_incpcK', 'pcturban_MA', 'high_school',
        'pop_pcthisp', 'pop_pctblk']
    method = 'did'

    ## Solution 1: panel FE effect of social capital on welfare generosity
    formula1 = ('tanfbenefit ~ SC8609_ma + pctfb_ma + SC8609_ma:pctfb_ma + tanfcaseloadK + '
                'avgnbrtanf + unmarried_births + pov_rtfull + cit_ideology + totdempct + '
                'hvd_4yr + women_leg + unemp + adj_GSPpcK + adj_incpcK + pcturban_MA + '
                'high_school + pop_pcthisp + pop_pctblk')
    model1 = PanelOLS.from_formula(formula=formula1, data=model_df).fit(
        cov_type='clustered', cluster_entity=True)
    if debug:
        print(model1.summary)
    answer1  = model1.params[treat_var]
    std_err1 = model1.std_errors[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var='statename', time_var='year', is_rct=False)

    return {id_li[0]: solution1}


def build_paper73(debug=False):
    """
    Builds the representation of paper 73

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Give us your tired, your poor and we might buy them dinner: Social capital, immigration, and welfare generosity in the American states"
    dataset_name = "hawes_give_us"
    year = 2018
    domain = "political science"
    n_solutions = 1

    query1 = "How does social capital influence the maximum TANF benefit a family of three can claim when accounting for immigration?"

    solutions = replicated_paper73(title, dataset_name, [query1], [120], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

## This file contains the replication of paper 66
from pathlib import Path
from linearmodels.iv import IV2SLS
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 66

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper66(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 66
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    needed = ['gini', 'corr', 'al_ethnic', 'al_language', 'al_religion', 'unemptot',
        'ln_gdpcapcons', 'educprim', 'educsec', 'taxrev', 'p_democ', 'healthtot',
        'fdi', 'gov', 'cap', 'popgrow']
    data = df.dropna(subset=needed)

    treat_var = 'corr'
    outcome_var = 'gini'
    instrument_var = 'al_ethnic, al_language, al_religion'
    control_vars = ['unemptot', 'ln_gdpcapcons', 'educprim', 'educsec', 'taxrev',
        'p_democ', 'healthtot', 'fdi', 'gov', 'cap', 'popgrow']
    method = 'iv'

    ## Solution 1: IV effect of corruption on income inequality (Column 1: 3 instruments, N=83)
    formula1 = ('gini ~ [corr ~ al_ethnic + al_language + al_religion] + unemptot + '
                'ln_gdpcapcons + educprim + educsec + taxrev + p_democ + healthtot + '
                'fdi + gov + cap + popgrow')
    model1 = IV2SLS.from_formula(formula1, data=data).fit(cov_type='unadjusted')
    if debug:
        print(model1.summary)
    answer1 = model1.params[treat_var]
    std_err1 = model1.std_errors[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, instrument_var=instrument_var,
                         is_rct=False)

    return {id_li[0]: solution1}


def build_paper66(debug=False):
    """
    Builds the representation of paper 66

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The corruption-income inequality trap: A study of Asian countries"
    dataset_name = "dwiputri_the_corruption"
    year = 2018
    domain = "political economics"
    n_solutions = 1

    query1 = "Does an increase in corruption lead to higher income inequality?"

    solutions = replicated_paper66(title, dataset_name, [query1], [112], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

## This file contains the replication of paper 62
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 62

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper62(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 62
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    df['Male'] = (df['Gender'] == 'Male').astype(int)

    outcome_var = 'Outcome'
    treat_var = 'Dist_nuclear'
    control_vars = ['Dist_nuclear', 'Dist_wind', 'Health', 'Job', 'Landscape', 'Supply',
        'Trust_reg', 'Trust_health', 'Trust_plant', 'Env', 'Urban_rural', 'Male',
        'Age', 'Income', 'Education', 'PID']
    method = 'ols'

    ## Solution 1: predictors of nuclear vs wind support
    formula1 = ("Outcome ~ Dist_nuclear + Dist_wind + Health + Job + Landscape + Supply + "
                "Trust_reg + Trust_health + Trust_plant + Env + Urban_rural + Male + "
                "C(Age, Treatment('Before 1946')) + C(Income, Treatment('Less than $40,000')) + "
                "C(Education, Treatment('Middle or below')) + C(PID, Treatment('Republican'))")
    model1 = smf.ols(formula1, data=df).fit()
    if debug:
        print(model1.summary())
    answer1 = model1.params['Dist_nuclear']
    std_err1 = model1.bse['Dist_nuclear']
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper62(debug=False):
    """
    Builds the representation of paper 62

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Comparing public support for nuclear and wind energy in Washington state"
    dataset_name = "uji_comparing_public"
    year = 2023
    domain = "environmental science"
    n_solutions = 1

    query1 = "Does the distance to nuclear facilities influence the difference in a person's support for nuclear or wind energy?"

    solutions = replicated_paper62(title, dataset_name, [query1], [107], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

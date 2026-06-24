## This file contains the replication of paper 71
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 71

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper71(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 71
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = 'femaleimage:hispanicname'
    outcome_var = 'color'
    control_vars = ['hispanicname', 'femaleimage', 'rs_color', 'age', 'gender',
                    'race', 'hhinc', 'educ', 'region', 'hispanic', 'born']
    method = 'ols'

    ## Solution 1: interaction effect of Hispanic name × female face on perceived skin color
    formula1 = ('color ~ hispanicname + femaleimage + femaleimage:hispanicname + '
                'rs_color + age + gender + C(race, Treatment(1)) + hhinc + educ + '
                'C(region, Treatment(1)) + hispanic + born + C(imageno) + C(namepair)')
    model1 = smf.ols(formula1, data=df).fit()
    if debug:
        print(model1.summary())
    answer1  = model1.params['femaleimage:hispanicname']
    std_err1 = model1.bse['femaleimage:hispanicname']
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=True)

    return {id_li[0]: solution1}


def build_paper71(debug=False):
    """
    Builds the representation of paper 71

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Colored perceptions: Racially distinctive names and assessments of skin color"
    dataset_name = "garcia_colored_perceptions"
    year = 2016
    domain = "psychology"
    n_solutions = 1

    query1 = "Do people consider that the skin color of hispanic people is darker when the person presented is a woman?"

    solutions = replicated_paper71(title, dataset_name, [query1], [118], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

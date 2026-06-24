## This file contains the replication of paper 75
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 75

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper75(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 75
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = 'aligned'
    outcome_var = 'remove'
    control_vars = ['age', 'gender', 'education', 'household_income', 'race',
        'hispanic', 'accuracy_order', 'headline_order', 'social_media_post_removed',
        'social_media_post_flagged', 'social_media_most_common_newsformat', 'political_interest']
    method = 'ols'

    ## Solution 1: effect of partisan alignment on content removal preferences
    formula1 = ('remove ~ aligned + age + gender + C(education) + household_income + '
                'C(race) + hispanic + accuracy_order + headline_order + '
                'social_media_post_removed + social_media_post_flagged + '
                'social_media_most_common_newsformat + political_interest')
    model1 = smf.ols(formula=formula1, data=df).fit()
    if debug:
        print(model1.summary())
    answer1  = model1.params[treat_var]
    std_err1 = model1.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper75(debug=False):
    """
    Builds the representation of paper 75

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Partisan conflict over content moderation is more than disagreement about facts"
    dataset_name = "ruth_partisan_conflict"
    year = 2023
    domain = "political science"
    n_solutions = 1

    query1 = "Does the headline being aligned with a participant's party increase the likelihood that they support its removal?"

    solutions = replicated_paper75(title, dataset_name, [query1], [123], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

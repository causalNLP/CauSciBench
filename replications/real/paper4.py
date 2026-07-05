## This file contains the replication of paper 4
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 4

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper4(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 4
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0)

    outcome_var = "invited"
    method = "ols"

    ## Solution 1: overall immigrant effect
    treatment_var1 = "immigrant"
    control_vars1 = ["citizen", "woman", "religious", "experience", "skilledjob", "time"]
    formula1 = "invited ~ immigrant + citizen + woman + religious + experience + skilledjob + time + C(stad)"
    model1 = smf.ols(formula1, data=df).fit(cov_type="HC3")
    if debug:
        print(model1.summary())
    answer1 = model1.params[treatment_var1]
    std_err1 = model1.bse[treatment_var1]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treatment_var1, outcome_var=outcome_var,
                         control_vars=control_vars1, state_var="stad", is_rct=True)

    ## Solution 2: country-of-origin effects
    treatment_var2 = "somalia"
    control_vars2 = ["poland", "iraq", "citizen", "woman", "religious", "experience", "skilledjob", "time"]
    formula2 = "invited ~ somalia + poland + iraq + citizen + woman + religious + experience + skilledjob + time + C(stad)"
    model2 = smf.ols(formula2, data=df).fit(cov_type="HC3")
    if debug:
        print(model2.summary())
    answer2 = model2.params[treatment_var2]
    std_err2 = model2.bse[treatment_var2]
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         answer2, std_err2, treat_var=treatment_var2, outcome_var=outcome_var,
                         control_vars=control_vars2, state_var="stad", is_rct=True)

    solution_dict = {id_li[0]: solution1, id_li[1]: solution2}

    return solution_dict


def build_paper4(debug=False):
    """
    Builds the representation of paper 4

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Can immigrants counteract employer discrimination? A factorial field experiment reveals the immutability of ethnic hierarchies"
    dataset_name = "vernby_can_immigrants"
    year = 2019
    domain = "labor economics"
    is_rct = True
    is_multirct = False
    n_solutions = 2

    query1 = "Does being an immigrant make it less likely to get an interview request?"
    query2 = "How does being born in Somalia, compared to other countries, affect a candidate's chances of getting a job interview?"

    solutions = replicated_paper4(title, dataset_name, [query1, query2], [7, 8], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, is_multirct,
                  is_rct, n_solutions)
    return paper

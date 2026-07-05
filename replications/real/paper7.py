## This file contains the replication of paper 7
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 7

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper7(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 7
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0)

    treat_var = "repeal"
    outcome_var = "lnr"
    control_vars = ["acc", "ir", "pi", "alcohol", "crack", "poverty", "income", "ur"]
    method = "did"
    state_var = "fip"
    time_var = "year"


    formula = f"{outcome_var} ~ {treat_var} + C({state_var}) + C({time_var}) + {' + '.join(control_vars)}"

    ## Solution 1: females
    df_female = df[df["female"] == 1]
    model_female = smf.wls(formula, data=df_female, weights=df_female["totpop"].values).fit(
        cov_type="cluster", cov_kwds={"groups": df_female["fip"].values}, method="pinv")
    if debug:
        print(model_female.summary())
    answer1 = model_female.params[treat_var]
    std_err1 = model_female.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         canonical_did=False, is_rct=False)

    ## Solution 2: males
    df_male = df[df["female"] == 0]
    model_male = smf.wls(formula, data=df_male, weights=df_male["totpop"].values).fit(
        cov_type="cluster", cov_kwds={"groups": df_male["fip"].values}, method="pinv")
    if debug:
        print(model_male.summary())
    answer2 = model_male.params[treat_var]
    std_err2 = model_male.bse[treat_var]
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         answer2, std_err2, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var=state_var, time_var=time_var,
                         canonical_did=False, is_rct=False)

    solution_dict = {id_li[0]: solution1, id_li[1]: solution2}

    return solution_dict


def build_paper7(debug=False):
    """
    Builds the representation of paper 7

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The long-run effect of abortion on sexually transmitted infections"
    dataset_name = "cunningham_the_long_run"
    year = 2013
    domain = "health economics"
    n_solutions = 2

    query1 = "Is there a difference in gonorrhea rates between early repeal and Roe states among Black females?"
    query2 = "Did abortion legalization before Roe v. Wade reduce gonorrhea rates among Black male teenagers in early-repeal states compared to other states?"

    solutions = replicated_paper7(title, dataset_name, [query1, query2], [11, 12], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions,  n_solutions=n_solutions)
    return paper

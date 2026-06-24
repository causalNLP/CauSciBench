## This file contains the replication of paper 13
from pathlib import Path
import statsmodels.api as sm
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 13

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper13(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 13
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path)

    df["demvoteshare_c"] = df["demvoteshare"] - 0.5
    df = df[~pd.isnull(df["demvoteshare_c"])]
    df["demvoteshare_sq"] = df["demvoteshare_c"] ** 2
    df = df[df["demvoteshare"].between(0.45, 0.55)]

    method = "rdd"
    running_var = "demvoteshare_c"

    ## Solution 1: effect of winning election on ADA score (lm_2 in reference)
    treat_var1 = "democrat"
    outcome_var1 = "score"

    df1 = df[[outcome_var1, treat_var1, "demvoteshare_c", "demvoteshare_sq", "id"]].dropna()
    formula1 = "score ~ democrat*demvoteshare_c + democrat*demvoteshare_sq"
    model1 = sm.OLS.from_formula(formula1, data=df1).fit(
        cov_type="cluster", cov_kwds={"groups": df1["id"]})
    if debug:
        print(model1.summary())
    answer1 = model1.params[treat_var1]
    std_err1 = model1.bse[treat_var1]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var1, outcome_var=outcome_var1,
                         running_var=running_var, is_rct=False)

    ## Solution 2: effect of lagged win on current democrat status (lm_3 in reference)
    treat_var2 = "lagdemocrat"
    outcome_var2 = "democrat"

    df2 = df[[outcome_var2, treat_var2, "demvoteshare_c", "demvoteshare_sq", "id"]].dropna()
    formula2 = "democrat ~ lagdemocrat*demvoteshare_c + lagdemocrat*demvoteshare_sq"
    model2 = sm.OLS.from_formula(formula2, data=df2).fit(
        cov_type="cluster", cov_kwds={"groups": df2["id"]})
    if debug:
        print(model2.summary())
    answer2 = model2.params[treat_var2]
    std_err2 = model2.bse[treat_var2]
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         answer2, std_err2, treat_var=treat_var2, outcome_var=outcome_var2,
                         running_var=running_var, is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper13(debug=False):
    """
    Builds the representation of paper 13

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Do voters affect or elect policies: Evidence from the U.S. House"
    dataset_name = "lee_do_voters"
    year = 2004
    domain = "political economics"
    n_solutions = 2

    query1 = "In close congressional elections, does winning the election as a Democrat lead to more liberal legislative voting behavior?"
    query2 = "If a Democrat won the previous congressional election in a district closely, is it likely that a Democrat will win the current election in that same district?"

    solutions = replicated_paper13(title, dataset_name, [query1, query2], [22, 23], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

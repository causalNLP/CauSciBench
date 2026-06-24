## This file contains the replication of paper 11
from pathlib import Path
import statsmodels.formula.api as smf
import pandas as pd
from solution import Solution, Paper

PAPER_ID = 11

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper11(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 11
    """

    file_path = BASE_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(file_path, index_col=0).reset_index()

    treat_var = "post_california"
    outcome_var = "rate"
    state_var = "state"
    time_var = "quarter_num"
    method = "did"

    df["post"] = (df["quarter_num"] > 3).astype(int)
    df["california"] = (df["state"] == "California").astype(int)
    df["post_california"] = df["post"] * df["california"]

    model = smf.ols("rate ~ post + california + post_california",
                    data=df).fit(cov_type="cluster", cov_kwds={"groups": df["state"]})
    if debug:
        print(model.summary())

    answer1 = model.params[treat_var]
    std_err1 = model.bse[treat_var]
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer1, std_err1, treat_var=treat_var, outcome_var=outcome_var,
                         state_var=state_var, time_var=time_var,
                         canonical_did=True, is_rct=False)

    return {id_li[0]: solution1}


def build_paper11(debug=False):
    """
    Builds the representation of paper 11

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Dont take no for an answer: An experiment with actual organ donor registrations"
    dataset_name = "kessler_dont_take"
    year = 2014
    domain = "health economics"
    n_solutions = 1

    query1 = "Did changing California's organ donation registration from opt-in to active choice in July 2011 (3rd quarter) increase donation rates compared to states that kept their existing systems?"

    solutions = replicated_paper11(title, dataset_name, [query1], [20], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

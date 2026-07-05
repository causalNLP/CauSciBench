## This file contains the replication of paper 39
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 39

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper39(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 39
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "treatment"
    outcome_var = "sb24"
    control_vars = ["flt", "full_co"]
    method = "glm"

    q2med = df["q2full"].median()
    df["full_co"] = (df["q2full"] <= q2med).astype(int)
    df["flt"] = df["treatment"] * df["full_co"]

    model_cols = [outcome_var, treat_var, "flt", "full_co", "match_category"]
    data = df.dropna(subset=model_cols)

    formula = f"{outcome_var} ~ {treat_var} + flt + full_co"
    model = smf.glm(formula=formula, data=data,
                    family=sm.families.Binomial(link=sm.families.links.Probit())).fit(
        cov_type="cluster", cov_kwds={"groups": data["match_category"]})

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=True)

    return {id_li[0]: solution1}


def build_paper39(debug=False):
    """
    Builds the representation of paper 39

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Can learning constituency opinion affect how legislators vote? Results from a field experiment"
    dataset_name = "butler_can_learning"
    year = 2011
    domain = "political science"
    n_solutions = 1

    query1 = "In districts with above-median support for the funding proposal, does receiving a letter about constituent opinion affect how legislators vote on the bill?"

    solutions = replicated_paper39(title, dataset_name, [query1], [68], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

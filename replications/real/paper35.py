## This file contains the replication of paper 35
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 35

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper35(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 35
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "HispanicPct_Year2017"
    outcome_var = "TotalCrimeIndex"
    state_var = "fips_state_county"
    time_var = "year"
    method = "did"

    data = df[df["year"] >= 2016].copy()
    data["Year2017"] = (data["year"] >= 2017).astype(int)
    data["HispanicPct_Year2017"] = data["HispanicPct"] * data["Year2017"]
    data = data.dropna(subset=["TotalCrimeIndex", "PropertyCrimeIndex", "ViolentCrimeIndex",
                                "HispanicPct", "fips_state_county", "year"])

    formula = f"{outcome_var} ~ {treat_var} + C({state_var}) + C({time_var})"
    model = smf.ols(formula, data=data).fit()

    if debug:
        print(model.summary())

    answer = model.params[treat_var]
    std_err = model.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         state_var=state_var, time_var=time_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper35(debug=False):
    """
    Builds the representation of paper 35

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Immigration policies and access to the justice system: The effect of enforcement escalations on undocumented immigrants and their communities"
    dataset_name = "dhingra_immigration_policies"
    year = 2021
    domain = "criminology"
    n_solutions = 1

    query1 = "What is the effect of a country's Hispanic population share on the change in total reported crime in 2017?"

    solutions = replicated_paper35(title, dataset_name, [query1], [56], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

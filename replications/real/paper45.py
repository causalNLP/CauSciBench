## This file contains the replication of paper 45
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 45

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper45(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 45
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df = df[["spp", "treatment_postp", "postp", "codi"]].dropna()

    treat_var = "treatment_postp"
    outcome_var = "spp"
    method = "did"
    state_var = "codi"
    time_var = "year"

    res = smf.ols("spp ~ treatment_postp + postp + C(codi)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["codi"]})

    if debug:
        print(res.summary())

    answer = res.params[treat_var]
    std_err = res.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         state_var=state_var, time_var=time_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper45(debug=False):
    """
    Builds the representation of paper 45

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Incumbents beware: The impact of offshoring on elections"
    dataset_name = "rickard_incumbents"
    year = 2021
    domain = "political science"
    n_solutions = 1

    query1 = "Does experiencing an offshoring event cause a reduction in the incumbent government party's (PP) vote share in Catalan municipalities between the 2000 and 2004 elections?"

    solutions = replicated_paper45(title, dataset_name, [query1], [83], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    
    return paper

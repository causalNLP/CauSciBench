## This file contains the replication of paper 47
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 47

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper47(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 47
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    df = df[(df["duration"] > 150) & (df["both_waves"] != 0)].copy()

    treat_var = "vaccine"
    method = "did"
    state_var = "respondent_code"
    time_var = "wave"

    outcome_vars = ["china_score", "usa_score"]
    solutions = {}

    for i, outcome_var in enumerate(outcome_vars):
        data = df[[outcome_var, treat_var, "respondent_code", "wave"]].dropna()

        res = smf.ols(
            f"{outcome_var} ~ vaccine + C(respondent_code) + C(wave)", data=data).fit(cov_type="cluster", 
                                                                                      cov_kwds={"groups": data["respondent_code"]})

        if debug:
            print(f"{outcome_var}: {res.params[treat_var]:.4f} ({res.bse[treat_var]:.4f})")

        solutions[id_li[i]] = Solution(id_li[i], title, query_li[i], method, dataset_name,
                                       res.params[treat_var], res.bse[treat_var],
                                       treat_var=treat_var, outcome_var=outcome_var,
                                       state_var=state_var, time_var=time_var,
                                       is_rct=False)

    return solutions


def build_paper47(debug=False):
    """
    Builds the representation of paper 47

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Undermining U.S. Reputation: Chinese vaccines and aid and the alternative provision of public goods during covid‑19"
    dataset_name = "urdinez_undermining"
    year = 2024
    domain = "political science"
    n_solutions = 2

    query1 = "Does receiving a Chinese COVID-19 vaccine cause an increase in an individual's favorability toward China?"
    query2 = "Does receiving a Chinese COVID-19 vaccine cause a change in an individual's opinion of the United States?"

    solutions = replicated_paper47(title, dataset_name, [query1, query2], [85, 86], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

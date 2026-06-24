## This file contains the replication of paper 50
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS
from solution import Solution, Paper

PAPER_ID = 50

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper50(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 50
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    for col in ["trustNHS", "cancerDelay", "emergencyDelay", "women", "income", "age",
                "ethnicity", "higherEd", "conVote19", "unemploymentRate", "conShare", "region", "W8"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["region"])
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True).astype(float)
    data = pd.concat([df, region_dummies], axis=1).dropna()

    treat_var = "cancerDelay"
    outcome_var = "trustNHS"
    control_vars = ["emergencyDelay", "women", "income", "age", "ethnicity", "higherEd",
                    "conVote19", "unemploymentRate", "conShare"] + list(region_dummies.columns)
    method = "ols"

    X = sm.add_constant(data[[treat_var] + control_vars].astype(float))
    res = WLS(data[outcome_var].astype(float), X, weights=data["W8"].astype(float)).fit()

    if debug:
        print(res.summary())

    answer = res.params[treat_var]
    std_err = res.bse[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, is_rct=False)

    return {id_li[0]: solution1}


def build_paper50(debug=False):
    """
    Builds the representation of paper 50

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The influence of waiting times and sociopolitical variables on public trust in healthcare: A cross-sectional study of the NHS in England"
    dataset_name = "dorussen_the_influence"
    year = 2024
    domain = "public health"
    n_solutions = 1

    query1 = "How does the local proportion of GP‑to‑specialist cancer referrals that breach the two‑week wait standard affect individuals' reported trust in the NHS on a 1–7 scale?"

    solutions = replicated_paper50(title, dataset_name, [query1], [89], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    
    return paper

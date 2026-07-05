## This file contains the replication of paper 53
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from solution import Solution, Paper

PAPER_ID = 53

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper53(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 53
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    needed = ["dep_dummy_200", "tbill_l1_1000", "total_mills", "pre", "parcelnumber", "tax_year"]
    df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

    treat_var = "tbill_l1_1000"
    outcome_var = "dep_dummy_200"
    control_vars = ["total_mills", "pre"]
    method = "ols"

    df = df.sort_values(["parcelnumber", "tax_year"]).set_index(["parcelnumber", "tax_year"])

    y = df[outcome_var]
    X = sm.add_constant(df[[treat_var] + control_vars])

    res = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)

    if debug:
        print(res.summary)

    answer = res.params[treat_var]
    std_err = res.std_errors[treat_var]

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         answer, std_err, treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=control_vars, state_var="parcelnumber",
                         time_var="tax_year", is_rct=False)

    return {id_li[0]: solution1}


def build_paper53(debug=False):
    """
    Builds the representation of paper 53

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The effect of property assessment reductions on tax delinquency and tax foreclosure"
    dataset_name = "alfaro_the_effect"
    year = 2025
    domain = "public economics"
    n_solutions = 1

    query1 = "How does a $1,000 increase in last year's property tax bill affect the probability that a residential parcel is tax‑delinquent this year?"

    solutions = replicated_paper53(title, dataset_name, [query1], [92], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

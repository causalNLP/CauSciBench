## This file contains the replication of paper 89
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from solution import Solution, Paper

PAPER_ID = 89

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper89(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 89
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")
    d = df.loc[df["dist_from_cut"].between(-0.6, 0.6)].copy()

    treat_var = "probation_year1"
    outcome_var = "left_school"
    running_var = "dist_from_cut"

    y = d[outcome_var]
    T = d[treat_var].astype(int)
    X = pd.DataFrame({"const": 1.0, treat_var: T, running_var: d[running_var]})

    ## Solution 1: RDD effect of academic probation on leaving after first year
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["clustervar"]})
    if debug:
        print(m.summary())
    solution1 = Solution(id_li[0], title, query_li[0], "rdd", dataset_name,
                         float(m.params[treat_var]), float(m.bse[treat_var]),
                         treat_var=treat_var, outcome_var=outcome_var,
                         control_vars=[running_var], running_var=running_var, is_rct=False)

    return {id_li[0]: solution1}


def build_paper89(debug=False):
    """
    Builds the representation of paper 89

    Returns:
        (Paper): The constructed Paper object
    """

    title = "Improving balance in regression discontinuity design by matching: Estimating the effect of academic probation after the first year of college"
    dataset_name = "chi_improving"
    year = 2014
    domain = "education"
    n_solutions = 1

    query1 = "What is the effect of being placed on academic probation after the first year on the probability that a student leaves school?"

    solutions = replicated_paper89(title, dataset_name, [query1], [152], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

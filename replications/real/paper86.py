## This file contains the replication of paper 86
from pathlib import Path
import numpy as np
import pandas as pd
from rdd.rdd import optimal_bandwidth
from linearmodels.iv import IV2SLS
from solution import Solution, Paper

PAPER_ID = 86

BASE_DIR = Path("../data/csv_files/realdata/")


def _kernelwts(X, center, bw):
    dist = (X - center) / bw
    w = 1 - np.abs(dist)
    w = np.maximum(0, w)
    w = w / np.sum(w)
    return w


def _run(df_full, analysis_type):
    data = df_full.copy().rename(columns={"er.v.c_l": "running_var_name", "multic.logit_fd": "y_name", "er.in_l": "seatId"})
    if analysis_type == 2:
        data["country.mean.rile.logit"] = data.groupby("iso2c")["rile.logit"].transform("mean")
        data["mean.rile.logit"] = data.groupby("party")["rile.logit"].transform("mean")
        data = data[data["mean.rile.logit"] < data["country.mean.rile.logit"]]
    data = data[["y_name", "running_var_name", "seatId", "iso2c", "party", "edate"]].dropna()
    bandwidth = optimal_bandwidth(data["y_name"], data["running_var_name"], 0)
    weights = _kernelwts(data["running_var_name"], 0, bandwidth)
    data = data[weights > 0].copy()
    weights = weights[weights > 0]
    for col in ["iso2c", "party"]:
        data[col] = data[col].astype("category")
    data["edate"] = data["edate"].astype("datetime64[ns]").dt.year
    data["instrument"] = (data["running_var_name"] >= 0).astype(int)
    data["running_var_name_above"] = np.maximum(0, data["running_var_name"])
    formula = "y_name ~ poly(running_var_name, 1) + poly(running_var_name_above, 1) + iso2c + [seatId ~ instrument]"
    return IV2SLS.from_formula(formula, data=data, weights=weights).fit(cov_type='clustered', clusters=data[["party", "edate"]])


def replicated_paper86(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 86
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv")

    treat_var = "seatId"
    method = "rdd"

    ## Solution 1: effect on mainstream left parties
    res1 = _run(df, analysis_type=2)
    if debug:
        print(res1.summary)
    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         float(res1.params[treat_var]), float(res1.std_errors[treat_var]),
                         treat_var="er.in_l", outcome_var="multic.logit_fd",
                         running_var="er.v.c_l", is_rct=False)

    ## Solution 2: effect on all mainstream parties
    res2 = _run(df, analysis_type=1)
    if debug:
        print(res2.summary)
    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         float(res2.params[treat_var]), float(res2.std_errors[treat_var]),
                         treat_var="er.in_l", outcome_var="multic.logit_fd",
                         running_var="er.v.c_l", is_rct=False)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper86(debug=False):
    """
    Builds the representation of paper 86

    Returns:
        (Paper): The constructed Paper object
    """

    title = "The causal effect of radical right success on mainstream parties policy positions. A regression discontinuity approach"
    dataset_name = "abouchadi_causal_effect"
    year = 2020
    domain = "government"
    n_solutions = 2

    query1 = "What is the effect of a radical right party's representation in parliament on mainstream left parties' position change on cultural protectionism?"
    query2 = "What is the effect of a radical right party's representation in parliament on mainstream parties' position change on cultural protectionism?"

    solutions = replicated_paper86(title, dataset_name, [query1, query2], [148, 149], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper

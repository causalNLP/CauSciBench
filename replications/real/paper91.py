## This file contains the replication of paper 91
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from solution import Solution, Paper

PAPER_ID = 91

BASE_DIR = Path("../data/csv_files/realdata/")


def replicated_paper91(title, dataset_name, query_li, id_li, debug=False):
    """
    Replicates the analysis of paper 91
    """

    df = pd.read_csv(BASE_DIR / f"{dataset_name}.csv", encoding="latin-1")
    df["turnout"] = df["vote_PCC2012"].replace({3: 1, 1: 0, 2: np.nan})
    df["group"] = df["treatment"].map({1: "C1_exp", 2: "C2_hh", 3: "T1_high", 4: "T2_low"})

    def hh_type(s):
        if any(s == 3): return "HH_T1"
        if any(s == 4): return "HH_T2"
        if any(s == 1) and not any(s.isin([3, 4])): return "HH_Control"
        return "Other"

    df["hh_type"] = df.groupby("hh_id")["treatment"].transform(hh_type)
    spill = df[(df["group"] == "C2_hh") & (df["hh_type"].isin(["HH_Control", "HH_T1", "HH_T2"]))].copy()

    treat_var = "treated"
    outcome_var = "turnout"
    method = "ols"

    def _fit(sub, treated_tag):
        s = sub[sub["hh_type"].isin([treated_tag, "HH_Control"])].copy()
        s["treated"] = (s["hh_type"] == treated_tag).astype(int)
        m = smf.ols("turnout ~ treated", data=s).fit(cov_type="HC1")
        return float(m.params["treated"]), float(m.bse["treated"])

    ## Solution 1: spillover effect in low-partisan households (HH_T2 vs HH_Control)
    att_low, se_low = _fit(spill, "HH_T2")
    if debug:
        print(f"Low-partisan spillover: coef={att_low:.4f}, SE={se_low:.4f}")

    solution1 = Solution(id_li[0], title, query_li[0], method, dataset_name,
                         att_low, se_low, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    ## Solution 2: spillover effect in high-partisan households (HH_T1 vs HH_Control)
    att_high, se_high = _fit(spill, "HH_T1")
    if debug:
        print(f"High-partisan spillover: coef={att_high:.4f}, SE={se_high:.4f}")

    solution2 = Solution(id_li[1], title, query_li[1], method, dataset_name,
                         att_high, se_high, treat_var=treat_var, outcome_var=outcome_var, is_rct=True)

    return {id_li[0]: solution1, id_li[1]: solution2}


def build_paper91(debug=False):
    """
    Builds the representation of paper 91

    Returns:
        (Paper): The constructed Paper object
    """

    title = "All in the family: Partisan disagreement and electoral mobilization in intimate networks"
    dataset_name = "foos_all_in"
    year = 2017
    domain = "political science"
    n_solutions = 2

    query1 = "Among unassigned household members, what is the effect of assigning a low‑partisan message to the other household member on turnout?"
    query2 = "Among unassigned household members, what is the effect of assigning a high‑partisan message to the other household member on turnout?"

    solutions = replicated_paper91(title, dataset_name, [query1, query2], [155, 156], debug=debug)
    paper = Paper(PAPER_ID, title, dataset_name, year, domain, solutions, n_solutions=n_solutions)
    return paper
